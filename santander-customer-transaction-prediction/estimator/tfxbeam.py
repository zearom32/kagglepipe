from absl import app
from absl import flags

import os
import tensorflow as tf
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, Trainer
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input
from tfx.orchestration import pipeline
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

FLAGS = flags.FLAGS

def generate_pipeline(pipeline_name, pipeline_root, data_root, train_steps, eval_steps):
  examples = external_input(data_root)
  example_gen = CsvExampleGen(input=examples)
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=False)
  trainer = Trainer(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file='util.py', # util.py is a file in the same folder
      train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
      eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen, statistics_gen, schema_gen,
          trainer
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          os.path.join(pipeline_root, 'metadata.sqlite')))

def main(_):
  pipeline = generate_pipeline(
      flags.FLAGS.pipeline_name,
      flags.FLAGS.pipeline_root,
      flags.FLAGS.data_root,
      flags.FLAGS.train_steps,
      flags.FLAGS.eval_steps)

  BeamDagRunner().run(pipeline)

if __name__ == '__main__':
  flags.DEFINE_string(
      name="pipeline_name", default="santander",
      help="pipeline name used to identity different pipelines")
  flags.DEFINE_string(
      name="pipeline_root", default="/var/tmp/santander/pipeestimator/",
      help="pipeline root for storing artifacts")
  flags.DEFINE_string(
      name="data_root", default="/var/tmp/santander/data/train",
      help="Folder for Kaggle train.csv. No test.csv in the folder.")
  flags.DEFINE_integer(
      name="train_steps", default=10000,
      help="Steps to train a model")
  flags.DEFINE_integer(
      name="eval_steps", default=1000,
      help="Steps to train a model")

  app.run(main)