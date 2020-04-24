from absl import app
from absl import flags

import os
import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.components import CsvExampleGen, StatisticsGen, SchemaGen, Trainer, Transform, Evaluator, Pusher, ResolverNode
from tfx.components.base import executor_spec
from tfx.components.trainer.executor import GenericExecutor
from tfx.proto import pusher_pb2
from tfx.proto import trainer_pb2
from tfx.utils.dsl_utils import external_input
from tfx.orchestration import pipeline
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.dsl.experimental import latest_blessed_model_resolver
from tfx.types import Channel
from tfx.types.standard_artifacts import Model
from tfx.types.standard_artifacts import ModelBlessing
from hello_component import component

FLAGS = flags.FLAGS

def generate_pipeline(pipeline_name, pipeline_root, data_root, train_steps, eval_steps, pusher_target):
  module_file = 'util.py' # util.py is a file in the same folder
  examples = external_input(data_root)
  example_gen = CsvExampleGen(input=examples)
  hello = component.HelloComponent(
      input_data=example_gen.outputs['examples'], instance_name='HelloWorld')
  statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
  schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'],
      infer_feature_shape=True) # infer_feature_shape controls sparse or dense

  # Transform is too slow in my side.
  transform = Transform(
      examples=example_gen.outputs['examples'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file)
  
  trainer = Trainer(
      custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
      examples=transform.outputs['transformed_examples'],
      transform_graph=transform.outputs['transform_graph'],
      schema=schema_gen.outputs['schema'],
      module_file=module_file,
      train_args=trainer_pb2.TrainArgs(num_steps=train_steps),
      eval_args=trainer_pb2.EvalArgs(num_steps=eval_steps))

  # Get the latest blessed model for model validation.
  model_resolver = ResolverNode(
      instance_name='latest_blessed_model_resolver',
      resolver_class=latest_blessed_model_resolver.LatestBlessedModelResolver,
      model=Channel(type=Model),
      model_blessing=Channel(type=ModelBlessing))

  # Uses TFMA to compute a evaluation statistics over features of a model and
  # perform quality validation of a candidate model (compared to a baseline).
  eval_config = tfma.EvalConfig(
      model_specs=[tfma.ModelSpec(label_key='target')],
      slicing_specs=[
          tfma.SlicingSpec(),
          tfma.SlicingSpec(feature_keys=['var_0', 'var_1'])],
      metrics_specs=[
          tfma.MetricsSpec(
              thresholds={
                  'binary_accuracy':
                      tfma.config.MetricThreshold(
                          value_threshold=tfma.GenericValueThreshold(
                              lower_bound={'value': 0.4}), # always bless
                          change_threshold=tfma.GenericChangeThreshold(
                              direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                              absolute={'value': -1e-10}))
              })
      ])
  evaluator = Evaluator(
      examples=example_gen.outputs['examples'],
      model=trainer.outputs['model'],
      baseline_model=model_resolver.outputs['model'],
      # Change threshold will be ignored if there is no baseline (first run).
      eval_config=eval_config)

  # Checks whether the model passed the validation steps and pushes the model
  # to a file destination if check passed.
  pusher = Pusher(
      model=trainer.outputs['model'],
      model_blessing=evaluator.outputs['blessing'],
      push_destination=pusher_pb2.PushDestination(
          filesystem=pusher_pb2.PushDestination.Filesystem(
              base_directory=pusher_target)))

  return pipeline.Pipeline(
      pipeline_name=pipeline_name,
      pipeline_root=pipeline_root,
      components=[
          example_gen, statistics_gen, schema_gen, transform, trainer,
          model_resolver, evaluator, pusher, hello
      ],
      enable_cache=True,
      metadata_connection_config=metadata.sqlite_metadata_connection_config(
          os.path.join(pipeline_root, 'metadata.sqlite')),
      beam_pipeline_args=['--direct_num_workers=0'])

def main(_):
  pipeline = generate_pipeline(
      flags.FLAGS.pipeline_name,
      flags.FLAGS.pipeline_root,
      flags.FLAGS.data_root,
      flags.FLAGS.train_steps,
      flags.FLAGS.eval_steps,
      flags.FLAGS.pusher_target)

  BeamDagRunner().run(pipeline)

if __name__ == '__main__':
  flags.DEFINE_string(
      name="pipeline_name", default="santander",
      help="pipeline name used to identity different pipelines")
  flags.DEFINE_string(
      name="pipeline_root", default="/var/tmp/santander/keras-tft/",
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
  flags.DEFINE_string(
      name="pusher_target", default="/var/tmp/santander/pusher",
      help="Pusher can't create this folder for you")

  app.run(main)