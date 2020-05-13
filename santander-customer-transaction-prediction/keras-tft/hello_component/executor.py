import json
import os
from typing import Any, Dict, List, Text
import absl

import tensorflow as tf

from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tensorflow_serving.apis import prediction_log_pb2
import apache_beam as beam


class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    self._log_startup(input_dict, output_dict, exec_properties)

    absl.logging.info('Hello Component - Executor - Do Start')

    for artifact in input_dict['input_data']:
      input_dir = artifact.uri
      output_dir = artifact_utils.get_single_uri(output_dict['output_data'])

      input_uri = io_utils.all_files_pattern(input_dir)
      output_uri = os.path.join(output_dir, 'result')

      with self._make_beam_pipeline() as p:
        intrim = p | 'ReadData' >> beam.io.ReadFromTFRecord(file_pattern=input_uri, coder=beam.coders.ProtoCoder(prediction_log_pb2.PredictionLog))
        intrim = intrim | 'Process' >> beam.Map(process_item)
        intrim | 'Sink' >> beam.io.WriteToText(file_path_prefix=output_uri,
                                               file_name_suffix='.csv',
                                               num_shards=1,
                                               # CompressionTypes.UNCOMPRESSED,
                                               header='ID_code,target')

    absl.logging.info('Output file ready here: {0}'.format(output_dir))
    absl.logging.info('Hello Component - Executor - Do End')
  
def process_item(item):
  example_bytes = item.predict_log.request.inputs['input_example_tensor'].string_val[0]    
  # parsed = tf.train.Example.FromString(example_bytes)
  # parsed is tf.Example (list of feature, hard to find the ID_code)

  features = {
      'ID_code': tf.io.FixedLenFeature((), tf.string)
  }
  parsed = tf.io.parse_single_example(example_bytes, features=features)
  # parsed['ID_code'] is a Tensor with string value, .numpy() can gets the value like b'id1'
  
  id_string = parsed['ID_code'].numpy().decode()
  output = item.predict_log.response.outputs['output_0'].float_val[0]
  target = 1 if output >= 0.5 else 0
  absl.logging.info('==={0},{1}'.format(output, target))
  
  return '{0},{1}'.format(id_string, target)
 