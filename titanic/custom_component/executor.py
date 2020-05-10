# Lint as: python3
"""TODO(wendycbh): DO NOT SUBMIT without one-line documentation for component.

TODO(wendycbh): DO NOT SUBMIT without a detailed description of component.
"""

import os
from typing import Any, Dict, List, Text
import absl
import apache_beam as beam
import tensorflow as tf
from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils

# Not running
class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    self._log_startup(input_dict, output_dict, exec_properties)

    transformed_examples_uri = artifact_utils.get_split_uri(input_dict['transformed_examples'], 'train')
    
               
    model = artifact_utils.get_single_instance(input_dict['model'])
    model_path = path_utils.serving_model_path(model.uri)
    absl.logging.info('Using {} as current model.'.format(model_path))
    
    live_model = tf.keras.models.load_model(model_path)
    absl.logging.info('Model info: {}'.format(live_model.summary())) 
    
    output_uri = os.path.join(
        artifact_utils.get_single_uri(output_dict['output_data']), 'pred.csv')
    with self._make_beam_pipeline() as pipeline:
      test_data = pipeline | 'ReadFromTFRecord' >> beam.io.ReadFromTFRecord(file_pattern=io_utils.all_files_pattern(transformed_examples_uri))
      (test_data | 'Values' >> beam.Values() 
      | 'ParseToExample' >> beam.Map(tf.train.Example.FromString)
      | 'Predict' >> beam.Map(lambda x: live_model.predict(x))
      | 'WriteToFile' >> beam.io.WriteToText(output_uri))
    absl.logging.info('TestPredComponent result written to %s.', output_uri)
