# Lint as: python3
"""TODO(wendycbh): DO NOT SUBMIT without one-line documentation for component.

TODO(wendycbh): DO NOT SUBMIT without a detailed description of component.
"""

import os
from typing import Any, Dict, List, Text

from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils
from tfx.utils import path_utils


class Executor(base_executor.BaseExecutor):
  """Executor for HelloComponent."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:

    self._log_startup(input_dict, output_dict, exec_properties)

    transformed_examples_uri = artifact_utils.get_single_uri(input_dict['transformed_examples'])
               
    model = artifact_utils.get_single_instance(input_dict['model'])
    absl.logging.info('Using {} as current model.'.format(model.uri))
    
    live_model = tf.keras.models.load_model(model.uri)
    absl.logging.info('Model info: {}'.format(live_model.summary())) 
    
    output_uri = os.path.join(
        artifact_utils.get_single_uri(output_dict['output_data']), 'pred.csv')
    
    
    with self._make_beam_pipeline() as pipeline:
      test_data = (pipeline | 'ReadData' >> beam.io.ReadFromTFRecord(file_pattern=io_utils.all_files_pattern(transformed_examples_uri)))
      pred = live_model.predict(test_data)
      absl.logging.info('Shape of prediction: {}'.format(pred.shape))
      formated_str = '\n'.join(['1' if item > 0.5 else '0' for item in pred])
      io_utils.write_string_file(
          file_name=output_uri, string_value=formated_str)
    
    
    

    
    # io_utils.write_string_file(
#         file_name=output_uri, string_value=input_uri.value())
