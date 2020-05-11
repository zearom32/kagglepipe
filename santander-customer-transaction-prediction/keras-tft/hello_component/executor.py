import json
import os
from typing import Any, Dict, List, Text
import absl

import tensorflow as tf

from tfx import types
from tfx.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils


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

      for filename in tf.io.gfile.listdir(input_dir):
        input_uri = os.path.join(input_dir, filename)
        output_uri = os.path.join(output_dir, filename)

        # TODO convert the format to CSV
        # Input format: https://github.com/tensorflow/serving/blob/master/tensorflow_serving/apis/prediction_log.proto#L40
        # Output format: CSV

        # https://github.com/tensorflow/tfx-bsl/blob/21dc8b397d1db30d5a8b40244668afe9081d9052/tfx_bsl/beam/run_inference.py#L790

        io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)


    absl.logging.info('Hello Component - Executor - Do End')
