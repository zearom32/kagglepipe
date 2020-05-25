# Lint as: python3
r"""Client code to get prediction and write results to submission.cvs.

Start tensorflow serving before running.
$ docker run -p 8500:8500 --mount type=bind,source=\
<titanic_project_root>/serving_model/titanic/,target=/models/titanic \
-e MODEL_NAME=titanic -t tensorflow/serving

Assume you have titanic testing data under /tmp directory, this is the example
command to run:
$ python3 client_keras.py --test_data=/tmp/titanic/data/test/test.tfrecord \
--output_file=/tmp/titanic/submission.csv
"""
from absl import app
from absl import flags

import grpc
import tensorflow as tf
from tensorflow.core.framework import types_pb2
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

FLAGS = flags.FLAGS

flags.DEFINE_string('test_data', '~/titanic/data/test/test.tfrecord',
                    'Path of test data.')
flags.DEFINE_string('output_file', '~/titanic/submission.csv',
                    'Where to write result.')
flags.DEFINE_string('hostport', 'localhost:8500', 'Addr of serving')


def main(unused_argv):
  # Enable eager mode.
  tf.executing_eagerly()
  filenames = [FLAGS.test_data]
  # Read file to raw TFRecordDataset.
  raw_dataset = tf.data.TFRecordDataset(filenames)

  channel = grpc.insecure_channel(FLAGS.hostport)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

  pred_result = {}
  for raw_record in raw_dataset:
    example = tf.train.Example()
    # Decode byte string to tf.Example proto.
    example.ParseFromString(raw_record.numpy())
    passenger_id = example.features.feature['PassengerId'].int64_list.value[0]
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'titanic'
    request.model_spec.signature_name = 'serving_default'
    request.inputs['examples'].CopyFrom(
        tf.make_tensor_proto([example.SerializeToString()],
                             dtype=types_pb2.DT_STRING,
                             shape=[1]))
    result = stub.Predict(request, 5.0)
    prediction = '1' if result.outputs['output_0'].float_val[0] > 0.5 else '0'
    pred_result[passenger_id] = prediction
  if len(pred_result) == 418:
    f = open(FLAGS.output_file, 'w')
    f.write('PassengerId,Survived\n')
    for passenger_id, pred in pred_result.items():
      f.write('%s,%s\n' % (passenger_id, pred))
    f.close()
    print('Output file to %s.' % FLAGS.output_file)
  else:
    print(
        'Error: Number of prediction not equal to test size. Number prediction %d'
        % len(pred_result))


if __name__ == '__main__':
  app.run(main)
