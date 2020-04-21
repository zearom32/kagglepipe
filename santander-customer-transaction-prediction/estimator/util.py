import tensorflow as tf
import tensorflow_model_analysis as tfma
from tfx.proto import trainer_pb2

def gzip_reader_fn(filenames):
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')

def get_feature_spec():
  feature_spec = {}
  for key in range(0, 200):
    feature_spec['var_{0}'.format(key)] = tf.io.FixedLenFeature([], dtype=tf.float32)
  feature_spec['target'] = tf.io.FixedLenFeature([], dtype=tf.int64)
  return feature_spec

def input_fn(filenames, tf_transform_output=None, batch_size=200):
  feature_spec = feature_spec = get_feature_spec()

  dataset = tf.data.experimental.make_batched_features_dataset(filenames, 10, feature_spec, reader=gzip_reader_fn)

  features_dict = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
  label = features_dict.pop('target')
  return features_dict, label

def eval_input_receiver():
  feature_spec = get_feature_spec()

  # input placeholder, bytes of a tf.Example
  serialized_tf_example = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_example_tensor')
  # parse input raw to tf.Example
  features = tf.io.parse_example(serialized_tf_example, feature_spec)

  return tfma.export.EvalInputReceiver(
      features=features,
      labels=features['target'],
      receiver_tensors={'examples': serialized_tf_example})

def trainer_fn(trainer_fn_args, schema):
  feature_columns = [tf.feature_column.numeric_column('var_{0}'.format(key)) for key in range(0, 200)]

  estimator = tf.estimator.DNNClassifier(
      feature_columns=feature_columns,
      hidden_units=[50, 25])

  train_spec = tf.estimator.TrainSpec(
      lambda: input_fn(trainer_fn_args.train_files),
      max_steps=trainer_fn_args.train_steps)

  eval_spec = tf.estimator.EvalSpec(
      lambda: input_fn(trainer_fn_args.eval_files),
      steps=trainer_fn_args.eval_steps)

  receiver_fn = lambda: eval_input_receiver()

  return {
      'estimator': estimator,
      'train_spec': train_spec,
      'eval_spec': eval_spec,
      'eval_input_receiver_fn': receiver_fn
  }