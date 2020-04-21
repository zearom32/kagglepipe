import absl
import tensorflow as tf
from tensorflow import keras
from typing import Text
import os

def gzip_reader_fn(filenames):
  return tf.data.TFRecordDataset(
      filenames,
      compression_type='GZIP')

def get_feature_spec(include_label):
  feature_spec = {}
  for key in range(0, 200):
    feature_spec['var_{0}'.format(key)] = tf.io.FixedLenFeature([1], dtype=tf.float32)

  if include_label:
    feature_spec['target'] = tf.io.FixedLenFeature([1], dtype=tf.int64)
  return feature_spec

def input_fn(file_pattern: Text,
              batch_size: int = 200) -> tf.data.Dataset:
  feature_spec = get_feature_spec(include_label=True)

  dataset = tf.data.experimental.make_batched_features_dataset(
      file_pattern=file_pattern,
      batch_size=batch_size,
      features=feature_spec,
      reader=gzip_reader_fn,
      label_key='target')

  return dataset

def build_keras_model() -> tf.keras.Model:
  inputs = [
      keras.layers.Input(shape=(1,), name='var_{0}'.format(key))
      for key in range(0, 200)
  ]
  d = keras.layers.concatenate(inputs)
  d = keras.layers.Dense(50, activation='relu')(d)
  d = keras.layers.Dense(25, activation='relu')(d)
  outputs = keras.layers.Dense(2, activation='softmax')(d)
  model = keras.Model(inputs=inputs, outputs=outputs)
  model.compile(
      optimizer=keras.optimizers.Adam(lr=0.0005),
      loss='binary_crossentropy',
      metrics=[keras.metrics.BinaryCrossentropy()])

  model.summary(print_fn=absl.logging.info)
  return model

def get_serve_tf_examples_fn(model):

  @tf.function
  def serve_tf_examples_fn(serialized_tf_examples):
    feature_spec = get_feature_spec(include_label=False)
    parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)
    return model(parsed_features)

  return serve_tf_examples_fn

def run_fn(fn_args):
  """TFX Trainer will call this function.
  """

  train_dataset = input_fn(fn_args.train_files, batch_size=20)
  eval_dataset = input_fn(fn_args.eval_files, batch_size=10)


  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
    model = build_keras_model()
  
  # View all logs in different runs
  # tensorboard --logdir /var/tmp/santander/pipekeras/Trainer/
  log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), 'logs')
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq='batch')

  model.fit(
      train_dataset,
      epochs=2,
      steps_per_epoch=1000,
      validation_data=eval_dataset,
      validation_steps=fn_args.eval_steps,
      callbacks=[tensorboard_callback])

  signatures = {
      'serving_default': get_serve_tf_examples_fn(model).get_concrete_function(
          tf.TensorSpec(shape=[None],
                        dtype=tf.string,
                        name='input_example_tensor')),
  }

  model.save(fn_args.serving_model_dir, save_format='tf', signatures=signatures)