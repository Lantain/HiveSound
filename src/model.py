import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Resizing, Dropout, Normalization
from tensorflow.keras.models import Model, Sequential

# shape
def get_conv_model(shape, classes_count: int):
  input_shape = shape
  input_layer = Input(shape=input_shape)
  x = Conv2D(32, (3, 3), activation='relu')(input_layer)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(64, (3, 3), activation='relu')(x)
  x = MaxPooling2D((2, 2))(x)
  x = Flatten()(x)
  x = Dropout(0.4)(x)
  x = Dense(64, activation='relu')(x)
  output_layer = Dense(classes_count, activation='softmax')(x)
  model = Model(input_layer, output_layer)
  return model


def get_conv_model_mini(shape, classes_count: int):
  input_shape = shape
  input_layer = Input(shape=input_shape)
  x = Conv2D(16, (3, 3), activation='relu', name='Згортковий шар 1')(input_layer)
  # x = tf.keras.layers.BatchNormalization()(x)
  x = MaxPooling2D((2, 2), name='')(x)
  x = Conv2D(32, (3, 3), activation='relu', name='Згортковий шар 2')(x)
  # x = tf.keras.layers.BatchNormalization()(x)
  x = MaxPooling2D((2, 2), name='')(x)
  x = Flatten(name='')(x)
  x = Dropout(0.4, name='Dropout шар (0.4)')(x)
  x = Dense(32, activation='relu', name='Повністю зв’язний шар 1')(x)
  # x = tf.keras.layers.BatchNormalization()(x)
  output_layer = Dense(classes_count, activation='softmax', name='Вихідний шар')(x)
  model = Model(input_layer, output_layer)
  return model
# input_shape = example_spectrograms.shape[1:]
def get_seq_model(shape, num_labels: int, spectrogram_ds: list):
  norm_layer = Normalization()
  norm_layer.adapt(data=spectrogram_ds.map(map_func=lambda spec, label: spec))

  model = Sequential([
    Input(shape=shape),
    # Downsample the input.
    Resizing(32, 32),
    # Normalize.
    norm_layer,
    Conv2D(32, 3, activation='relu'),
    Conv2D(64, 3, activation='relu'),
    MaxPooling2D(),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(num_labels),
  ])
  return model

callbacks = [
  # tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=2, start_from_epoch=5),
  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, start_from_epoch=10),
  tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=20, start_from_epoch=10),
]