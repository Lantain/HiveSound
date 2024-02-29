import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Resizing, Dropout, Normalization
from keras.models import Model, Sequential

# shape
def get_conv_model(shape, classes_count: int):
  input_shape = shape
  input_layer = Input(shape=input_shape)
  x = Conv2D(32, (3, 3), activation='relu')(input_layer)
  x = MaxPooling2D((2, 2))(x)
  x = Conv2D(64, (3, 3), activation='relu')(x)
  x = MaxPooling2D((2, 2))(x)
  x = Flatten()(x)
  x = Dense(64, activation='relu')(x)
  output_layer = Dense(classes_count, activation='softmax')(x)
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
    Dropout(0.5),
    Dense(num_labels),
  ])
  return model

callbacks = [
  tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=6, start_from_epoch=6),
  tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, start_from_epoch=6),
  tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=6, start_from_epoch=6),
]