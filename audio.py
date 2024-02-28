import librosa
import tensorflow as tf
from tensorflow.image import resize
import numpy as np

def preprocess_mel_item(y, sample_rate: int, target_shape=(128, 128)):
  mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate)
  mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
  return mel_spectrogram

def waveform_to_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=32)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def waveform_to_mfcc(waveform, sample_rate=44100, num_mfccs=13):
  stfts = tf.signal.stft(waveform, frame_length=1024, frame_step=256, fft_length=1024)
  spectrograms = tf.abs(stfts)

  # Warp the linear scale spectrograms into the mel-scale.
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    sample_rate, 
    num_spectrogram_bins = stfts.shape[-1].value, 
    num_mel_bins = 80, 
    lower_edge_hertz = 80.0, 
    upper_edge_hertz = 7600.0
  )
  mel_spectrograms = tf.tensordot(spectrograms, linear_to_mel_weight_matrix, 1)
  mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))

  # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
  log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

  # Compute MFCCs from log_mel_spectrograms and take the first 13.
  mfccs = tf.signal.mfccs_from_log_mel_spectrograms(
    log_mel_spectrograms
  )[..., :num_mfccs]
  return mfccs

def to_spectrogram_dataset(ds):
  return ds.map(
      map_func=lambda audio,label: (waveform_to_spectrogram(audio), label),
      num_parallel_calls=tf.data.AUTOTUNE
  )

