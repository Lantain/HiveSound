import librosa
import tensorflow as tf
from tensorflow.image import resize
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain, LoudnessNormalization
import numpy as np
import soundfile as sf  # Adding soundfile for saving the file
import os

def preprocess_mel_item(y, sample_rate: int, target_shape=(128, 128)):
  mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sample_rate)
  mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
  return mel_spectrogram

def waveform_to_spectrogram(waveform):
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=32)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

def waveform_to_mfcc(waveform, sample_rate=44100, num_mfccs=16):
  stfts = tf.signal.stft(waveform, frame_length=1024, frame_step=256, fft_length=1024)

  spectrograms = tf.abs(stfts)
  num_spectrogram_bins = stfts.shape[-1]#.value
  print(num_spectrogram_bins, sample_rate, num_mfccs)
  lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 7600.0, 80
  # lower_edge_hertz, upper_edge_hertz, num_mel_bins = 80.0, 4096.0, 80
  # Warp the linear scale spectrograms into the mel-scale.
  linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
  num_mel_bins, num_spectrogram_bins, sample_rate, lower_edge_hertz,
  upper_edge_hertz)
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

def to_mfccs_dataset(ds):
  return ds.map(
      map_func=lambda audio,label: (waveform_to_mfcc(audio, 44100, 12), label),
      num_parallel_calls=tf.data.AUTOTUNE
  )

def transform_files_from(from_dir, to_dir, low_hz=1024, attenuation_max=16):
    files = os.scandir(from_dir)
    for file in files:
        basepath = os.path.basename(file)
        attenuate_frequencies(file, f"{to_dir}/tr_{basepath}", low_hz, attenuation_max)


def attenuate_frequencies(file_from, file_to, low_hz=1024, attenuation_max=16):
    y, sr = librosa.load(file_from)
    D = librosa.stft(y)
    frequencies = librosa.fft_frequencies(sr=sr)
    target_bins = np.where(frequencies >= low_hz)[0]
    bc = len(target_bins)

    for i in range(0, bc):
        tb = target_bins[i]
        x = (i / bc) * attenuation_max
        atten = (x * x) / attenuation_max
        af = librosa.db_to_amplitude(-atten)
        D[[tb], :] *= af

    y_modified = librosa.istft(D)
    sf.write(file_to, y_modified, sr)
             

def pitch_files_from(from_dir, to_dir, pitch=5):
    os.makedirs(to_dir, exist_ok=True)
    files = os.scandir(from_dir)
    for file in files:
        y, sr = librosa.load(file)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=pitch)
        basepath = os.path.basename(file)
        sf.write(f"{to_dir}/ptch_{basepath}", y_shifted, sr)


def depitch_files_from(from_dir, to_dir, pitch=5):
    os.makedirs(to_dir, exist_ok=True)
    files = os.scandir(from_dir)
    for file in files:
        y, sr = librosa.load(file)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-pitch)
        basepath = os.path.basename(file)
        sf.write(f"{to_dir}/deptch_{basepath}", y_shifted, sr)

def augment_dir(from_dir, to_dir):
    os.makedirs(to_dir, exist_ok=True)
    files = os.scandir(from_dir)
    for file in files:
        y, sr = librosa.load(file)
        augment = Compose([
          AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
          PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
          Gain(min_gain_db=-4, max_gain_db=12),
          LoudnessNormalization()
        ])
        y_augmented = augment(samples=y, sample_rate=sr)
        basepath = os.path.basename(file)
        sf.write(f"{to_dir}/aug_{basepath}", y_augmented, sr)

    

# y, sr = librosa.load(file)
# D = librosa.stft(y)
# frequencies = librosa.fft_frequencies(sr=sr)
# target_bins = np.where(frequencies >= low_hz)[0]
# bc = len(target_bins)

# for i in range(0, bc):
#     tb = target_bins[i]
#     x = (i / bc) * attenuation_max
#     atten = (x * x) / attenuation_max
#     af = librosa.db_to_amplitude(-atten)
#     D[[tb], :] *= af

# y_modified = librosa.istft(D)

# sf.write(f"{to_dir}/tr_{basepath}", y_modified, sr)