import tensorflow as tf
import numpy as np
import os
import librosa
import math

from src.audio import to_spectrogram_dataset, preprocess_mel_item, waveform_to_spectrogram, waveform_to_mfcc
from pydub import AudioSegment


def squeeze(audio, labels):
#   print(f"Audio shape: {audio.shape}")
#   print(f"Audio shape: {audio[0].shape}")
  #remove first and last dimention of tensor

  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def dataset_tf(dir: str):
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=dir,
        batch_size=32,
        validation_split=0.2,
        output_sequence_length=4000,
        seed=0,
        labels='inferred',
        subset='both'
    )
    label_names = np.array(train_ds.class_names)
    print("label names:", label_names)
    print(train_ds.element_spec)

    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    # Print examples brief
    for example_audio, example_labels in train_ds.take(1):  
        print(f"Example audio shape: {example_audio.shape}")
        print(f"Example label shape: {example_labels.shape}")

    for i in range(2):
        label = label_names[example_labels[i]]
        waveform = example_audio[i]
        spectrogram = waveform_to_spectrogram(waveform)
        mfccs = waveform_to_mfcc(waveform)

        print('Label:', label)
        print('Waveform shape:', waveform.shape)
        print('MFCCs shape:', mfccs.shape)
        print('Spectrogram shape:', spectrogram.shape)

    return train_ds, val_ds, label_names


def load_data(data_dir, classes): 
    data = []
    labels = []
    
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        for filename in os.listdir(class_dir):
            if filename.endswith('.wav'):
                file_path = os.path.join(class_dir, filename)
                audio_data, sample_rate = librosa.load(file_path, sr=None)
                data.append([audio_data, sample_rate])
                labels.append(i)
    
    return data, np.array(labels)

def preprocess_data(data, target_shape=(128, 128)):
    spectrograms = []
    for d in data:
        audio_data = d[0]
        sample_rate = d[1]
        spectrograms.append(preprocess_mel_item(audio_data, sample_rate, target_shape))
    return np.array(spectrograms)

def dataset_raw(data_dir: str):
    classes = os.listdir(data_dir)
    data, labels = load_data(data_dir, classes)
    return data, labels, classes

def segments_from_audio_file(audio_src: str, segment_length=4000):
    try:
        audio = AudioSegment.from_file(audio_src)
        segments = list()
        audio_len = len(audio)
        if audio_len > segment_length:
            for i in range(math.ceil(len(audio_len) / 4000)):
                chunk = audio[i * 4000:(i + 1) * 4000]
                if (len(chunk) == 4000):
                    mono_audios = chunk.split_to_mono()
                    mono_left = mono_audios[0]
        return segments
    except Exception as e:
        print(f"Failed to export from file {audio_src} length {len(audio)}")