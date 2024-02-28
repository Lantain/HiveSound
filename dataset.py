import tensorflow as tf
import numpy as np
import os
import librosa

from audio import to_spectrogram_dataset, preprocess_mel_item, waveform_to_spectrogram


def squeeze(audio, labels):
  print(f"Audio shape: {audio.shape}")
  print(f"Audio shape: {audio[0].shape}")
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

        print('Label:', label)
        print('Waveform shape:', waveform.shape)
        print('Spectrogram shape:', spectrogram.shape)
        print('Audio playback')

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

def dataset_raw(data_dir: str, classes: list[str]):
    data, labels = load_data(data_dir, classes)
    return data, labels
