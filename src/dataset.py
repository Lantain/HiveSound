import tensorflow as tf
import numpy as np
import pandas as pd
import os
import shutil
import librosa
import math
import random
import tensorflow_io as tfio

from src.audio import to_spectrogram_dataset, preprocess_mel_item, waveform_to_spectrogram, waveform_to_mfcc
from pydub import AudioSegment


def squeeze(audio, labels):
#   print(f"Audio shape: {audio.shape}")
#   print(f"Audio shape: {audio[0].shape}")
  #remove first and last dimention of tensor

  audio = tf.squeeze(audio, axis=-1)
  return audio, labels

def dataset_tf(dir: str, validation_split=0.2, batch_size=32):
    train_ds, val_ds = tf.keras.utils.audio_dataset_from_directory(
        directory=dir,
        batch_size=batch_size,
        validation_split=validation_split,
        output_sequence_length=4000,
        shuffle=True,
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

def validation_tf(dir: str):
    test_ds, ds = tf.keras.utils.audio_dataset_from_directory(
        directory=dir,
        batch_size=2,
        validation_split=.99,
        output_sequence_length=4000,
        seed=0,
        labels='inferred',
        subset='both'
    )
    label_names = np.array(ds.class_names)
    print("label names:", label_names)
    print(ds.element_spec)

    ds = ds.map(squeeze, tf.data.AUTOTUNE)

    # Print examples brief
    for example_audio, example_labels in ds.take(1):  
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

    return ds


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

def segments_from_audio_file(audio_src: str, segment_length=4000, start_from=0, end_at=None) -> list[AudioSegment]:
    if os.path.exists(audio_src) is False:
        return list()
    
    audio = AudioSegment.from_file(audio_src)
    if end_at is None:
        end_at = len(audio)
    a = audio[start_from:end_at]
    segments = list([])
    audio_len = len(a)
    if audio_len > segment_length:
        for i in range(math.ceil(audio_len / 4000)):
            chunk = a[i * 4000:(i + 1) * 4000]
            if (len(chunk) == 4000):
                mono_audios = chunk.split_to_mono()
                mono_left = mono_audios[0]
                segments.append(mono_left)
    return segments

def tfio_segments_from_audio_file(audio_src: str, segment_length=4000, start_from=0, end_at=None) -> list:
    audio = tfio.audio.AudioIOTensor(audio_src, tf.float32)
    if end_at is None:
        end_at = len(audio)
    a = audio[start_from:end_at]
    segments = list([])
    audio_len = len(a)
    if audio_len > segment_length:
        for i in range(math.ceil(audio_len / 4000)):
            chunk = a[i * 4000:(i + 1) * 4000]
            if (len(chunk) == 4000):
                if audio.dtype == tf.int16:
                    chunk = tf.cast(chunk, tf.float32) / 32768.0
                segments.append(tf.squeeze(chunk, axis=[-1]) )
    return segments

def create_sbcm_original(src_dir: str, out_dir: str, df: pd.DataFrame):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(f"{out_dir}/queen", exist_ok=True)
    os.makedirs(f"{out_dir}/noqueen", exist_ok=True)

    files = os.listdir(src_dir)
    for index, row in df.iterrows():
        queen_status = row['queen status']
        queen_presence = row['queen presence']
        queen_acceptance = row['queen acceptance']
        file_name = row['file name'].replace('.raw', '')

        related_files = list()
        for f in files:
            if f.startswith(file_name):
                related_files.append(f)
        print(f"Queen Status: {queen_status}({queen_presence + queen_acceptance}), File Name: {file_name}, Files: {related_files}")
        for rf in related_files:
            if queen_presence == 0 and queen_acceptance == 0:
                shutil.copyfile(f"{src_dir}/{rf}", f"{out_dir}/noqueen/{rf}")
            if queen_presence == 1 and queen_acceptance == 2:
                shutil.copyfile(f"{src_dir}/{rf}", f"{out_dir}/queen/{rf}")
            # if queen_status == 0:
            #     shutil.copyfile(f"{src_dir}/{rf}", f"{out_dir}/noqueen/{rf}")
            # if queen_status == 3:
            #     shutil.copyfile(f"{src_dir}/{rf}", f"{out_dir}/queen/{rf}")


def create_segmented_dataset_from_dir(orig_dir: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    for label in list(["queen", "noqueen"]):
        os.makedirs(f"{out_dir}/{label}", exist_ok=True)
        files = os.listdir(f"{orig_dir}/{label}")
        for file in files:
            segments = segments_from_audio_file(f"{orig_dir}/{label}/{file}", 4000, 200, None)
            print(f"{label} - File {file} segments: {len(segments)}")
            for i, seg in enumerate(segments):
                seg.set_sample_width(2)
                seg.export(
                    f"{out_dir}/{label}/{os.path.basename(file)}_{i}.wav", 
                    format="wav", 
                    bitrate='16k', 
                    parameters=["-sample_fmt", "s16"]
                )

def create_dir_split_from(src_dir: str, train_dir: str, val_dir: str, val_files_count: int = 100, train_files_count: int = 100):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for label in list(["queen", "noqueen"]):
        os.makedirs(f"{train_dir}/{label}", exist_ok=True)
        os.makedirs(f"{val_dir}/{label}", exist_ok=True)
        src_files = os.listdir(f"{src_dir}/{label}")
        random.seed(0)
        random.shuffle(src_files)
        train_files = src_files[:train_files_count]
        print(f"Train files: {len(train_files)}")
        val_files = src_files[train_files_count:][:val_files_count]
        print(f"Validation files: {len(val_files)}")
        for file in train_files:
            shutil.copyfile(f"{src_dir}/{label}/{file}", f"{train_dir}/{label}/{file}")
        for file in val_files:
            shutil.copyfile(f"{src_dir}/{label}/{file}", f"{val_dir}/{label}/{file}")

def file_mfccs(file_src: str):
    segments = segments_from_audio_file(file_src)
    mfccs = [waveform_to_mfcc(seg) for seg in segments]
    return mfccs

def validate_on(dir: str, model):
    for label in list(["queen", "noqueen"]):
        print(f"\n=== {label} ===")
        files = os.listdir(f"{dir}/{label}")
        for file in files:
            mfccs = file_mfccs(dir + '/' + file)
            predictions = list([0, 0])
            for mfcc in mfccs:
                try:
                    mfcc_batch = np.expand_dims(mfcc, axis=0)
                    res = model.predict(mfcc_batch, verbose=0)
                    predictions[0] = predictions[0] + res[0][0]
                    predictions[1] = predictions[1] + res[0][1]
                except Exception as e:
                    print(f"Error in {file}:", e)

            mid_preds = np.array(predictions) / len(mfccs)
            print(f"File {label}/{file} predictions: NQ:{mid_preds[0]:.2f}  Q:{mid_preds[1]:.2f}")





