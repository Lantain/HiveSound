import matplotlib.pyplot as plt
import librosa
import numpy as np
import librosa.display as ld
import tensorflow as tf
from tensorflow import keras

def plot_audio(audio, sr: float):
	n_fft = 2048
	# ft = np.abs(librosa.stft(example_audio[:n_fft], hop_length = n_fft+1))
	# plt.plot(ft)
	# plt.title('Spectrum')
	# plt.xlabel('Frequency Bin')
	# plt.ylabel('Amplitude')

	plt.figure(figsize=(20, 4))

	plt.subplot(1, 3, 1)
	ft = np.abs(librosa.stft(audio[:n_fft], hop_length = n_fft+1))
	plt.plot(ft)
	plt.title('Spectrum')
	plt.xlabel('Frequency Bin')
	plt.ylabel('Amplitude')

	plt.subplot(1, 3, 2)
	X = librosa.stft(audio)
	s = librosa.amplitude_to_db(abs(X))
	ld.specshow(s, sr=sr, x_axis = 'time', y_axis='linear')
	plt.colorbar()

	melspectrum = librosa.feature.melspectrogram(y=audio, sr = sr,
												hop_length =512, n_mels = 40)
	librosa.display.specshow(melspectrum, sr=sr, x_axis='time', y_axis='mel')

	plt.subplot(1, 3, 3)
	X = librosa.stft(audio)
	s = librosa.amplitude_to_db(abs(X))
	ld.specshow(s, sr=sr, x_axis = 'time', y_axis='linear')
	plt.colorbar()
	plt.title('Mel spectrogram')
	plt.tight_layout()

	plt.figure(figsize=(12,4))
	ld.waveshow(audio, sr=sr, color='b')
	
def plot_spectrogram(spectrogram, ax):
	if len(spectrogram.shape) > 2:
		assert len(spectrogram.shape) == 3
		spectrogram = np.squeeze(spectrogram, axis=-1)
	# Convert the frequencies to log scale and transpose, so that the time is
	# represented on the x-axis (columns).
	# Add an epsilon to avoid taking a log of zero.
	log_spec = np.log(spectrogram.T + np.finfo(float).eps)
	height = log_spec.shape[0]
	width = log_spec.shape[1]
	X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
	Y = range(height)
	ax.pcolormesh(X, Y, log_spec)

def plot_spectrograms(spectrograms: list, spectrogram_labels: list, classes: list):
	rows = 3
	cols = 3
	n = rows*cols
	fig, axes = plt.subplots(rows, cols, figsize=(16, 9))

	for i in range(n):
		r = i // cols
		c = i % cols
		ax = axes[r][c]
		plot_spectrogram(spectrograms[i].numpy(), ax)
		ax.set_title(classes[spectrogram_labels[i].numpy()])

	plt.show()

def show_spectrograms(ds, classes):
	for example_spectrograms, example_spect_labels in ds.take(1):
  		plot_spectrograms(example_spectrograms, example_spect_labels, classes)

def plot_model(model, to_file="model.png"):
	keras.utils.plot_model(
        model,
        to_file=to_file,
        show_shapes=True,
        show_dtype=True,
        show_layer_names=True,
        expand_nested=False,
        dpi=200,
        show_layer_activations=True,
        show_trainable=True
    )


def plot_history(history):
	metrics = history.history
	plt.figure(figsize=(16,6))
	plt.subplot(1,2,1)
	plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
	plt.legend(['loss', 'val_loss'])
	plt.ylim([0, max(plt.ylim())])
	plt.xlabel('Epoch')
	plt.ylabel('Loss [CrossEntropy]')

	plt.subplot(1,2,2)
	plt.plot(history.epoch, 100*np.array(metrics['accuracy']), 100*np.array(metrics['val_accuracy']))
	plt.legend(['accuracy', 'val_accuracy'])
	plt.ylim([0, 100])
	plt.xlabel('Epoch')
	plt.ylabel('Accuracy [%]')