import csv
import datetime
import pandas as pd
from numpy import sort
from sklearn import metrics
from scipy import signal
from sklearn.preprocessing import Normalizer
from skimage.transform import resize
from keras.callbacks import EarlyStopping
from skimage.restoration import denoise_wavelet
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import adam_v2
from sklearn.model_selection import StratifiedKFold
from statistics import stdev
from scipy.fft import fftshift
import matplotlib.pyplot as plt
import numpy as np
import math
import pywt
from scipy.signal import lfilter
from scipy.fft import fft, fftfreq

import tensorflow as tf
from tensorflow import keras
from keras import regularizers

import seaborn as sn

filtered_arr = []
train_digits = []
test_dataset = []
test_digits = []
img_size = 127
SIZE_SQ = img_size * img_size


class MyThresholdCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(MyThresholdCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        val_accuracy = logs["sparse_categorical_accuracy"]
        if val_accuracy >= self.threshold:
            self.model.stop_training = True


def create_model(dr, lr):
    model = keras.Sequential()

    model.add(keras.layers.Conv1D(filters=10, kernel_size=30, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))

    model.add(keras.layers.Conv1D(filters=10, kernel_size=30, activation='relu'))

    model.add(keras.layers.MaxPooling1D(pool_size=1, strides=1))
    model.add(keras.layers.Conv1D(filters=20, kernel_size=10, activation='relu'))

    model.add(keras.layers.Dropout(dr))
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(units=16, activation='relu'))

    model.add(keras.layers.Dense(units=32, activation='relu'))
    model.add(keras.layers.Dense(units=10, activation='softmax'))

    opt = adam_v2.Adam(learning_rate=lr)
    # opt2 = keras.optimizers.RMSprop(learning_rate=lr)
    # opt3 = keras.optimizers.SGD(learning_rate=lr)
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
    return model


def dwt_filter(eeg_data, lvl, mode='soft'):
    scaled_eeg = eeg_data / 220

    denoised = denoise_wavelet(scaled_eeg, method='VisuShrink', mode=mode, wavelet_levels=lvl, wavelet='sym8',
                               rescale_sigma=True)
    return denoised


def white_noise_augmentation(eeg_data, mean, std_dev, size):
    noise = np.random.normal(loc=mean, scale=std_dev, size=size)
    eeg_filt = eeg_data + noise
    return eeg_filt


def iir_filter(eeg_data, n):
    b = [1.0 / n] * n
    a = 1
    y_filtered = lfilter(b, a, eeg_data)
    return y_filtered


def butter_bandpass_filter(data, lowcut, highcut, fs, order=2):
    nyq = 0.5 * fs
    low = lowcut /nyq
    high = highcut/nyq
    b, a = signal.butter(order, [low, high], btype='band')
    #print(b,a)
    y = lfilter(b, a, data)
    return y


def notch_filter(data, notch_freq, quality_factor, sample_freq=256/2):
    b_notch, a_notch = signal.iirnotch(notch_freq, quality_factor, sample_freq)
    freq, h = signal.freqz(b_notch, a_notch, fs=sample_freq)

    y_notched = signal.filtfilt(b_notch, a_notch, data)  # Apply the notch filter at 60 hz
    return y_notched


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def dc_filter(eeg_data, freq_sample):
    n = 3  # filter order
    Fc = 3  # cut off freq (Hz), any dc offset should be removed between 1 and 5 hz
    Nyq = freq_sample / 2.  # Nyquist freq (1/2 Fs)
    normalized_cutoff_freq = float(Fc) / float(Nyq)
    sos = signal.butter(n, normalized_cutoff_freq, btype='hp', output='sos', analog=False)
    y_dc_filtered = signal.sosfilt(sos, eeg_data)
    return y_dc_filtered


if __name__ == '__main__':
    with open('../train_dataset4.csv', 'r') as csvfile:
        spamwriter = list(csv.reader(csvfile))
        index = 0
        for line in spamwriter:
            x = line
            # print('line = ', x)
            x = list(map(int, x))
            freq = 220
            freq2 = 256
            T = 2
            N = freq * T
            N2 = freq2 * T


            eeg = np.array(x)
            length = len(eeg)
            dc_filtered = dc_filter(eeg, freq)
            yf = fft(dc_filtered) / N
            yf2 = fft(dc_filtered) / N2
            PSD = 2 * np.abs(yf[1:length//2+1])


            ds1 = dwt_filter(PSD, 2, 'hard')


            # sos = signal.butter(3, 38, 'hp', fs=freq, output='sos')

            filtered = butter_bandpass_filter(ds1, 4, 30, freq, 3)

            n = 10  # the larger n is, the smoother curve will be
            b = [1.0 / n] * n
            a = 1
            eeg_filt = filtered
            eeg_filt = white_noise_augmentation(filtered, np.mean(filtered), 10, len(filtered))

            ar = np.asarray(eeg_filt)

            r_ar = resize(ar, (300, 1), mode="constant")
            filtered_arr.append(r_ar)

    X_train = np.stack(filtered_arr, axis=0)
    X_train = np.array(X_train).astype(np.float)
    print('X_train shape is: ', X_train.shape)

    with open('../train_digits4.csv', 'r') as csvfile:
        writer = csv.reader(csvfile)
        for line in writer:
            train_digits.append(line)

    y_train = np.array(train_digits)[0]
    y_train = np.array(y_train).astype(np.float)


lrs = [ 0.2, 0.001, 0.0001]
dr = [0.0, 0.2]
acc = []


s = np.arange(0, len(X_train), 1)
shuffle(s)
X_data = X_train[s]
Y_labels = y_train[s]
summary = ''

for r in range(len(dr)):
    for c in range(len(lrs)):
        kfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=2)
        accuracies2 = []
        for train_mask, test_mask in kfolds.split(X_train, y_train):
            X_trainC = X_train[train_mask]
            y_trainC = y_train[train_mask]
            # X_trainC /= 255
            X_testC = X_train[test_mask]
            y_testC = y_train[test_mask]
            # X_testC /= 255
            model2 = create_model(lrs[c], dr[r])

            callback = EarlyStopping(
                monitor='sparse_categorical_accuracy', min_delta=0.0001,
                patience=10)

            hist = model2.fit(X_trainC, y_trainC, epochs=200,  callbacks=[callback], verbose=1)
            y_predicted = model2.predict(X_testC)
            y_predicted_labels = [np.argmax(i) for i in y_predicted]
            acc = metrics.accuracy_score(y_testC, y_predicted_labels)
            print("Accuracy on Test: ", acc)

            accuracies2.append(acc)
            if max(accuracies2) == acc:
                best_hist = hist

            summary += f'Dropout: {dr[r]}, Learning Rate: {lrs[c]}; Accuracy: {acc} - {datetime.datetime.now()} \n'
        print(accuracies2, "\nAverage Accuracy: ", np.average(accuracies2))


print(summary)
