import numpy as np
import os
import librosa
from scipy.fftpack import dct

def sort_audio(files_name):
    files_name_new = []
    for item in files_name:
        files_name_new.append(int(item[:-4]))
    files_name_new.sort()
    files_name = []
    for item in files_name_new:
        files_name.append(str(item) + '.wav')
    return files_name

def get_mfccs(signal, sample_rate, NFFT=512, nfilt=40, num_ceps=12, frame_size=0.025, frame_stride=0.01):
    emphasized_signal = signal

    frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples

    signal_length = len(emphasized_signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))

    num_frames = int(
        np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

    # добавка нулей
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)
    # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal
    # нарезаем сигнал на фреймы
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
        np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    np.shape(frames)  # 348 фреймов по 200 отсчетов

    frames *= np.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
    low_freq_mel = 0
    hz_points = np.linspace(low_freq_mel, sample_rate / 2, nfilt + 2)  # Equally spaced in Mel scale
    bin = np.floor((NFFT + 1) * hz_points / sample_rate)
    fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])  # left
        f_m = int(bin[m])  # center
        f_m_plus = int(bin[m + 1])  # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1: (num_ceps + 1)]  # Keep 2-13
    return mfcc

def extract_features(audio, sr, hop_length, mfccs_num):
    mfccs = get_mfccs(signal=audio, sample_rate=sr, NFFT=2048, nfilt=128, num_ceps=128, frame_size=0.025,
                          frame_stride=0.01)
    mfccs_mean = np.mean(mfccs, axis=0)
    return mfccs.T, mfccs_mean


def features_creation(path, sr, duration, mono, mfccs_num, hop_length):
    q = os.listdir(path)
    audiodata = []
    class_label = []
    # features for fully connected NN
    df_fc = []
    df_temp_fc = []
    columns_fc = []
    # features for CNN
    df_cnn = []
    columns_cnn = []
    folder_sorted = []

    for folder in os.listdir(path):
        folder_sorted.append(folder)
    folder_sorted.sort()
    flag = True

    for folder in folder_sorted:
        folder_path = path + folder

        file_names_in_folder = os.listdir(folder_path)
        file_names_in_folder_sorted = sort_audio(file_names_in_folder)

        for file in file_names_in_folder_sorted:
            class_label = folder
            path_to_file = os.path.join(folder_path, file)
            #print(path_to_file)

            # read audio file
            librosa_audio, librosa_sample_rate = librosa.load(path_to_file, sr=sr, duration=duration, mono=mono)

            audio = np.asfortranarray(librosa_audio, dtype=float)

            # create features
            mfccs, mfccs_mean = extract_features(audio=audio,
                                                 sr=librosa_sample_rate,
                                                 hop_length=hop_length,
                                                 mfccs_num=mfccs_num
                                                 )

            # create feature for cnn
            df_cnn.append((mfccs, class_label))
            if flag:
                columns_cnn.append('mfccs')

                # create features for fully connected NN
            for item in enumerate(mfccs_mean):
                df_temp_fc.append(item[1])
                if flag:
                    columns_fc.append('mfccs_mean' + str(item[0]))

            df_temp_fc.append(class_label)
            df_fc.append(df_temp_fc)
            df_temp_fc = []

            flag = False

    columns_cnn.append('class_label')
    columns_fc.append('class_label')

    return df_fc, columns_fc, df_cnn, columns_cnn

def make_np_array(features_cnn):
    np_array = []
    for item in features_cnn.values:
        np_array.append(item)
    return np.array(np_array)