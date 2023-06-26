from abc import ABC, abstractmethod
import numpy as np
import librosa


class Data_Transfer(ABC):
    def __init__(self, y, sr , nfft, mfcc_num):
        self.y = y
        self.sr = sr
        self.nfft = nfft
        self.mfcc_num = mfcc_num

    @abstractmethod
    # Функция параметризации аудио
    def get_features(y, sr):
        # Получаем различные параметры аудио
        mfcc = librosa.feature.mfcc(y= y, sr=sr, n_mfcc=y.mfcc_num)  # Мел кепстральные коэффициенты (по умолчанию 20)
        rmse = np.mean(librosa.feature.rmse(y=y))  # Среднеквадратичная амплитуда
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))  # среднее спектральныго центроида
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))  # среднее ширины полосы частот
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))  # среднее спектрального спада частоты


        # Добавляем все параметры в один список
        out = []  # создаем пустой список
        out.append(rmse)  # добавляем среднеквадратичную амплитуду
        out.append(spec_cent)  # добавляем спектральный центроид
        out.append(spec_bw)  # добавляем ширину полосы частот
        out.append(rolloff)  # добавляем спектральный спад частоты


        # добавляем среднее всех Мел спектральных коэффициентов (20 значений)
        for e in mfcc:
            out.append(np.mean(e))

        return out