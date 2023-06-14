from __future__ import division
import warnings

warnings.filterwarnings('ignore')
import tensorflow as tf
import math

# обработка аудио

import librosa
import librosa.display

# работа с масивамми
import numpy as np
from matplotlib import pyplot as plt

# работа с данными
import os
import pandas as pd
from scipy.fftpack import dct

from PIL import Image, ImageDraw, ImageFont
import sys

import numpy as np
import os

from abc import abstractmethod, abstractproperty


class Data_Transfer(object):
    @abstractmethod
    def prepare(self):
        pass


class Mfcc(Data_Transfer):
    pass


class Spectrum(Data_Transfer):
    pass


class Write_File:
    pass

class Algorytm:
    pass


class CNN(Algorytm):
    pass

class FCNN(Algorytm):
    pass

class RNN:
    pass

class Learning(alg,data):
    self.alg = alg
    self.data = data

    def learn(self,data):
        pass

class Factory(Mfcc,RNN,FCNN):
    pass

