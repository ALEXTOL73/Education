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
import cnn,rnn,fcnn
from cnn import CNN
from fcnn import FCNN
from rnn import RNN

from dotenv.main import load_dotenv
import os

import algorytm

load_dotenv()
SR = os.environ['SR']
DURATION = os.environ['DURATION']
MFCC_NUM = os.environ['MFCC_NUM']



class AlgotytmFactory():
    def CreateAlgotytm(self, name):
        if name == FCNN:
            return FCNN()
        elif name == CNN:
            return CNN()
        elif name == RNN:
            return RNN()
        else:
            return None
