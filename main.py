import os
import runpy

'# алгоритмы: K-Nearest Neighbors:KM, Random Forest:RF, Support Vector Machine:SVM, XGBoost: XGB, FCNN, CNN, RNN'
name = 'KM'  # алгоритм
sr = 96000  # частота дискретизации
mfccs_num = 128  # кол-во мел-частотных кепстральных коэффициентов
hop_length = 256  # размер кадра(БПФ)
NFFT = 512  # кол-во отсчетов БПФ
mono = False

path = './Data'
path_w = './Weights'

paths = []
for ch in os.listdir(path):
    paths.append(path + '/' + ch + '/')
path_ws = path_w + '/' + name + '/'

num_channels = 1
num_folders = len(os.listdir(path))

if name == 'FCNN':
    num_epochs = 200
    num_batch_size = 32
    duration = 0.25  # длина фрейма
elif name == 'CNN':
    num_epochs = 200
    num_batch_size = 64
    duration = 0.25
elif name == 'RNN':
    num_epochs = 150
    num_batch_size = 128
    duration = 0.25
else:
    num_epochs = 150
    num_batch_size = 128
    duration = 0.25


def main():
    runpy.run_module(mod_name=name)
    try:
        runpy.run_module(mod_name=name)
        return 0
    except:
        return 1


if __name__ == "__main__":
    exit(main())
