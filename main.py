import os
import runpy
from dotenv.main import load_dotenv

load_dotenv()
name = os.environ['NAME']
sr = int(os.environ['SR'])
duration = float(os.environ['DURATION'])
mono = False
mfccs_num = int(os.environ['MFCC_NUM'])
hop_length = int(os.environ['HOP_LENGTH'])

path = os.environ['PATH_SIG']
path_w = os.environ['PATH_WEIGHTS']

paths = []
for ch in os.listdir(path):
    paths.append(path + '/' + ch + '/')
path_ws = path_w + '/' + name + '/'

num_channels = int(os.environ['NUM_CHANNELS'])

if name == 'FCNN':
    num_epochs = 30
    num_batch_size = 128
elif name == 'CNN':
    num_epochs = 20
    num_batch_size = 64
elif name == 'RNN':
    num_epochs = 10
    num_batch_size = 64
else:
    num_epochs = 50
    num_batch_size = 128


def main():
    try:
        runpy.run_module(mod_name=name + '_all')
        return 0
    except:
        return 1


if __name__ == "__main__":
    exit(main())
