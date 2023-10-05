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

num_epochs = int(os.environ['NUM_EPOCHS'])
num_batch_size = int(os.environ['NUM_BATCH_SIZE'])
num_channels = int(os.environ['NUM_CHANNELS'])

def main():
    try:
        runpy.run_module(mod_name=name + '_all')
        return 0
    except:
        return 1


if __name__ == "__main__":
    exit(main())
