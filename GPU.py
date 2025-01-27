import tensorflow as tf

TF_ENABLE_ONEDNN_OPTS = '1'

GPU = tf.config.list_physical_devices('GPU')
print('Num GPUs available:', len(GPU))