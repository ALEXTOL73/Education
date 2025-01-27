from __future__ import division
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import time

from Preprocess import features_creation,make_np_array
from History_Plots import plot_accuracy_and_loss

from keras import regularizers
from keras import layers
from keras import Model
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from main import *

# print(device_lib.list_local_devices())

start = time.perf_counter()
# Подготовка данных: Convert features into a Panda dataframe for RNN, Separate data into features and class labels
for i in range(num_folders):
    exec(f"path_ch{i} = paths[i]")
    exec(f"df_fc_ch{i},_,df_cnn_ch{i},columns_cnn =  features_creation(path_ch{i}, \
    sr, duration, mono, mfccs_num, hop_length)")
    exec(f"ch{i}_rnn_df = pd.DataFrame(columns=columns_cnn)")
    for j in range(np.shape(df_cnn_ch0)[0]):
        exec(f"ch{i}_rnn_df.loc[j] = df_cnn_ch{i}[j]")
    exec(f"features_rnn_ch{i} = ch{i}_rnn_df.iloc[:, 0:-1]")
    for k in range(np.shape(ch0_rnn_df.mfccs)[0]):
            exec(f"features_rnn_ch{i}.mfccs[k] = (features_rnn_ch{i}.mfccs[k] - \
            features_rnn_ch{i}.mfccs[k].mean())/np.abs(features_rnn_ch{i}.mfccs[k]).max()")
    exec(f"features_rnn_ch{i} = make_np_array(features_rnn_ch{i}.mfccs)")


# Encode the classification labels

le = LabelEncoder()

max_filter = 128
for i in range(num_folders):
    exec(f"y_pca_cat_ch{i} = to_categorical(le.fit_transform(ch{i}_rnn_df['class_label']))")
    exec(f"xTrain_rnn_ch{i}, xTest_rnn_ch{i}, yTrain_rnn_ch{i}, yTest_rnn_ch{i} = train_test_split(features_rnn_ch{i},\
                                                                    y_pca_cat_ch{i}, test_size=0.15, random_state=0)")
    exec(f"xTrain_rnn_ch{i}_cut = xTrain_rnn_ch{i}[:,0:max_filter,:]")
    exec(f"xTest_rnn_ch{i}_cut = xTest_rnn_ch{i}[:,0:max_filter,:]")


class_labels_rnn = np.unique(ch1_rnn_df['class_label'])
num_labels = class_labels_rnn.shape[0]
print("Num_labels = ", num_labels)

num_rows = np.shape(xTest_rnn_ch0_cut[0])[0]
num_columns = np.shape(xTest_rnn_ch0_cut[0])[1]

RNN_input = tf.keras.Input(shape=(num_rows, num_columns, num_channels))

for i in range(1, 5):
    exec(f"x_lstm{i} = layers.Permute((2, 1, 3))(RNN_input)")
    exec(f"x_lstm{i} = layers.Lambda(lambda x: tf.squeeze(x, 3))(x_lstm{i})")
    exec(f"x_lstm{i} = layers.Bidirectional(layers.GRU(48,\
                                           return_sequences=False,\
                                           dropout=0.4,\
                                           recurrent_dropout=0.3,\
                                           kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),\
                                           bias_regularizer=regularizers.l2(1e-5),\
                                           activity_regularizer=regularizers.l2(1e-5)\
                                           ))(x_lstm{i})")


x_lstm = layers.concatenate([x_lstm1, x_lstm2, x_lstm3, x_lstm4])

RNN_output = layers.Dense(num_labels, activation='softmax')(x_lstm)

model_rnn_branch = Model(RNN_input, RNN_output, name="GRU")

print(model_rnn_branch.summary())

model_rnn_branch.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                         optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))

volume_train = np.shape(xTrain_rnn_ch0_cut)[0]
volume_test = np.shape(xTest_rnn_ch0_cut)[0]

y = np.shape(xTrain_rnn_ch0_cut)[1]
x = np.shape(xTrain_rnn_ch0_cut)[2]

print(volume_train, volume_test, y, x)

for i in range(num_folders):
    exec(f"xTrain_rnn_ch{i}_cut = np.reshape(xTrain_rnn_ch{i}_cut, (volume_train,y,x,1))")
    exec(f"xTest_rnn_ch{i}_cut  = np.reshape(xTest_rnn_ch{i}_cut, (volume_test,y,x,1))")


print(np.shape(xTrain_rnn_ch0_cut))
print(np.shape(xTest_rnn_ch0_cut))

for i in range(num_folders):
    exec(f"checkpointer_rnn_ch{i} = ModelCheckpoint(filepath= path_ws + 'weights_rnn_ch{i}.hdf5',\
                               monitor='val_accuracy',verbose=2,save_best_only=True)")
    exec(f"history_rnn_ch{i} = model_rnn_branch.fit(xTrain_rnn_ch{i}_cut, yTrain_rnn_ch{i}, batch_size=num_batch_size,\
callbacks=[checkpointer_rnn_ch{i}],epochs=num_epochs,validation_data=(xTest_rnn_ch{i}_cut,yTest_rnn_ch{i}),verbose=1)")

# Evaluating the model on the training and testing set

xTrain = [eval(f"xTrain_rnn_ch{i}_cut") for i in range(num_folders)]
yTrain = [eval(f"yTrain_rnn_ch{i}") for i in range(num_folders)]
xTest = [eval(f"xTest_rnn_ch{i}_cut") for i in range(num_folders)]
yTest = [eval(f"yTest_rnn_ch{i}") for i in range(num_folders)]
history = [eval(f"history_rnn_ch{i}") for i in range(num_folders)]



for i in range(num_folders):
    print(f'channel{i}')
    model_rnn_branch.load_weights(path_ws + f'weights_rnn_ch{i}.hdf5')
    score = model_rnn_branch.evaluate(xTrain[i], yTrain[i], verbose=0)
    print("Training Accuracy: ", round(score[1], 3))

    score = model_rnn_branch.evaluate(xTest[i], yTest[i], verbose=0)
    print("Testing Accuracy: ", round(score[1], 3))

plot_accuracy_and_loss(history)
finish = time.perf_counter()
time = round((finish - start), 3)
print("\nВремя обработки = ", time)
# Записать xTrain, YTrain, num_labels
params = [num_labels, list(class_labels_rnn), xTrain, xTest, yTrain, yTest]
write_data = params
datafile = open(path_ws+'params'+'_'+name+'.dat', "wb")
pickle.dump(write_data, datafile)
datafile.close()

exit(0)
