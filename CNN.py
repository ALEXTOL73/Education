from __future__ import division
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import time


from Preprocess import features_creation, make_np_array
from History_Plots import plot_accuracy_and_loss
from keras import regularizers
from keras import layers
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from main import *

# print(device_lib.list_local_devices())

start = time.perf_counter()
# Подготовка данных: Convert features into a Panda dataframe for CNN, Separate data into features and class labels
for i in range(num_folders):
    exec(f"path_ch{i} = paths[i]")
    exec(f"_,_,df_cnn_ch{i},columns_cnn = features_creation(path_ch{i}, sr, duration, mono, mfccs_num, hop_length)")
    exec(f"ch{i}_cnn_df = pd.DataFrame(columns=columns_cnn)")
    for j in range(np.shape(df_cnn_ch0)[0]):
        exec(f"ch{i}_cnn_df.loc[j] = df_cnn_ch{i}[j]")
    exec(f"features_cnn_ch{i} = ch{i}_cnn_df.iloc[:, 0:-1]")
    for k in range(np.shape(ch0_cnn_df.mfccs)[0]):
        exec(f"features_cnn_ch{i}.mfccs[k] = (features_cnn_ch{i}.mfccs[k] - features_cnn_ch{i}.mfccs[k].mean())/\
           np.abs(features_cnn_ch{i}.mfccs[k]).max()")
    exec(f"features_cnn_ch{i} = make_np_array(features_cnn_ch{i}.mfccs)")



# Encode the classification labels, Обучение
le = LabelEncoder()
max_filter = 128
for i in range(num_folders):
    exec(f"y_pca_cat_ch{i} = to_categorical(le.fit_transform(ch{i}_cnn_df['class_label']))")
    exec(f"xTrain_cnn_ch{i}, xTest_cnn_ch{i}, yTrain_cnn_ch{i}, yTest_cnn_ch{i} = train_test_split(features_cnn_ch{i},\
                                                                    y_pca_cat_ch{i}, test_size=0.15, random_state=0)")
    exec(f"xTrain_cnn_ch{i}_cut = xTrain_cnn_ch{i}[:,0:max_filter,:]")
    exec(f"xTest_cnn_ch{i}_cut = xTest_cnn_ch{i}[:,0:max_filter,:]")



num_rows = np.shape(xTest_cnn_ch0_cut[0])[0]
num_columns = np.shape(xTest_cnn_ch0_cut[0])[1]
num_channels = 1


#CNN_input = layers.Input(shape=(num_rows, num_columns, num_channels)) #для слоя 2D
CNN_input = layers.Input(shape=(num_rows, num_columns)) #для слоя 1D

x = layers.Conv1D(128, 3, activation="relu", padding='same',
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
                  bias_regularizer=regularizers.l2(1e-5),
                  activity_regularizer=regularizers.l2(1e-5)
                  )(CNN_input)

x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Dropout(0.4)(x)

x = layers.Conv1D(256, 3, activation="relu",padding='same',
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
                  bias_regularizer=regularizers.l2(1e-5),
                  activity_regularizer=regularizers.l2(1e-5)
                  )(x)

x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Dropout(0.4)(x)

x = layers.Flatten()(x)

class_labels_cnn = np.unique(ch1_cnn_df['class_label'])
num_labels = class_labels_cnn.shape[0]

CNN_output = layers.Dense(num_labels, activation='softmax')(x)

model_cnn_branch = Model(CNN_input, CNN_output, name="CNN")

print(model_cnn_branch.summary())

model_cnn_branch.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                         optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001))

num_rows = np.shape(xTest_cnn_ch0_cut[0])[0]
num_columns = np.shape(xTest_cnn_ch0_cut[0])[1]
print(xTest_cnn_ch0_cut.shape)

for i in range(num_folders):
    exec(f"checkpointer_cnn_ch{i} = ModelCheckpoint(filepath= path_ws + 'weights_cnn_ch{i}.hdf5',\
                               monitor='val_accuracy',verbose=2,save_best_only=True)")
    exec(f"history_cnn_ch{i} = model_cnn_branch.fit(xTrain_cnn_ch{i}_cut, yTrain_cnn_ch{i}, batch_size=num_batch_size,\
callbacks=[checkpointer_cnn_ch{i}],epochs=num_epochs,validation_data=(xTest_cnn_ch{i}_cut,yTest_cnn_ch{i}),verbose=1)")

# Evaluating the model on the training and testing set
xTrain = [eval(f"xTrain_cnn_ch{i}_cut") for i in range(num_folders)]
yTrain = [eval(f"yTrain_cnn_ch{i}") for i in range(num_folders)]
xTest = [eval(f"xTest_cnn_ch{i}_cut") for i in range(num_folders)]
yTest = [eval(f"yTest_cnn_ch{i}") for i in range(num_folders)]
history = [eval(f"history_cnn_ch{i}") for i in range(num_folders)]


for i in range(num_folders):
    print(f'channel{i}')
    model_cnn_branch.load_weights(path_ws + f'weights_cnn_ch{i}.hdf5')
    score = model_cnn_branch.evaluate(xTrain[i], yTrain[i], verbose=0)
    print("Training Accuracy: ", round(score[1], 3))

    score = model_cnn_branch.evaluate(xTest[i], yTest[i], verbose=0)
    print("Testing Accuracy: ", round(score[1], 3))

finish = time.perf_counter()
time = round((finish - start), 3)
print("\nВремя обработки = ", time )
plot_accuracy_and_loss(history)

# Записать xTrain, YTrain,num_labels,model_branch
params = [num_labels, list(class_labels_cnn), xTrain, xTest, yTrain, yTest]
write_data = params
datafile = open(path_ws+'params'+'_'+name+'.dat', "wb")
pickle.dump(write_data, datafile)
datafile.close()

exit(0)
