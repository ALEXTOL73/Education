from __future__ import division
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

from Preprocess import features_creation,make_np_array
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

print(device_lib.list_local_devices())

path_ch0,path_ch1,path_ch2,path_ch3 = paths


_,_,df_cnn_ch0,columns_cnn =  features_creation(path_ch0, sr, duration, mono, mfccs_num, hop_length)
_,_,df_cnn_ch1,_ =  features_creation(path_ch1, sr, duration, mono, mfccs_num, hop_length)
_,_,df_cnn_ch2,_ =  features_creation(path_ch2, sr, duration, mono, mfccs_num, hop_length)
_,_,df_cnn_ch3,_ =  features_creation(path_ch3, sr, duration, mono, mfccs_num, hop_length)

print('\n size of features for CNN')
print(np.shape(df_cnn_ch0))


# Convert features into a Panda dataframe for CNN

ch0_cnn_df = pd.DataFrame(columns = columns_cnn)
ch1_cnn_df = pd.DataFrame(columns = columns_cnn)
ch2_cnn_df = pd.DataFrame(columns = columns_cnn)
ch3_cnn_df = pd.DataFrame(columns = columns_cnn)




for i in range(np.shape(df_cnn_ch0)[0]):
    ch0_cnn_df.loc[i] = df_cnn_ch0[i]
    ch1_cnn_df.loc[i] = df_cnn_ch1[i]
    ch2_cnn_df.loc[i] = df_cnn_ch2[i]
    ch3_cnn_df.loc[i] = df_cnn_ch3[i]

#separate data into features and class labels
features_cnn_ch0    = ch0_cnn_df.iloc[:,0:-1]
features_cnn_ch1    = ch1_cnn_df.iloc[:,0:-1]
features_cnn_ch2    = ch2_cnn_df.iloc[:,0:-1]
features_cnn_ch3    = ch3_cnn_df.iloc[:,0:-1]

for i in range(np.shape(ch0_cnn_df.mfccs)[0]):
    features_cnn_ch0.mfccs[i] = (features_cnn_ch0.mfccs[i] - features_cnn_ch0.mfccs[i].mean())/np.abs(features_cnn_ch0.mfccs[i]).max()
    features_cnn_ch1.mfccs[i] = (features_cnn_ch1.mfccs[i] - features_cnn_ch1.mfccs[i].mean())/np.abs(features_cnn_ch1.mfccs[i]).max()
    features_cnn_ch2.mfccs[i] = (features_cnn_ch2.mfccs[i] - features_cnn_ch2.mfccs[i].mean())/np.abs(features_cnn_ch2.mfccs[i]).max()
    features_cnn_ch3.mfccs[i] = (features_cnn_ch3.mfccs[i] - features_cnn_ch3.mfccs[i].mean())/np.abs(features_cnn_ch3.mfccs[i]).max()

features_cnn_ch0 = make_np_array(features_cnn_ch0.mfccs)
features_cnn_ch1 = make_np_array(features_cnn_ch1.mfccs)
features_cnn_ch2 = make_np_array(features_cnn_ch2.mfccs)
features_cnn_ch3 = make_np_array(features_cnn_ch3.mfccs)

# Encode the classification labels
le = LabelEncoder()
y_pca_cat_ch0 = to_categorical(le.fit_transform(ch0_cnn_df['class_label']))
y_pca_cat_ch1 = to_categorical(le.fit_transform(ch1_cnn_df['class_label']))
y_pca_cat_ch2 = to_categorical(le.fit_transform(ch2_cnn_df['class_label']))
y_pca_cat_ch3 = to_categorical(le.fit_transform(ch3_cnn_df['class_label']))


#Обучение
xTrain_cnn_ch0, xTest_cnn_ch0, yTrain_cnn_ch0,yTest_cnn_ch0 = train_test_split(features_cnn_ch0, y_pca_cat_ch0,test_size = 0.15, random_state = 0)
xTrain_cnn_ch1, xTest_cnn_ch1, yTrain_cnn_ch1,yTest_cnn_ch1 = train_test_split(features_cnn_ch1, y_pca_cat_ch1,test_size = 0.15, random_state = 0)
xTrain_cnn_ch2, xTest_cnn_ch2, yTrain_cnn_ch2,yTest_cnn_ch2 = train_test_split(features_cnn_ch2, y_pca_cat_ch2,test_size = 0.15, random_state = 0)
xTrain_cnn_ch3, xTest_cnn_ch3, yTrain_cnn_ch3,yTest_cnn_ch3 = train_test_split(features_cnn_ch3, y_pca_cat_ch3,test_size = 0.15, random_state = 0)

max_filter = 128
xTrain_cnn_ch0_cut = xTrain_cnn_ch0[:,0:max_filter,:]
xTrain_cnn_ch1_cut = xTrain_cnn_ch1[:,0:max_filter,:]
xTrain_cnn_ch2_cut = xTrain_cnn_ch2[:,0:max_filter,:]
xTrain_cnn_ch3_cut = xTrain_cnn_ch3[:,0:max_filter,:]

xTest_cnn_ch0_cut = xTest_cnn_ch0[:,0:max_filter,:]
xTest_cnn_ch1_cut = xTest_cnn_ch1[:,0:max_filter,:]
xTest_cnn_ch2_cut = xTest_cnn_ch2[:,0:max_filter,:]
xTest_cnn_ch3_cut = xTest_cnn_ch3[:,0:max_filter,:]



num_rows = np.shape(xTest_cnn_ch0_cut[0])[0]
num_columns = np.shape(xTest_cnn_ch0_cut[0])[1]
num_channels = 1


#CNN_input = layers.Input(shape=(num_rows, num_columns, num_channels)) #для слоя 2D
CNN_input = layers.Input(shape=(num_rows, num_columns)) #для слоя 1D

x = layers.Conv1D(128, 3, activation="relu",
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
                  bias_regularizer=regularizers.l2(1e-5),
                  activity_regularizer=regularizers.l2(1e-5)
                  )(CNN_input)

x = layers.MaxPooling1D()(x)
x = layers.Dropout(0.2)(x)

x = layers.Conv1D(128, 3, activation="relu",
                  kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
                  bias_regularizer=regularizers.l2(1e-5),
                  activity_regularizer=regularizers.l2(1e-5)
                  )(x)

x = layers.MaxPooling1D()(x)
x = layers.Dropout(0.2)(x)


x = layers.Flatten()(x)

class_labels_cnn = np.unique(ch1_cnn_df['class_label'])
num_labels = class_labels_cnn.shape[0]
print("Num_labels = ",num_labels)

CNN_output = layers.Dense(num_labels, activation='softmax')(x)

model_cnn_branch = Model(CNN_input, CNN_output, name="CNN")
print(model_cnn_branch.summary())
model_cnn_branch.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                         optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

num_rows = np.shape(xTest_cnn_ch0_cut[0])[0]
num_columns = np.shape(xTest_cnn_ch0_cut[0])[1]



checkpointer_cnn_ch0 = ModelCheckpoint(filepath= path_ws + 'weights_cnn_ch0.hdf5',
                               monitor='val_accuracy',
                               verbose=2,
                               save_best_only=True)

checkpointer_cnn_ch1 = ModelCheckpoint(filepath= path_ws + 'weights_cnn_ch1.hdf5',
                               monitor='val_accuracy',
                               verbose=2,
                               save_best_only=True)

checkpointer_cnn_ch2 = ModelCheckpoint(filepath= path_ws + 'weights_cnn_ch2.hdf5',
                               monitor='val_accuracy',
                               verbose=2,
                               save_best_only=True)

checkpointer_cnn_ch3 = ModelCheckpoint(filepath= path_ws + 'weights_cnn_ch3.hdf5',
                               monitor='val_accuracy',
                               verbose=2,
                               save_best_only=True)

# print(np.shape(xTrain_cnn_ch0_cut))
#
# xTrain_cnn_ch0_reduced =  xTrain_cnn_ch0_cut.copy()
# # xTrain_cnn_ch0_reduced [:,:5,:,0] = 0
# # xTrain_cnn_ch0_reduced [:,110:,:,0] = 0
history_cnn_ch0 = model_cnn_branch.fit(xTrain_cnn_ch0_cut, yTrain_cnn_ch0, batch_size=num_batch_size,
callbacks=[checkpointer_cnn_ch0], epochs=num_epochs, validation_data=(xTest_cnn_ch0_cut, yTest_cnn_ch0),  verbose=1)
# answer = yTrain_cnn_ch0
# # data = xTrain_cnn_ch0_cut
# data = xTrain_cnn_ch0_reduced
#
# print(np.shape(xTrain_cnn_ch1_cut))
# xTrain_cnn_ch1_reduced =  xTrain_cnn_ch1_cut.copy()
# # xTrain_cnn_ch1_reduced [:,:5,:,0] = 0
# # xTrain_cnn_ch1_reduced [:,110:,:,0] = 0
history_cnn_ch1 = model_cnn_branch.fit(xTrain_cnn_ch1_cut, yTrain_cnn_ch1, batch_size=num_batch_size,
callbacks=[checkpointer_cnn_ch1], epochs=num_epochs, validation_data=(xTest_cnn_ch1_cut, yTest_cnn_ch1),  verbose=1)
#
# answer = yTrain_cnn_ch1
# # data = xTrain_cnn_ch1_cut
# data = xTrain_cnn_ch1_reduced
#
# print(np.shape(xTrain_cnn_ch2_cut))
#
# xTrain_cnn_ch2_reduced =  xTrain_cnn_ch2_cut.copy()
# # xTrain_cnn_ch2_reduced [:,:5,:,0] = 0
# # xTrain_cnn_ch2_reduced [:,110:,:,0] = 0
history_cnn_ch2 = model_cnn_branch.fit(xTrain_cnn_ch2_cut, yTrain_cnn_ch2, batch_size=num_batch_size,
callbacks=[checkpointer_cnn_ch2], epochs=num_epochs, validation_data=(xTest_cnn_ch2_cut, yTest_cnn_ch2),  verbose=1)
# answer = yTrain_cnn_ch2
# # data = xTrain_cnn_ch2_cut
# data = xTrain_cnn_ch2_reduced
#
# print(np.shape(xTrain_cnn_ch3_cut))
#
# xTrain_cnn_ch3_reduced =  xTrain_cnn_ch3_cut.copy()
# # xTrain_cnn_ch3_reduced [:,:5,:,0] = 0
# # xTrain_cnn_ch3_reduced [:,110:,:,0] = 0
history_cnn_ch3 = model_cnn_branch.fit(xTrain_cnn_ch3_cut, yTrain_cnn_ch3, batch_size=num_batch_size,
callbacks=[checkpointer_cnn_ch3], epochs=num_epochs, validation_data=(xTest_cnn_ch3_cut, yTest_cnn_ch3),  verbose=1)
# answer = yTrain_cnn_ch3
# # data = xTrain_cnn_ch3_cut
# data = xTrain_cnn_ch3_reduced
#
#
# def attention(p):
#     expected_output = tf.cast(answer[p], tf.float32)
#     i = np.argmax(expected_output)
#
#
#     with tf.GradientTape() as tape:
#         # cast image to float
#         inputs = tf.cast(data[p:p + 1], dtype=tf.float32)
#         # watch the input pixels
#         tape.watch(inputs)
#
#         #generate the predictions
#         predictions = model_cnn_branch(inputs)
#
#         # get the loss
#         loss = tf.keras.losses.categorical_crossentropy(
#             expected_output, predictions[0]
#         )
#
#     gradients = tape.gradient(loss, inputs)
#     # reduce the RGB image to grayscale
#     grayscale_tensor = tf.reduce_sum(tf.abs(gradients), axis=-1)
#
#     # normalize the pixel values to be in the range [0, 255].
#     # the max value in the grayscale tensor will be pushed to 255.
#     # the min value will be pushed to 0.
#     normalized_tensor = tf.cast(
#         255
#         * (grayscale_tensor - tf.reduce_min(grayscale_tensor))
#         / (tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)),
#         tf.uint8,
#     )
#     return normalized_tensor, grayscale_tensor

# Evaluating the model on the training and testing set

xTrain = [xTrain_cnn_ch0_cut,xTrain_cnn_ch1_cut,xTrain_cnn_ch2_cut,xTrain_cnn_ch3_cut]
yTrain = [yTrain_cnn_ch0,yTrain_cnn_ch1,yTrain_cnn_ch2,yTrain_cnn_ch3]
xTest = [xTest_cnn_ch0_cut,xTest_cnn_ch1_cut,xTest_cnn_ch2_cut,xTest_cnn_ch3_cut]
yTest = [yTest_cnn_ch0,yTest_cnn_ch1,yTest_cnn_ch2,yTest_cnn_ch3]


for i in range(4):
    print(f'channel{i}')
    model_cnn_branch.load_weights(path_ws + f'weights_cnn_ch{i}.hdf5')
    score = model_cnn_branch.evaluate(xTrain[i], yTrain[i],verbose=0)
    print("Training Accuracy: ", score[1])

    score = model_cnn_branch.evaluate(xTest[i], yTest[i],verbose=0)
    print("Testing Accuracy: ", score[1])

history = [history_cnn_ch0,history_cnn_ch1,history_cnn_ch2,history_cnn_ch3]
plot_accuracy_and_loss(history)

# Записать xTrain, YTrain,num_labels,model_branch
params = [num_labels,list(class_labels_cnn),xTrain,xTest,yTrain,yTest]
write_data = params
datafile=open(path_ws+'params'+'_'+name+'.dat',"wb")
pickle.dump(write_data,datafile)
datafile.close()

exit(0)