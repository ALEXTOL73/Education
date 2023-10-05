from __future__ import division
import warnings
warnings.filterwarnings('ignore')

from Preprocess import features_creation
from History_Plots import plot_accuracy_and_loss

import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from sklearn.preprocessing import StandardScaler



print(device_lib.list_local_devices())

path_ch0 = 'd:/Project/full dataset/audio/ch0/'
path_ch1 = 'd:/Project/full dataset/audio/ch1/'
path_ch2 = 'd:/Project/full dataset/audio/ch2/'
path_ch3 = 'd:/Project/full dataset/audio/ch3/'

df_fc_ch0,columns_fc,_,_ =  features_creation(path_ch0,sr = None, duration = 0.3,
                                                mono = False,mfccs_num = 128,hop_length = 512)
df_fc_ch1,_,_,_ =  features_creation(path_ch1,sr = None, duration = 0.3,
                                                mono = False,mfccs_num = 128,hop_length = 512)
df_fc_ch2,_,_,_ =  features_creation(path_ch2,sr = None, duration = 0.3,
                                                mono = False,mfccs_num = 128,hop_length = 512)
df_fc_ch3,_,_,_ =  features_creation(path_ch3,sr = None, duration = 0.3,
                                                mono = False,mfccs_num = 128,hop_length = 512)

print('\n size of features for FNN')
print(np.shape(df_fc_ch0))
print(np.shape(df_fc_ch1))
print(np.shape(df_fc_ch2))
print(np.shape(df_fc_ch3))


# Convert features into a Panda dataframe for fully connected NN
ch0_fc_df = pd.DataFrame(columns = columns_fc)
ch1_fc_df = pd.DataFrame(columns = columns_fc)
ch2_fc_df = pd.DataFrame(columns = columns_fc)
ch3_fc_df = pd.DataFrame(columns = columns_fc)
# num_labels = np.unique(ch1_fc_df['class_label']).shape[0]
# print(num_labels)

for i in range(np.shape(df_fc_ch0)[0]):
    ch0_fc_df.loc[i] = df_fc_ch0[i]
    ch1_fc_df.loc[i] = df_fc_ch1[i]
    ch2_fc_df.loc[i] = df_fc_ch2[i]
    ch3_fc_df.loc[i] = df_fc_ch3[i]

#separate data into features and class labels
features_fc_ch0    = ch0_fc_df.iloc[:,0:-1]
classes_fc_ch0     = ch0_fc_df.iloc[:,-1]
classes_fc_ch0_str = classes_fc_ch0.astype(str)

features_fc_ch1    = ch1_fc_df.iloc[:,0:-1]
classes_fc_ch1     = ch1_fc_df.iloc[:,-1]
classes_fc_ch1_str = classes_fc_ch1.astype(str)

features_fc_ch2    = ch2_fc_df.iloc[:,0:-1]
classes_fc_ch2     = ch2_fc_df.iloc[:,-1]
classes_fc_ch2_str = classes_fc_ch2.astype(str)

features_fc_ch3    = ch3_fc_df.iloc[:,0:-1]
classes_fc_ch3     = ch3_fc_df.iloc[:,-1]
classes_fc_ch3_str = classes_fc_ch3.astype(str)


scaler = StandardScaler()

features_fc_ch0_scaled = scaler.fit_transform(features_fc_ch0)
features_fc_ch1_scaled = scaler.fit_transform(features_fc_ch1)
features_fc_ch2_scaled = scaler.fit_transform(features_fc_ch2)
features_fc_ch3_scaled = scaler.fit_transform(features_fc_ch3)



#tsne = TSNE(n_components=2, random_state=42, perplexity = 11, n_jobs=-1)

#tsne_transformed_fc_ch2 = tsne.fit_transform(features_fc_ch2_scaled)
#tsne_transformed_fc_ch3 = tsne.fit_transform(features_fc_ch3_scaled)

#tsne_features = [tsne_transformed_fc_ch0,tsne_transformed_fc_ch1,tsne_transformed_fc_ch2,tsne_transformed_fc_ch3]
#print(np.shape(tsne_features))

classes = [classes_fc_ch0_str, classes_fc_ch1_str,classes_fc_ch2_str,classes_fc_ch3_str]
print(np.shape(classes))

# pca = PCA()
# explained_threshold = 0.999
# X = [features_fc_ch0_scaled,features_fc_ch1_scaled,features_fc_ch2_scaled,features_fc_ch3_scaled]

# Convert features and corresponding classification labels into numpy arrays

# Encode the classification labels
le = LabelEncoder()
y_pca_cat_ch0 = to_categorical(le.fit_transform(ch0_fc_df['class_label']))
y_pca_cat_ch1 = to_categorical(le.fit_transform(ch1_fc_df['class_label']))
y_pca_cat_ch2 = to_categorical(le.fit_transform(ch2_fc_df['class_label']))
y_pca_cat_ch3 = to_categorical(le.fit_transform(ch3_fc_df['class_label']))

xTrain_fc_ch0, xTest_fc_ch0, yTrain_fc_ch0,yTest_fc_ch0 = train_test_split(features_fc_ch0_scaled,
                                                y_pca_cat_ch0,test_size = 0.2, random_state = 0)
xTrain_fc_ch1, xTest_fc_ch1, yTrain_fc_ch1,yTest_fc_ch1 = train_test_split(features_fc_ch1_scaled,
                                                y_pca_cat_ch1,test_size = 0.2, random_state = 0)
xTrain_fc_ch2, xTest_fc_ch2, yTrain_fc_ch2,yTest_fc_ch2 = train_test_split(features_fc_ch2_scaled,
                                                y_pca_cat_ch2,test_size = 0.2, random_state = 0)
xTrain_fc_ch3, xTest_fc_ch3, yTrain_fc_ch3,yTest_fc_ch3 = train_test_split(features_fc_ch3_scaled,
                                                y_pca_cat_ch3,test_size = 0.2, random_state = 0)
# для передачи в Ansamble.py
Xtr_fc = [xTrain_fc_ch0,xTrain_fc_ch1,xTrain_fc_ch2,xTrain_fc_ch3]
Xts_fc = [xTest_fc_ch0,xTest_fc_ch1,xTest_fc_ch2,xTest_fc_ch3]
Ytr_fc = [yTrain_fc_ch0,yTrain_fc_ch1,yTrain_fc_ch2,yTrain_fc_ch3]
Yts_fc = [yTest_fc_ch0,yTest_fc_ch1,yTest_fc_ch2,yTest_fc_ch3]

#Construct FCNN
num_labels = np.unique(ch1_fc_df['class_label']).shape[0]
print(num_labels)


# Construct model
model_fc_branch = Sequential()

model_fc_branch.add(Dense(512,
                activation = 'relu',
                input_shape=(np.shape(xTrain_fc_ch0)[1],),
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-4)))

model_fc_branch.add(Dropout(0.2))

model_fc_branch.add(Dense(512,
                activation = 'relu',
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-4)))

model_fc_branch.add(Dropout(0.2))

model_fc_branch.add(Dense(num_labels,activation = 'softmax',
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-4)))
# Compile the model

model_fc_branch.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                        optimizer= keras.optimizers.RMSprop(learning_rate=0.001))
model_fc_branch.summary()

num_epochs = 20
num_batch_size = 128

checkpointer_fc_ch0 = ModelCheckpoint(filepath='d:/Project/Weights/FCNN/weights_fc_ch0.hdf5',
                               monitor='val_accuracy',
                               verbose=1,
                               save_best_only=True)

checkpointer_fc_ch1 = ModelCheckpoint(filepath='d:/Project/Weights/FCNN/weights_fc_ch1.hdf5',
                               monitor='val_accuracy',
                               verbose=1,
                               save_best_only=True)

checkpointer_fc_ch2 = ModelCheckpoint(filepath='d:/Project/Weights/FCNN/weights_fc_ch2.hdf5',
                               monitor='val_accuracy',
                               verbose=1,
                               save_best_only=True)

checkpointer_fc_ch3 = ModelCheckpoint(filepath='d:/Project/Weights/FCNN/weights_fc_ch3.hdf5',
                               monitor='val_accuracy',
                               verbose=1,
                               save_best_only=True)

#Обучаем модель
history_fc_ch0 = model_fc_branch.fit(xTrain_fc_ch0, yTrain_fc_ch0, batch_size=num_batch_size, epochs=num_epochs,
                        validation_data=(xTest_fc_ch0, yTest_fc_ch0), callbacks=[checkpointer_fc_ch0], verbose=1)
history_fc_ch1 = model_fc_branch.fit(xTrain_fc_ch1, yTrain_fc_ch1, batch_size=num_batch_size, epochs=num_epochs,
                        validation_data=(xTest_fc_ch1, yTest_fc_ch1), callbacks=[checkpointer_fc_ch1], verbose=1)
history_fc_ch2 = model_fc_branch.fit(xTrain_fc_ch2, yTrain_fc_ch2, batch_size=num_batch_size, epochs=num_epochs,
                        validation_data=(xTest_fc_ch2, yTest_fc_ch2), callbacks=[checkpointer_fc_ch2], verbose=1)
history_fc_ch3 = model_fc_branch.fit(xTrain_fc_ch3, yTrain_fc_ch3, batch_size=num_batch_size, epochs=num_epochs,
                        validation_data=(xTest_fc_ch3, yTest_fc_ch3), callbacks=[checkpointer_fc_ch3], verbose=1)



history = [history_fc_ch0, history_fc_ch1, history_fc_ch2, history_fc_ch3]
plot_accuracy_and_loss(history)

# Evaluating the model on the training and testing set

xTrain = [xTrain_fc_ch0, xTrain_fc_ch1, xTrain_fc_ch2, xTrain_fc_ch3]
yTrain = [yTrain_fc_ch0, yTrain_fc_ch1, yTrain_fc_ch2, yTrain_fc_ch3]

xTest = [xTest_fc_ch0, xTest_fc_ch1, xTest_fc_ch2, xTest_fc_ch3]
yTest = [yTest_fc_ch0, yTest_fc_ch1, yTest_fc_ch2, yTest_fc_ch3]


for i in range(4):
    print(f'channel{i}')
    model_fc_branch.load_weights(f'd:/Project/Weights/FCNN/weights_fc_ch{i}.hdf5')
    score = model_fc_branch.evaluate(xTrain[i], yTrain[i], verbose=0)
    print("Training Accuracy: ", score[1])

    score = model_fc_branch.evaluate(xTest[i], yTest[i], verbose=0)
    print("Testing Accuracy: ", score[1])

