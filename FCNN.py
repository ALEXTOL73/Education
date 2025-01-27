from __future__ import division
import warnings
warnings.filterwarnings('ignore')

import tensorflow.python.framework.config
from Preprocess import features_creation
from History_Plots import plot_accuracy_and_loss

import pandas as pd
import numpy as np
import time
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
import pickle
from main import *

GPU = tensorflow.config.list_physical_devices('GPU')

print('Num GPUs available:', len(GPU))

start = time.perf_counter()

# Подготовка данных: Convert features into a Panda dataframe for FCNN, Separate data into features and class labels
for i in range(num_folders):
    exec(f"path_ch{i} = paths[i]")
    exec(f"df_fc_ch{i},columns_fc,_,_ = features_creation(path_ch{i}, sr, duration, mono, mfccs_num, hop_length)")
    exec(f"ch{i}_fc_df = pd.DataFrame(columns=columns_fc)")
    for j in range(np.shape(df_fc_ch0)[0]):
        exec(f"ch{i}_fc_df.loc[j] = df_fc_ch{i}[j]")
    exec(f"features_fc_ch{i} = ch{i}_fc_df.iloc[:, 0:-1]")
    exec(f"classes_fc_ch{i}     = ch{i}_fc_df.iloc[:,-1]")
    exec(f"classes_fc_ch{i}_str = classes_fc_ch{i}.astype(str)")


print('size of features for FCNN :')
print(np.shape(df_fc_ch0))

classes = []
scaler = StandardScaler()
for i in range(num_folders):
    exec(f"features_fc_ch{i}_scaled = scaler.fit_transform(features_fc_ch{i})")
    exec(f"classes.append(classes_fc_ch{i}_str)")
#tsne = TSNE(n_components=2, random_state=42, perplexity = 11, n_jobs=-1)

#tsne_transformed_fc_ch2 = tsne.fit_transform(features_fc_ch2_scaled)
#tsne_transformed_fc_ch3 = tsne.fit_transform(features_fc_ch3_scaled)

#tsne_features = [tsne_transformed_fc_ch0,tsne_transformed_fc_ch1,tsne_transformed_fc_ch2,tsne_transformed_fc_ch3]
#print(np.shape(tsne_features))

# classes = [classes_fc_ch0_str, classes_fc_ch1_str,classes_fc_ch2_str,classes_fc_ch3_str,classes_fc_ch4_str,
#            classes_fc_ch5_str, classes_fc_ch6_str, classes_fc_ch7_str]
print(np.shape(classes))

# pca = PCA()
# explained_threshold = 0.999
# X = [features_fc_ch0_scaled,features_fc_ch1_scaled,features_fc_ch2_scaled,features_fc_ch3_scaled]

# Convert features and corresponding classification labels into numpy arrays

# Encode the classification labels
le = LabelEncoder()

for i in range(num_folders):
    exec(f"y_pca_cat_ch{i} = to_categorical(le.fit_transform(ch{i}_fc_df['class_label']))")
    exec(f"xTrain_fc_ch{i}, xTest_fc_ch{i}, yTrain_fc_ch{i}, yTest_fc_ch{i} = train_test_split(features_fc_ch{i}_scaled,\
                                                                    y_pca_cat_ch{i}, test_size=0.15, random_state=0)")

# для передачи в Ansamble.py

#Construct FCNN
class_labels_fc = np.unique(ch1_fc_df['class_label'])
num_labels = class_labels_fc.shape[0]

# Construct model
model_fc_branch = Sequential()

model_fc_branch.add(Dense(512,
                activation = 'relu',
                input_shape=(np.shape(xTrain_fc_ch0)[1],),
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-4)))

model_fc_branch.add(Dropout(0.4))

model_fc_branch.add(Dense(512,
                activation = 'relu',
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-4)))

model_fc_branch.add(Dropout(0.4))

model_fc_branch.add(Dense(num_labels,activation = 'softmax',
                kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                bias_regularizer=regularizers.l2(1e-4),
                activity_regularizer=regularizers.l2(1e-4)))

# Compile the model
model_fc_branch.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                        optimizer= keras.optimizers.RMSprop(learning_rate=0.001))
model_fc_branch.summary()

for i in range(num_folders):
    exec(f"checkpointer_fc_ch{i} = ModelCheckpoint(filepath= path_ws + 'weights_fc_ch{i}.hdf5',\
                               monitor='val_accuracy',verbose=2,save_best_only=True)")
    exec(f"history_fc_ch{i} = model_fc_branch.fit(xTrain_fc_ch{i}, yTrain_fc_ch{i}, batch_size=num_batch_size,\
callbacks=[checkpointer_fc_ch{i}],epochs=num_epochs,validation_data=(xTest_fc_ch{i},yTest_fc_ch{i}),verbose=1)")


# Evaluating the model on the training and testing set

xTrain = [eval(f"xTrain_fc_ch{i}") for i in range(num_folders)]
yTrain = [eval(f"yTrain_fc_ch{i}") for i in range(num_folders)]
xTest = [eval(f"xTest_fc_ch{i}") for i in range(num_folders)]
yTest = [eval(f"yTest_fc_ch{i}") for i in range(num_folders)]
history = [eval(f"history_fc_ch{i}") for i in range(num_folders)]


plot_accuracy_and_loss(history)

for i in range(num_folders):
    print(f'channel{i}')
    model_fc_branch.load_weights(path_ws + f'weights_fc_ch{i}.hdf5')
    score = model_fc_branch.evaluate(xTrain[i], yTrain[i], verbose=0)
    print("Training Accuracy: ", round(score[1], 3))
    score = model_fc_branch.evaluate(xTest[i], yTest[i], verbose=0)
    print("Testing Accuracy: ", round(score[1], 3))

finish = time.perf_counter()
time = round((finish - start), 3)

print("\nВремя обработки = ", time)
# Записать xTrain, YTrain,num_labels,model_branch

params = [num_labels, list(class_labels_fc), xTrain, xTest, yTrain, yTest]
write_data = params
datafile = open(path_ws+'params'+'_'+name+'.dat', "wb")
pickle.dump(write_data, datafile)
datafile.close()


exit(0)

