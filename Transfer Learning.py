import numpy as np
import seaborn as sns
import pickle
from matplotlib import pyplot as plt
from keras.utils import to_categorical
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import Model
from main import *
import keras
import time
from sklearn.metrics import confusion_matrix
from keras.layers import Dense, Dropout, Flatten
from keras.models import load_model
from keras import regularizers
from keras import layers
from keras.callbacks import ModelCheckpoint
from History_Plots import plot_accuracy_and_loss
from keras import Input
from keras import Model as model
from keras.layers import concatenate
from Preprocess import features_creation
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from Predict_model import in_top_ansamble
from Frases import right_wrong
import warnings
warnings.filterwarnings('ignore')

duration = 0.3

# Подготовка данных: Convert features into a Panda dataframe for CNN, Separate data into features and class labels
for i in range(num_folders):
    exec(f"path_ch{i} = paths[i]")
    exec(f"df_ch{i}, columns = features_creation(path_ch{i})")
    exec(f"ch{i}_df = pd.DataFrame(columns=columns)")
    for j in range(np.shape(df_ch0)[0]):
        exec(f"ch{i}_df.loc[j] = df_ch{i}[j]")
    exec(f"features_ch{i} = ch{i}_df.iloc[:, 0:-1]")
    exec(f"hue_name_ch{i} = ch{i}_df.iloc[:, -1]")
    exec(f"hue_names_ch{i}_int = hue_name_ch{i}.astype(str)")

# df_ch0, columns = features_creation(path_ch0)
# df_ch1, columns = features_creation(path_ch1)
# df_ch2, columns = features_creation(path_ch2)
# df_ch3, columns = features_creation(path_ch3)
print('OK')
# Convert into a Panda dataframe

# ch0_df = pd.DataFrame(columns=columns)
# ch1_df = pd.DataFrame(columns=columns)
# ch2_df = pd.DataFrame(columns=columns)
# ch3_df = pd.DataFrame(columns=columns)

# for i in range(np.shape(df_ch0)[0]):
#     ch0_df.loc[i] = df_ch0[i]
#     ch1_df.loc[i] = df_ch1[i]
#     ch2_df.loc[i] = df_ch2[i]
#     ch3_df.loc[i] = df_ch3[i]

ch0_df.head(30)

# features_ch0 = ch0_df.iloc[:, 0:-1]
# hue_name_ch0 = ch0_df.iloc[:, -1]
# hue_names_ch0_int = hue_name_ch0.astype(str)
#
# features_ch1 = ch1_df.iloc[:, 0:-1]
# hue_name_ch1 = ch1_df.iloc[:, -1]
# hue_names_ch1_int = hue_name_ch1.astype(str)
#
# features_ch2 = ch2_df.iloc[:, 0:-1]
# hue_name_ch2 = ch2_df.iloc[:, -1]
# hue_names_ch2_int = hue_name_ch2.astype(str)
#
# features_ch3 = ch3_df.iloc[:, 0:-1]
# hue_name_ch3 = ch3_df.iloc[:, -1]
# hue_names_ch3_int = hue_name_ch3.astype(str)

scaler = StandardScaler()

for i in range(num_folders):
    exec(f"features_ch{i}_scaled = scaler.fit_transform(features_ch{i})")
    exec(f"x_pca_ch{i} = features_ch{i}_scaled")
    exec(f"y_pca_ch{i} = ch{i}_df['class_label']")

# features_ch0_scaled = scaler.fit_transform(features_ch0)
# features_ch1_scaled = scaler.fit_transform(features_ch1)
# features_ch2_scaled = scaler.fit_transform(features_ch2)
# features_ch3_scaled = scaler.fit_transform(features_ch3)

# x_pca_ch0 = features_ch0_scaled
# y_pca_ch0 = ch0_df['class_label']
#
# x_pca_ch1 = features_ch1_scaled
# y_pca_ch1 = ch1_df['class_label']
#
# x_pca_ch2 = features_ch2_scaled
# y_pca_ch2 = ch2_df['class_label']
#
# x_pca_ch3 = features_ch3_scaled
# y_pca_ch3 = ch3_df['class_label']

# Convert features and corresponding classification labels into numpy arrays

# Encode the classification labels
le = LabelEncoder()
for i in range(num_folders):
    exec(f"y_pca_cat_ch{i} = to_categorical(le.fit_transform(y_pca_ch{i}))")
    exec(f"xTrain_ch{i}, xTest_ch{i}, yTrain_ch{i}, yTest_ch{i} = \
         train_test_split(x_pca_ch{i}, y_pca_cat_ch{i}, test_size=0.2, random_state=0)")


# y_pca_cat_ch0 = to_categorical(le.fit_transform(y_pca_ch0))
# y_pca_cat_ch1 = to_categorical(le.fit_transform(y_pca_ch1))
# y_pca_cat_ch2 = to_categorical(le.fit_transform(y_pca_ch2))
# y_pca_cat_ch3 = to_categorical(le.fit_transform(y_pca_ch3))



#

# %%

num_labels = np.unique(y_pca_ch0[:]).shape[0]

model1 = load_model('weights_all_ch.hdf5')

for layer in model1.layers:
    layer.trainable = False

model1.layers.pop()  # удаляем последний слой классификации
# model1.layers.pop() # удаляем dropout
# model1.layers.pop() # удаляем полносвязный слой

# fc = Dense(256,activation='relu')(model1.layers[-1].output)
# drop = Dropout(0.5)(fc)
fc2 = Dense(num_labels, activation='softmax')(model1.layers[-1].output)

model1 = Model(inputs=model1.input, outputs=fc2)

model1.compile(loss='categorical_crossentropy', metrics=['accuracy'],
               optimizer=keras.optimizers.RMSprop(learning_rate=0.01))
model1.summary()
# %%
from keras.callbacks import ModelCheckpoint
from datetime import datetime

num_epochs = 600
num_batch_size = 16

checkpointer_all_ch = ModelCheckpoint(filepath='tl_weights_all_ch.hdf5',
                                      monitor='val_accuracy',
                                      verbose=1,
                                      save_best_only=True)
# %%
history_tl = model1.fit(
    x=[xTrain_ch0, xTrain_ch1, xTrain_ch2, xTrain_ch3],
    y=yTrain_ch0,
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(
        [xTest_ch0, xTest_ch1, xTest_ch2, xTest_ch3],
        yTest_ch0),
    callbacks=[checkpointer_all_ch],
    verbose=1)
# %%
from matplotlib import pyplot as plt

# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
acc_ch0 = history_tl.history['accuracy']  # history.history is a dictionary with 'accuracy' and etc. being keys
val_acc_ch0 = history_tl.history['val_accuracy']
loss_ch0 = history_tl.history['loss']
val_loss_ch0 = history_tl.history['val_loss']

epochs = range(len(acc_ch0))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
fig = plt.figure(figsize=(12, 8))
plt.plot(epochs, acc_ch0, 'b--', label='Training accuracy ch0')
plt.plot(epochs, val_acc_ch0, 'b', label='Validation accuracy ch0')

plt.title('Training and validation accuracy')
plt.legend()
plt.show()
fig.savefig("accuracy.png")
# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
fig = plt.figure(figsize=(12, 8))
plt.plot(epochs, loss_ch0, 'b--', label='Training Loss ch0')
plt.plot(epochs, val_loss_ch0, 'b', label='Validation Loss  ch0')

plt.title('Training and validation loss')
plt.legend()
plt.show()
fig.savefig("loss.png")


# %%
def ensamble_predict(phrase1_scaled_ch0, phrase1_scaled_ch1, phrase1_scaled_ch2, phrase1_scaled_ch3):
    model.load_weights('tl_weights_all_ch.hdf5')
    predictions_all_ch = model.predict([phrase1_scaled_ch0,
                                        phrase1_scaled_ch1,
                                        phrase1_scaled_ch2,
                                        phrase1_scaled_ch3])

    answer = []
    for i in range(np.shape(predictions_all_ch)[0]):

        answer_temp = []
        for pp in predictions_all_ch[i]:
            answer_temp.append(category[np.argmax(predictions_all_ch[i])])
            predictions_all_ch[i][np.argmax(predictions_all_ch[i])] = -np.inf

        print(answer_temp[0:5])
        answer.append(answer_temp)
    return answer


# %%
columns = ensamble_predict(phrase1_scaled_ch0, phrase1_scaled_ch1, phrase1_scaled_ch2, phrase1_scaled_ch3)
right_pr1, wrong_pr1, text_pr1 = right_wrong(columns, 5)

columns = ensamble_predict(phrase2_scaled_ch0, phrase2_scaled_ch1, phrase2_scaled_ch2, phrase2_scaled_ch3)
right_pr2, wrong_pr2, text_pr2 = right_wrong(columns, 5)

columns = ensamble_predict(phrase3_scaled_ch0, phrase3_scaled_ch1, phrase3_scaled_ch2, phrase3_scaled_ch3)
right_pr3, wrong_pr3, text_pr3 = right_wrong(columns, 5)

columns = ensamble_predict(phrase4_scaled_ch0, phrase4_scaled_ch1, phrase4_scaled_ch2, phrase4_scaled_ch3)
right_pr4, wrong_pr4, text_pr4 = right_wrong(columns, 5)

columns = ensamble_predict(phrase5_scaled_ch0, phrase5_scaled_ch1, phrase5_scaled_ch2, phrase5_scaled_ch3)
right_pr5, wrong_pr5, text_pr5 = right_wrong(columns, 5)

print(right_pr1, right_pr2, right_pr3, right_pr4, right_pr4)
print(wrong_pr1, wrong_pr2, wrong_pr3, wrong_pr4, wrong_pr5)
print(text_pr1, '\n\n', text_pr2, '\n\n', text_pr3, '\n\n', text_pr4, '\n\n', text_pr5)
# %%
from keras.models import load_model

num_labels = np.unique(y_pca_ch0[:]).shape[0]

model1 = load_model('weights_all_ch.hdf5')

for layer in model1.layers:
    layer.trainable = False

model1.layers[-1].trainable = True
model1.compile(loss='categorical_crossentropy', metrics=['accuracy'],
               optimizer=keras.optimizers.RMSprop(learning_rate=0.0005))
model1.summary()
# %%
num_epochs = 600
num_batch_size = 16

checkpointer_all_ch = ModelCheckpoint(filepath='tl_weights_all_ch.hdf5',
                                      monitor='val_loss',
                                      verbose=1,
                                      save_best_only=True)

history_tl = model1.fit(
    x=[xTrain_ch0, xTrain_ch1, xTrain_ch2, xTrain_ch3],
    y=yTrain_ch0,
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(
        [xTest_ch0, xTest_ch1, xTest_ch2, xTest_ch3],
        yTest_ch0),
    callbacks=[checkpointer_all_ch],
    verbose=1)


# %%
def ensamble_predict(phrase1_scaled_ch0, phrase1_scaled_ch1, phrase1_scaled_ch2, phrase1_scaled_ch3):
    model = load_model('tl_weights_all_ch.hdf5')
    predictions_all_ch = model.predict([phrase1_scaled_ch0,
                                        phrase1_scaled_ch1,
                                        phrase1_scaled_ch2,
                                        phrase1_scaled_ch3])

    answer = []
    for i in range(np.shape(predictions_all_ch)[0]):

        answer_temp = []
        for pp in predictions_all_ch[i]:
            answer_temp.append(category[np.argmax(predictions_all_ch[i])])
            predictions_all_ch[i][np.argmax(predictions_all_ch[i])] = -np.inf

        print(answer_temp[0:5])
        answer.append(answer_temp)
    return answer


# %%
columns = ensamble_predict(phrase1_scaled_ch0, phrase1_scaled_ch1, phrase1_scaled_ch2, phrase1_scaled_ch3)
right_pr1, wrong_pr1, text_pr1 = right_wrong(columns, 5)

columns = ensamble_predict(phrase2_scaled_ch0, phrase2_scaled_ch1, phrase2_scaled_ch2, phrase2_scaled_ch3)
right_pr2, wrong_pr2, text_pr2 = right_wrong(columns, 5)

columns = ensamble_predict(phrase3_scaled_ch0, phrase3_scaled_ch1, phrase3_scaled_ch2, phrase3_scaled_ch3)
right_pr3, wrong_pr3, text_pr3 = right_wrong(columns, 5)

columns = ensamble_predict(phrase4_scaled_ch0, phrase4_scaled_ch1, phrase4_scaled_ch2, phrase4_scaled_ch3)
right_pr4, wrong_pr4, text_pr4 = right_wrong(columns, 5)

columns = ensamble_predict(phrase5_scaled_ch0, phrase5_scaled_ch1, phrase5_scaled_ch2, phrase5_scaled_ch3)
right_pr5, wrong_pr5, text_pr5 = right_wrong(columns, 5)

print(right_pr1, right_pr2, right_pr3, right_pr4, right_pr4)
print(wrong_pr1, wrong_pr2, wrong_pr3, wrong_pr4, wrong_pr5)
