
import numpy as np
import pickle
from main import *

import keras
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from keras import layers
from keras.callbacks import ModelCheckpoint
from History_Plots import plot_accuracy_and_loss

datafile_rc = open(path_w+'/CNN/'+'params'+'_'+'CNN'+'.dat',"rb")
read_data_c= pickle.load(datafile_rc)
num_labels = num_labels = read_data_c[0]
Xtr_cnn = read_data_c[2]
Xts_cnn = read_data_c[3]
Ytr_cnn = read_data_c[4]
Yts_cnn = read_data_c[5]
datafile_rc.close()

xTrain_cnn_ch0_cut,xTrain_cnn_ch1_cut,xTrain_cnn_ch2_cut,xTrain_cnn_ch3_cut = Xtr_cnn[0],Xtr_cnn[1],Xtr_cnn[2],Xtr_cnn[3]
xTest_cnn_ch0_cut,xTest_cnn_ch1_cut,xTest_cnn_ch2_cut,xTest_cnn_ch3_cut = Xts_cnn[0],Xts_cnn[1],Xts_cnn[2],Xts_cnn[3]
yTrain_cnn_ch0,yTrain_cnn_ch1,yTrain_cnn_ch2,yTrain_cnn_ch3 = Ytr_cnn[0],Ytr_cnn[1],Ytr_cnn[2],Ytr_cnn[3]
yTest_cnn_ch0,yTest_cnn_ch1,yTest_cnn_ch2,yTest_cnn_ch3 = Yts_cnn[0],Yts_cnn[1],Yts_cnn[2],Yts_cnn[3]

# %%
def StackConv(x, filt_num=16, filt_size=3, dropout_rate=0.5):
    x = layers.Conv2D(filt_num, filt_size, activation="relu",
                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      activity_regularizer=regularizers.l2(1e-5)
                      )(x)

    x = layers.MaxPooling2D()(x)
    x = Dropout(dropout_rate)(x)

    return x


# %%
from keras import Input
from keras import Model
from keras.layers import concatenate

num_rows = np.shape(xTest_cnn_ch0_cut[0])[0]
num_columns = np.shape(xTest_cnn_ch0_cut[0])[1]
num_channels = 1


input_ch0 = Input(shape=(num_rows, num_columns, num_channels))
input_ch1 = Input(shape=(num_rows, num_columns, num_channels))
input_ch2 = Input(shape=(num_rows, num_columns, num_channels))
input_ch3 = Input(shape=(num_rows, num_columns, num_channels))

x_ch0 = StackConv(input_ch0, 32, 5, 0.3)
x_ch0 = StackConv(x_ch0, 64, 5, 0.3)
x_ch0 = StackConv(x_ch0, 128, 3, 0.3)

x_ch1 = StackConv(input_ch1, 32, 5, 0.3)
x_ch1 = StackConv(x_ch1, 64, 5, 0.3)
x_ch1 = StackConv(x_ch1, 128, 3, 0.3)

x_ch2 = StackConv(input_ch2, 32, 5, 0.3)
x_ch2 = StackConv(x_ch2, 64, 5, 0.3)
x_ch2 = StackConv(x_ch2, 128, 3, 0.3)

x_ch3 = StackConv(input_ch3, 32, 5, 0.3)
x_ch3 = StackConv(x_ch3, 64, 5, 0.3)
x_ch3 = StackConv(x_ch3, 128, 3, 0.3)

# combine the output of the all branches
combined = layers.Average()([x_ch0, x_ch1, x_ch2, x_ch3])

combined = Flatten()(combined)
# apply a FC layer and softmax
all_ch = Dense(128,
               activation="relu",
               kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=regularizers.l2(1e-4),
               activity_regularizer=regularizers.l2(1e-4))(combined)

all_ch = Dropout(0.2)(all_ch)
all_ch = Dense(num_labels, activation='softmax')(all_ch)

model_cnn_ensamble = Model(inputs=[input_ch0, input_ch1, input_ch2, input_ch3], outputs=all_ch)

# Compile the model

model_cnn_ensamble.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                           optimizer=keras.optimizers.Adam(learning_rate=0.001))
model_cnn_ensamble.summary()

# %%

num_epochs = 30
num_batch_size = 64

checkpointer_cnn_ensamble = ModelCheckpoint(filepath='d:/Project/Weights/AnsambleCNN/weights_cnn_ansamble.hdf5',
                                            monitor='val_accuracy',
                                            verbose=1,
                                            save_best_only=True)

history_cnn_ensamble = model_cnn_ensamble.fit(
    x=[xTrain_cnn_ch0_cut, xTrain_cnn_ch1_cut, xTrain_cnn_ch2_cut, xTrain_cnn_ch3_cut],
    y=yTrain_cnn_ch0,
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(
        [xTest_cnn_ch0_cut, xTest_cnn_ch1_cut, xTest_cnn_ch2_cut, xTest_cnn_ch3_cut],
        yTest_cnn_ch0
    ),
    callbacks=[checkpointer_cnn_ensamble],
    verbose=1
    )

history = [history_cnn_ensamble]
plot_accuracy_and_loss(history)

xTrain = [xTrain_cnn_ch0_cut,xTrain_cnn_ch1_cut,xTrain_cnn_ch2_cut,xTrain_cnn_ch3_cut]
yTrain = [yTrain_cnn_ch0,yTrain_cnn_ch1,yTrain_cnn_ch2,yTrain_cnn_ch3]
xTest = [xTest_cnn_ch0_cut,xTest_cnn_ch1_cut,xTest_cnn_ch2_cut,xTest_cnn_ch3_cut]
yTest = [yTest_cnn_ch0,yTest_cnn_ch1,yTest_cnn_ch2,yTest_cnn_ch3]

print(f'channel')
model_cnn_ensamble.load_weights(path_w + '/AnsambleCNN/' + f'weights_cnn_ansamble.hdf5')
score = model_cnn_ensamble.evaluate(xTrain, yTrain, verbose=0)
print("Training Accuracy: ", score[1])
score = model_cnn_ensamble.evaluate(xTest, yTest, verbose=0)
print("Testing Accuracy: ", score[1])

# Записать xTrain, YTrain,num_labels
params = [num_labels,xTrain,xTest,yTrain,yTest]
write_data = params
datafile=open(path_w + '/AnsambleCNN/'+'params'+'_CNN.dat',"wb")
pickle.dump(write_data,datafile)
datafile.close()

exit()