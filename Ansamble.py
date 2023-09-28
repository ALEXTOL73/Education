from keras import Input
from keras import Model
from keras.layers import concatenate
import numpy as np

import keras
from keras.layers import Dense, Dropout,Flatten
from keras import regularizers
from keras import layers
from keras.callbacks import ModelCheckpoint
from FCNN_all import Xtr_fc,Xts_fc,Ytr_fc,Yts_fc,num_labels,plot_accuracy_and_loss
from CNN_all import Xtr_cnn,Xts_cnn,Ytr_cnn,Yts_cnn

# Unused libs:
# from matplotlib import pyplot as plt

xTrain_fc_ch0,xTrain_fc_ch1,xTrain_fc_ch2,xTrain_fc_ch3 = Xtr_fc[0],Xtr_fc[1],Xtr_fc[2],Xtr_fc[3]
xTest_fc_ch0,xTest_fc_ch1,xTest_fc_ch2,xTest_fc_ch3 = Xts_fc[0],Xts_fc[1],Xts_fc[2],Xts_fc[3]
yTrain_fc_ch0,yTrain_fc_ch1,yTrain_fc_ch2,yTrain_fc_ch3 = Ytr_fc[0],Ytr_fc[1],Ytr_fc[2],Ytr_fc[3]
yTest_fc_ch0,yTest_fc_ch1,yTest_fc_ch2,yTest_fc_ch3 = Yts_fc[0],Yts_fc[1],Yts_fc[2],Yts_fc[3]

xTrain_cnn_ch0_cut,xTrain_cnn_ch1_cut,xTrain_cnn_ch2_cut,xTrain_cnn_ch3_cut = Xtr_cnn[0],Xtr_cnn[1],Xtr_cnn[2],Xtr_cnn[3]
xTest_cnn_ch0_cut,xTest_cnn_ch1_cut,xTest_cnn_ch2_cut,xTest_cnn_ch3_cut = Xts_cnn[0],Xts_cnn[1],Xts_cnn[2],Xts_cnn[3]
yTrain_cnn_ch0,yTrain_cnn_ch1,yTrain_cnn_ch2,yTrain_cnn_ch3 = Ytr_cnn[0],Ytr_cnn[1],Ytr_cnn[2],Ytr_cnn[3]
yTest_cnn_ch0,yTest_cnn_ch1,yTest_cnn_ch2,yTest_cnn_ch3 = Yts_cnn[0],Yts_cnn[1],Yts_cnn[2],Yts_cnn[3]

# define two sets of inputs
input_ch0 = Input(shape=(np.shape(xTrain_fc_ch0)[1],))
input_ch1 = Input(shape=(np.shape(xTrain_fc_ch1)[1],))
input_ch2 = Input(shape=(np.shape(xTrain_fc_ch2)[1],))
input_ch3 = Input(shape=(np.shape(xTrain_fc_ch3)[1],))

# the ch0 branch operates on the first input
m_ch0 = Dense(256,
              input_shape=(np.shape(xTrain_fc_ch0)[1],),
              activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(input_ch0)

m_ch0 = Dropout(0.5)(m_ch0)
m_ch0 = Dense(256,
              activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(m_ch0)
m_ch0 = Dropout(0.5)(m_ch0)

m_ch0 = Model(inputs=input_ch0, outputs=m_ch0)

# the ch1 branch opreates on the second input
m_ch1 = Dense(256, input_shape=(np.shape(xTrain_fc_ch1)[1],),
              activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(input_ch1)

m_ch1 = Dropout(0.5)(m_ch1)
m_ch1 = Dense(256,
              activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(m_ch1)
m_ch1 = Dropout(0.5)(m_ch1)

m_ch1 = Model(inputs=input_ch1, outputs=m_ch1)

# the ch2 branch opreates on the second input
m_ch2 = Dense(256, input_shape=(np.shape(xTrain_fc_ch2)[1],),
              activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(input_ch2)

m_ch2 = Dropout(0.5)(m_ch2)
m_ch2 = Dense(256,
              activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(m_ch2)
m_ch2 = Dropout(0.5)(m_ch2)

m_ch2 = Model(inputs=input_ch2, outputs=m_ch2)

# the ch3 branch opreates on the second input
m_ch3 = Dense(256,
              input_shape=(np.shape(xTrain_fc_ch3)[1],),
              activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(input_ch3)

m_ch3 = Dropout(0.5)(m_ch3)
m_ch3 = Dense(256,
              activation="relu",
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-4))(m_ch3)
m_ch3 = Dropout(0.5)(m_ch3)

m_ch3 = Model(inputs=input_ch3, outputs=m_ch3)

# combine the output of the all branches
combined = concatenate([m_ch0.output, m_ch1.output, m_ch2.output, m_ch3.output])

# apply a FC layer and softmax
all_ch = Dense(256,
               activation="relu",
               kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=regularizers.l2(1e-4),
               activity_regularizer=regularizers.l2(1e-4))(combined)

all_ch = Dropout(0.5)(all_ch)
all_ch = Dense(num_labels, activation='softmax')(all_ch)

model_fc_ensamble = Model(inputs=[m_ch0.input, m_ch1.input, m_ch2.input, m_ch3.input], outputs=all_ch)

# Compile the model

model_fc_ensamble.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                          optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
model_fc_ensamble.summary()
# %%

num_epochs = 200
num_batch_size = 64

checkpointer_fc_ensamble = ModelCheckpoint(filepath='d:/Project/Weights/Ansamble/weights_fc_ansamble.hdf5',
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_best_only=True)

history_fc_ensamble = model_fc_ensamble.fit(x=[xTrain_fc_ch0, xTrain_fc_ch1, xTrain_fc_ch2, xTrain_fc_ch3],
                                            y=yTrain_fc_ch0,
                                            batch_size=num_batch_size,
                                            epochs=num_epochs,
                                            validation_data=(
                                                [xTest_fc_ch0, xTest_fc_ch1, xTest_fc_ch2, xTest_fc_ch3],
                                                yTest_fc_ch0
                                            ),
                                            callbacks=[checkpointer_fc_ensamble],
                                            verbose=1
                                            )

# %%
history = [history_fc_ensamble]
plot_accuracy_and_loss(history)


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
                           optimizer=keras.optimizers.RMSprop(learning_rate=0.001))
model_cnn_ensamble.summary()

# %%

num_epochs = 30
num_batch_size = 64

checkpointer_cnn_ensamble = ModelCheckpoint(filepath='d:/Project/Weights/Ansamble/weights_cnn_ansamble.hdf5',
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