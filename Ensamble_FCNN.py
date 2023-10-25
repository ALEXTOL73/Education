from keras import Input
from keras import Model
from keras.layers import concatenate
import numpy as np
import pickle
from main import *

import keras
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from History_Plots import plot_accuracy_and_loss

# Читать из файлов xTrain,xTest,yTrain,yTest,num_labels(если пустые, то запуск FCNN,CNN)
datafile_rf = open(path_w+'/FCNN/'+'params'+'_'+'FCNN'+'.dat',"rb")
read_data_f= pickle.load(datafile_rf)
num_labels = read_data_f[0]
Xtr_fc = read_data_f[2]
Xts_fc = read_data_f[3]
Ytr_fc = read_data_f[4]
Yts_fc = read_data_f[5]
datafile_rf.close()



xTrain_fc_ch0,xTrain_fc_ch1,xTrain_fc_ch2,xTrain_fc_ch3 = Xtr_fc[0],Xtr_fc[1],Xtr_fc[2],Xtr_fc[3]
xTest_fc_ch0,xTest_fc_ch1,xTest_fc_ch2,xTest_fc_ch3 = Xts_fc[0],Xts_fc[1],Xts_fc[2],Xts_fc[3]
yTrain_fc_ch0,yTrain_fc_ch1,yTrain_fc_ch2,yTrain_fc_ch3 = Ytr_fc[0],Ytr_fc[1],Ytr_fc[2],Ytr_fc[3]
yTest_fc_ch0,yTest_fc_ch1,yTest_fc_ch2,yTest_fc_ch3 = Yts_fc[0],Yts_fc[1],Yts_fc[2],Yts_fc[3]


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
                          optimizer=keras.optimizers.Adam(learning_rate=0.001))
model_fc_ensamble.summary()
# %%

num_epochs = 200
num_batch_size = 64

checkpointer_fc_ensamble = ModelCheckpoint(filepath='d:/Project/Weights/AnsambleFCNN/weights_fc_ansamble.hdf5',
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

xTrain = [xTrain_fc_ch0, xTrain_fc_ch1, xTrain_fc_ch2, xTrain_fc_ch3]
yTrain = [yTrain_fc_ch0, yTrain_fc_ch1, yTrain_fc_ch2, yTrain_fc_ch3]
xTest = [xTest_fc_ch0, xTest_fc_ch1, xTest_fc_ch2, xTest_fc_ch3]
yTest = [yTest_fc_ch0, yTest_fc_ch1, yTest_fc_ch2, yTest_fc_ch3]

print(f'channel')
model_fc_ensamble.load_weights(path_w + '/AnsambleFCNN/' + f'weights_fc_ansamble.hdf5')
score = model_fc_ensamble.evaluate(xTrain, yTrain, verbose=0)
print("Training Accuracy: ", score[1])
score = model_fc_ensamble.evaluate(xTest, yTest, verbose=0)
print("Testing Accuracy: ", score[1])

# Записать xTrain, YTrain,num_labels
params = [num_labels,xTrain,xTest,yTrain,yTest]
write_data = params
datafile=open(path_w + '/AnsambleFCNN/'+'params'+'_FCNN.dat',"wb")
pickle.dump(write_data,datafile)
datafile.close()

exit()


