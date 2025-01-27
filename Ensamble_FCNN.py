from keras import Input
from keras import Model
from keras.layers import concatenate
import numpy as np
import pickle
from main import *
import time
import keras
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from History_Plots import plot_accuracy_and_loss
from keras.utils import plot_model

# Читать из файлов xTrain,xTest,yTrain,yTest,num_labels(если пустые, то запуск FCNN,CNN)
datafile_rf = open(path_w+'/FCNN/'+'params'+'_'+'FCNN'+'.dat', "rb")
read_data_f = pickle.load(datafile_rf)
num_labels = read_data_f[0]
Xtr_fc = read_data_f[2]
Xts_fc = read_data_f[3]
Ytr_fc = read_data_f[4]
Yts_fc = read_data_f[5]
datafile_rf.close()

start = time.perf_counter()
for i in range(num_folders):
    exec(f"xTrain_fc_ch{i} = Xtr_fc[i]")
    exec(f"xTest_fc_ch{i} = Xts_fc[i]")
    exec(f"yTrain_fc_ch{i} = Ytr_fc[i]")
    exec(f"yTest_fc_ch{i} = Yts_fc[i]")

    exec(f"input_ch{i} = Input(shape=(np.shape(xTrain_fc_ch{i})[1],))")

    exec(f"m_ch{i} = Dense(256,\
              input_shape=(np.shape(xTrain_fc_ch{i})[1],),\
              activation='relu',\
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),\
              bias_regularizer=regularizers.l2(1e-4),\
              activity_regularizer=regularizers.l2(1e-4))(input_ch{i})")
    exec(f"m_ch{i} = Dropout(0.5)(m_ch{i})")
    exec(f"m_ch{i} = Dense(256,\
              activation='relu',\
              kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),\
              bias_regularizer=regularizers.l2(1e-4),\
              activity_regularizer=regularizers.l2(1e-4))(m_ch{i})")
    exec(f"m_ch{i} = Dropout(0.5)(m_ch{i})")
    exec(f"m_ch{i} = Model(inputs=input_ch{i}, outputs=m_ch{i})")


# combine the output of the all branches
combined = concatenate([eval(f"m_ch{i}.output") for i in range(num_folders)])

# apply a FC layer and softmax
all_ch = Dense(256,
               activation="relu",
               kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=regularizers.l2(1e-4),
               activity_regularizer=regularizers.l2(1e-4))(combined)

all_ch = Dropout(0.5)(all_ch)
all_ch = Dense(num_labels, activation='softmax')(all_ch)

model_fc_ensamble = Model(inputs=[eval(f"m_ch{i}.input") for i in range(num_folders)], outputs=all_ch)

# Compile the model

model_fc_ensamble.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                          optimizer=keras.optimizers.Adam(learning_rate=0.001))
model_fc_ensamble.summary()

# plot_model(model_fc_ensamble, to_file="d:\\model_fc_ensamble.png", show_shapes=True, show_layer_names=True)
# %%


num_epochs = 200
num_batch_size = 128

checkpointer_fc_ensamble = ModelCheckpoint(filepath=path_w + '/Ansamble_FCNN/' + f'weights_fc_ansamble.hdf5',
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_best_only=True)

history_fc_ensamble = model_fc_ensamble.fit(x=[eval(f"xTrain_fc_ch{i}") for i in range(num_folders)],
                                            y=yTrain_fc_ch0,  # нет ошибки определение переменной в цикле выше
                                            batch_size=num_batch_size,
                                            epochs=num_epochs,
                                            validation_data=(
                                                [eval(f"xTest_fc_ch{i}") for i in range(num_folders)],
                                                yTest_fc_ch0  # нет ошибки определение переменной в цикле выше
                                            ),
                                            callbacks=[checkpointer_fc_ensamble],
                                            verbose=1
                                            )

# %%



xTrain = [eval(f"xTrain_fc_ch{i}") for i in range(num_folders)]
yTrain = [eval(f"yTrain_fc_ch{i}") for i in range(num_folders)]
xTest = [eval(f"xTest_fc_ch{i}") for i in range(num_folders)]
yTest = [eval(f"yTest_fc_ch{i}") for i in range(num_folders)]
history = [history_fc_ensamble]

plot_accuracy_and_loss(history)

print(f'channels')
model_fc_ensamble.load_weights(path_w + '/Ansamble_FCNN/' + f'weights_fc_ansamble.hdf5')
score = model_fc_ensamble.evaluate(xTrain, yTrain, verbose=0)
print("Training Accuracy: ", round(score[1], 3))
score = model_fc_ensamble.evaluate(xTest, yTest, verbose=0)
print("Testing Accuracy: ", round(score[1], 3))

finish = time.perf_counter()
time = round((finish - start), 3)
print("\nВремя обработки = ", time)

# Записать xTrain, YTrain,num_labels
params = [num_labels, xTrain, xTest, yTrain, yTest]
write_data = params
datafile = open(path_w + '/Ansamble_FCNN/'+'params'+'_FCNN.dat', "wb")
pickle.dump(write_data, datafile)
datafile.close()

exit()


