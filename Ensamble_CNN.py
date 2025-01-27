
import numpy as np
import pickle
from main import *
import time
import keras
from keras.layers import Dense, Dropout, Flatten
from keras import regularizers
from keras import layers
from keras.callbacks import ModelCheckpoint
from History_Plots import plot_accuracy_and_loss
from keras import Input
from keras import Model
from keras.layers import concatenate
from keras.utils import plot_model

datafile_rc = open(path_w+'/CNN/'+'params'+'_'+'CNN'+'.dat', "rb")
read_data_c= pickle.load(datafile_rc)
num_labels = read_data_c[0]
Xtr_cnn = read_data_c[2]
Xts_cnn = read_data_c[3]
Ytr_cnn = read_data_c[4]
Yts_cnn = read_data_c[5]
datafile_rc.close()

start = time.perf_counter()

for i in range(num_folders):
    exec(f"xTrain_cnn_ch{i}_cut = Xtr_cnn[i]")
    exec(f"xTest_cnn_ch{i}_cut = Xts_cnn[i]")
    exec(f"yTrain_cnn_ch{i} = Ytr_cnn[i]")
    exec(f"yTest_cnn_ch{i} = Yts_cnn[i]")


# %%
def StackConv(x, filt_num=16, filt_size=3, dropout_rate=0.5):
    x = layers.Conv1D(filt_num, filt_size, activation="relu",
                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-5),
                      bias_regularizer=regularizers.l2(1e-5),
                      activity_regularizer=regularizers.l2(1e-5)
                      )(x)

    x = layers.MaxPooling1D()(x)
    x = Dropout(dropout_rate)(x)
    return x

# %%

num_rows = np.shape(xTest_cnn_ch0_cut[0])[0]  # нет ошибки определение переменной в цикле выше
num_columns = np.shape(xTest_cnn_ch0_cut[0])[1]  # нет ошибки определение переменной в цикле выше
num_channels = 1

for i in range(num_folders):
    # exec(f"input_ch{i} = Input(shape=(num_rows, num_columns, num_channels))")
    exec(f"input_ch{i} = Input(shape=(num_rows, num_columns))")
    exec(f"x_ch{i} = StackConv(input_ch{i}, 32, 5, 0.3)")
    exec(f"x_ch{i} = StackConv(input_ch{i}, 64, 5, 0.5)")
    exec(f"x_ch{i} = StackConv(x_ch{i}, 128, 3, 0.5)")


# combine the output of the all branches
combined = layers.Average()([eval(f"x_ch{i}") for i in range(num_folders)])

combined = Flatten()(combined)
# apply a FC layer and softmax
all_ch = Dense(256,
               activation="relu",
               kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
               bias_regularizer=regularizers.l2(1e-4),
               activity_regularizer=regularizers.l2(1e-4))(combined)

all_ch = Dropout(0.4)(all_ch)
all_ch = Dense(num_labels, activation='softmax')(all_ch)

model_cnn_ensamble = Model(inputs=[eval(f"input_ch{i}") for i in range(num_folders)], outputs=all_ch)

# Compile the model
model_cnn_ensamble.compile(loss='categorical_crossentropy', metrics=['accuracy'],
                           optimizer=keras.optimizers.Adam(learning_rate=0.001))
model_cnn_ensamble.summary()
plot_model(model_cnn_ensamble, to_file="d:\\model_cnn_ensamble.png", show_shapes=True, show_layer_names=True)

# %%

num_epochs = 500
num_batch_size = 128

checkpointer_cnn_ensamble = ModelCheckpoint(filepath=path_w + '/Ansamble_CNN/weights_cnn_ansamble.hdf5',
                                            monitor='val_accuracy',
                                            verbose=1,
                                            save_best_only=True)

history_cnn_ensamble = model_cnn_ensamble.fit(
    x=[eval(f"xTrain_cnn_ch{i}_cut") for i in range(num_folders)],
    y=yTrain_cnn_ch0,  # нет ошибки определение переменной в цикле выше
    batch_size=num_batch_size,
    epochs=num_epochs,
    validation_data=(
        [eval(f"xTest_cnn_ch{i}_cut") for i in range(num_folders)],
        yTest_cnn_ch0  # нет ошибки определение переменной в цикле выше
    ),
    callbacks=[checkpointer_cnn_ensamble],
    verbose=1
    )



xTrain = [eval(f"xTrain_cnn_ch{i}_cut") for i in range(num_folders)]
yTrain = [eval(f"yTrain_cnn_ch{i}") for i in range(num_folders)]
xTest = [eval(f"xTest_cnn_ch{i}_cut") for i in range(num_folders)]
yTest = [eval(f"yTest_cnn_ch{i}") for i in range(num_folders)]
history = [history_cnn_ensamble]

plot_accuracy_and_loss(history)

print(f'channels')
model_cnn_ensamble.load_weights(path_w + '/Ansamble_CNN/' + f'weights_cnn_ansamble.hdf5')
score = model_cnn_ensamble.evaluate(xTrain, yTrain, verbose=0)
print("Training Accuracy: ", round(score[1], 3))
score = model_cnn_ensamble.evaluate(xTest, yTest, verbose=0)
print("Testing Accuracy: ", round(score[1], 3))

finish = time.perf_counter()
time = round((finish - start), 3)
print("\nВремя обработки = ", time)

# Записать xTrain, YTrain,num_labels
params = [num_labels, xTrain, xTest, yTrain, yTest]
write_data = params
datafile=open(path_w + '/Ansamble_CNN/'+'params'+'_CNN.dat', "wb")
pickle.dump(write_data, datafile)
datafile.close()

exit()
