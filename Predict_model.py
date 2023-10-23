from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from main import *
import pickle
import keras
from sklearn.preprocessing import LabelEncoder
from keras import Model



datafile_rf = open(path_w+'/FCNN/'+'params'+'_'+'FCNN'+'.dat',"rb")
read_data_f= pickle.load(datafile_rf)
num_labels = read_data_f[0]
model_fc_branch = keras.models.load_model(path_w+'/FCNN/'+'weights_fc_ch0.hdf5')
print(model_fc_branch.summary())
class_labels_fc = read_data_f[1]
Xtr_fc = read_data_f[2]
Xts_fc = read_data_f[3]
Ytr_fc = read_data_f[4]
Yts_fc = read_data_f[5]
datafile_rf.close()

datafile_rc = open(path_w+'/CNN/'+'params'+'_'+'CNN'+'.dat',"rb")
read_data_c= pickle.load(datafile_rc)
model_cnn_branch = keras.models.load_model(path_w+'/CNN/'+'weights_cnn_ch0.hdf5')
print(model_cnn_branch.summary())
class_labels_cnn = read_data_c[1]
Xtr_cnn = read_data_c[2]
Xts_cnn = read_data_c[3]
Ytr_cnn = read_data_c[4]
Yts_cnn = read_data_c[5]
datafile_rc.close()

# datafile_rr = open(path_w+'/RNN/'+'params'+'_'+'RNN'+'.dat',"rb")
# read_data_r= pickle.load(datafile_rr)
# model_rnn_branch = keras.models.load_model(path_w+'/RNN/'+'weights_rnn_ch0.hdf5')
# print(model_rnn_branch.summary())
# class_labels_rnn = read_data_r[1]
# Xtr_rnn = read_data_r[2]
# Xts_rnn = read_data_r[3]
# Ytr_rnn = read_data_r[4]
# Yts_rnn = read_data_r[5]
# datafile_rr.close()

xTrain_fc_ch0,xTrain_fc_ch1,xTrain_fc_ch2,xTrain_fc_ch3 = Xtr_fc[0],Xtr_fc[1],Xtr_fc[2],Xtr_fc[3]
xTest_fc_ch0,xTest_fc_ch1,xTest_fc_ch2,xTest_fc_ch3 = Xts_fc[0],Xts_fc[1],Xts_fc[2],Xts_fc[3]
yTrain_fc_ch0,yTrain_fc_ch1,yTrain_fc_ch2,yTrain_fc_ch3 = Ytr_fc[0],Ytr_fc[1],Ytr_fc[2],Ytr_fc[3]
yTest_fc_ch0,yTest_fc_ch1,yTest_fc_ch2,yTest_fc_ch3 = Yts_fc[0],Yts_fc[1],Yts_fc[2],Yts_fc[3]

xTrain_cnn_ch0_cut,xTrain_cnn_ch1_cut,xTrain_cnn_ch2_cut,xTrain_cnn_ch3_cut = Xtr_cnn[0],Xtr_cnn[1],Xtr_cnn[2],Xtr_cnn[3]
xTest_cnn_ch0_cut,xTest_cnn_ch1_cut,xTest_cnn_ch2_cut,xTest_cnn_ch3_cut = Xts_cnn[0],Xts_cnn[1],Xts_cnn[2],Xts_cnn[3]
yTrain_cnn_ch0,yTrain_cnn_ch1,yTrain_cnn_ch2,yTrain_cnn_ch3 = Ytr_cnn[0],Ytr_cnn[1],Ytr_cnn[2],Ytr_cnn[3]
yTest_cnn_ch0,yTest_cnn_ch1,yTest_cnn_ch2,yTest_cnn_ch3 = Yts_cnn[0],Yts_cnn[1],Yts_cnn[2],Yts_cnn[3]

# xTrain_rnn_ch0,xTrain_rnn_ch1,xTrain_rnn_ch2,xTrain_rnn_ch3 = Xtr_rnn[0],Xtr_rnn[1],Xtr_rnn[2],Xtr_rnn[3]
# xTest_rnn_ch0,xTest_rnn_ch1,xTest_rnn_ch2,xTest_rnn_ch3 = Xts_rnn[0],Xts_rnn[1],Xts_rnn[2],Xts_rnn[3]
# yTrain_rnn_ch0,yTrain_rnn_ch1,yTrain_rnn_ch2,yTrain_rnn_ch3 = Ytr_rnn[0],Ytr_cnn[1],Ytr_rnn[2],Ytr_rnn[3]
# yTest_rnn_ch0,yTest_rnn_ch1,yTest_rnn_ch2,yTest_rnn_ch3 = Yts_rnn[0],Yts_rnn[1],Yts_rnn[2],Yts_rnn[3]

def conf_matrix(model, weights, xTest, yTest):
    model.load_weights(weights)
    predictions = model.predict(xTest)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(yTest, axis=1)
    array = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(array)
    return df_cm

le = LabelEncoder()

df_cm = []
df_cm.append(conf_matrix(model_fc_branch, path_w+'/FCNN/'+'weights_fc_ch0.hdf5', xTest_fc_ch0, yTest_fc_ch0))
df_cm.append(conf_matrix(model_fc_branch, path_w+'/FCNN/'+'weights_fc_ch1.hdf5', xTest_fc_ch1, yTest_fc_ch1))
df_cm.append(conf_matrix(model_fc_branch, path_w+'/FCNN/'+'weights_fc_ch2.hdf5', xTest_fc_ch2, yTest_fc_ch2))
df_cm.append(conf_matrix(model_fc_branch, path_w+'/FCNN/'+'weights_fc_ch3.hdf5', xTest_fc_ch3, yTest_fc_ch3))

df_cm.append(conf_matrix(model_cnn_branch, path_w+'/CNN/'+'weights_cnn_ch0.hdf5', xTest_cnn_ch0_cut, yTest_cnn_ch0))
df_cm.append(conf_matrix(model_cnn_branch, path_w+'/CNN/'+'weights_cnn_ch1.hdf5', xTest_cnn_ch1_cut, yTest_cnn_ch1))
df_cm.append(conf_matrix(model_cnn_branch, path_w+'/CNN/'+'weights_cnn_ch2.hdf5', xTest_cnn_ch2_cut, yTest_cnn_ch2))
df_cm.append(conf_matrix(model_cnn_branch, path_w+'/CNN/'+'weights_cnn_ch3.hdf5', xTest_cnn_ch3_cut, yTest_cnn_ch3))

# df_cm.append(conf_matrix(model_rnn_branch, path_w+'/RNN/'+'weights_rnn_ch0.hdf5', xTest_rnn_ch0, yTest_rnn_ch0))
# df_cm.append(conf_matrix(model_rnn_branch, path_w+'/RNN/'+'weights_rnn_ch1.hdf5', xTest_rnn_ch1, yTest_rnn_ch1))
# df_cm.append(conf_matrix(model_rnn_branch, path_w+'/RNN/'+'weights_rnn_ch2.hdf5', xTest_rnn_ch2, yTest_rnn_ch2))
# df_cm.append(conf_matrix(model_rnn_branch, path_w+'/RNN/'+'weights_rnn_ch3.hdf5', xTest_rnn_ch3, yTest_rnn_ch3))

for i in range(len(df_cm)):
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.4)  # for label size
    sns.heatmap(df_cm[i], annot=True, annot_kws={"size": 16})  # font size


# %%
def print_prediction(array):
    predicted_vector = Model.predict_classes(array)
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = Model.predict_proba(array)
    predicted_proba = predicted_proba_vector[0]
    for i in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))
    return category


# %%
def in_top(model, model_type, path, xTest, yTest, class_labels):
    model.load_weights(path)
    yTest = np.argmax(yTest, axis=1)

    in_top1 = 0
    not_in_top1 = 0

    in_top2 = 0
    not_in_top2 = 0

    in_top3 = 0
    not_in_top3 = 0

    in_top4 = 0
    not_in_top4 = 0

    in_top5 = 0
    not_in_top5 = 0

    for i in enumerate(yTest):
        data = xTest[i[0], :]
        if model_type == 'cnn':
            data = np.reshape(data, (1, np.shape(xTest)[1], np.shape(xTest)[2]))
            #data = np.reshape(data, (1, np.shape(xTest)[1], np.shape(xTest)[2], np.shape(xTest)[3]))
            predicted_proba_vector = model.predict(data)
            predicted_proba = predicted_proba_vector[0]
        elif model_type == 'fc':
            data = np.reshape(data, (1, np.shape(xTest)[1]))
            predicted_proba_vector = model.predict(data)
            predicted_proba = predicted_proba_vector[0]
        else:
            data = np.reshape(data, (1, np.shape(xTest)[1], np.shape(xTest)[2]))
            # data = np.reshape(data, (1, np.shape(xTest)[1], np.shape(xTest)[2], np.shape(xTest)[3]))
            predicted_proba_vector = model.predict(data)
            predicted_proba = predicted_proba_vector[0]
        answer = []
        for pp in predicted_proba:
            answer.append(class_labels[np.argmax(predicted_proba)])
            predicted_proba[np.argmax(predicted_proba)] = -np.inf

            # print(answer[0:5])
        # print(category[i[1]])

        if class_labels[i[1]] in answer[0:1]:
            in_top1 += 1
        else:
            not_in_top1 += 1

        if class_labels[i[1]] in answer[0:2]:
            in_top2 += 1
        else:
            not_in_top2 += 1

        if class_labels[i[1]] in answer[0:3]:
            in_top3 += 1
        else:
            not_in_top3 += 1

        if class_labels[i[1]] in answer[0:4]:
            in_top4 += 1
        else:
            not_in_top4 += 1

        if class_labels[i[1]] in answer[0:5]:
            in_top5 += 1
        else:
            not_in_top5 += 1

    print('\n-------TOP1-------------')
    print(
        f'TOP1: {in_top1 / np.shape(yTest)[0]}, TOP2: {in_top2 / np.shape(yTest)[0]},TOP3: {in_top3 / np.shape(yTest)[0]}, TOP4: {in_top4 / np.shape(yTest)[0]}, TOP5: {in_top5 / np.shape(yTest)[0]}'
        )


# %%
print("FCC  TOP5")
in_top(model_fc_branch, 'fc', path_w+'/FCNN/' + f'weights_fc_ch0.hdf5', xTest_fc_ch0, yTest_fc_ch0, class_labels_fc)
in_top(model_fc_branch, 'fc', path_w+'/FCNN/' + f'weights_fc_ch1.hdf5', xTest_fc_ch1, yTest_fc_ch1, class_labels_fc)
in_top(model_fc_branch, 'fc', path_w+'/FCNN/' + f'weights_fc_ch2.hdf5', xTest_fc_ch2, yTest_fc_ch2, class_labels_fc)
in_top(model_fc_branch, 'fc', path_w+'/FCNN/' + f'weights_fc_ch3.hdf5', xTest_fc_ch3, yTest_fc_ch3, class_labels_fc)
# %%
print("CNN  TOP5")
in_top(model_cnn_branch, 'cnn', path_w+'/CNN/' + f'weights_cnn_ch0.hdf5', xTest_cnn_ch0_cut, yTest_cnn_ch0, class_labels_cnn)
in_top(model_cnn_branch, 'cnn', path_w+'/CNN/' + f'weights_cnn_ch1.hdf5', xTest_cnn_ch1_cut, yTest_cnn_ch1, class_labels_cnn)
in_top(model_cnn_branch, 'cnn', path_w+'/CNN/' + f'weights_cnn_ch2.hdf5', xTest_cnn_ch2_cut, yTest_cnn_ch2, class_labels_cnn)
in_top(model_cnn_branch, 'cnn', path_w+'/CNN/' + f'weights_cnn_ch3.hdf5', xTest_cnn_ch3_cut, yTest_cnn_ch3, class_labels_cnn)
# # %%
# print("RNN  TOP5")
# in_top(model_rnn_branch, 'rnn', path_w+'/RNN/' + f'weights_rnn_ch0.hdf5', xTest_rnn_ch0, yTest_rnn_ch0, class_labels_rnn)
# in_top(model_rnn_branch, 'rnn', path_w+'/RNN/' + f'weights_rnn_ch1.hdf5', xTest_rnn_ch1, yTest_rnn_ch1, class_labels_rnn)
# in_top(model_rnn_branch, 'rnn', path_w+'/RNN/' + f'weights_rnn_ch2.hdf5', xTest_rnn_ch2, yTest_rnn_ch2, class_labels_rnn)
# in_top(model_rnn_branch, 'rnn', path_w+'/RNN/' + f'weights_rnn_ch3.hdf5', xTest_rnn_ch3, yTest_rnn_ch3, class_labels_rnn)
