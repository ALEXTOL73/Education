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
from keras.utils import plot_model
import warnings

warnings.filterwarnings('ignore')

datafile_rf = open(path_w+'/FCNN/'+'params'+'_'+'FCNN'+'.dat', "rb")
read_data_f = pickle.load(datafile_rf)
num_labels = read_data_f[0]
model_fc_branch = keras.models.load_model(path_w+'/FCNN/'+'weights_fc_ch0.hdf5')
print('\n', model_fc_branch.summary())
class_labels_fc = read_data_f[1]
Xtr_fc = read_data_f[2]
Xts_fc = read_data_f[3]
Ytr_fc = read_data_f[4]
Yts_fc = read_data_f[5]
datafile_rf.close()

datafile_rc = open(path_w+'/CNN/'+'params'+'_'+'CNN'+'.dat', "rb")
read_data_c = pickle.load(datafile_rc)
model_cnn_branch = keras.models.load_model(path_w+'/CNN/'+'weights_cnn_ch0.hdf5')
print('\n', model_cnn_branch.summary())
class_labels_cnn = read_data_c[1]
Xtr_cnn = read_data_c[2]
Xts_cnn = read_data_c[3]
Ytr_cnn = read_data_c[4]
Yts_cnn = read_data_c[5]
datafile_rc.close()

datafile_rr = open(path_w+'/RNN/'+'params'+'_'+'RNN'+'.dat', "rb")
read_data_r = pickle.load(datafile_rr)
model_rnn_branch = keras.models.load_model(path_w+'/RNN/'+'weights_rnn_ch0.hdf5')
print('\n', model_rnn_branch.summary())
class_labels_rnn = read_data_r[1]
Xtr_rnn = read_data_r[2]
Xts_rnn = read_data_r[3]
Ytr_rnn = read_data_r[4]
Yts_rnn = read_data_r[5]
datafile_rr.close()


plot_model(model_fc_branch, to_file="model_fc.png", show_shapes=True, show_layer_names=True)
plot_model(model_cnn_branch, to_file="model_cnn.png", show_shapes=True, show_layer_names=True)
plot_model(model_rnn_branch, to_file="model_rnn.png", show_shapes=True, show_layer_names=True)

for i in range(num_folders):
    exec(f"xTrain_fc_ch{i} = Xtr_fc[i]")
    exec(f"xTest_fc_ch{i} = Xts_fc[i]")
    exec(f"yTrain_fc_ch{i} = Ytr_fc[i]")
    exec(f"yTest_fc_ch{i} = Yts_fc[i]")

    exec(f"xTrain_cnn_ch{i}_cut = Xtr_cnn[i]")
    exec(f"xTest_cnn_ch{i}_cut = Xts_cnn[i]")
    exec(f"yTrain_cnn_ch{i} = Ytr_cnn[i]")
    exec(f"yTest_cnn_ch{i} = Yts_cnn[i]")

    exec(f"xTrain_rnn_ch{i}_cut = Xtr_rnn[i]")
    exec(f"xTest_rnn_ch{i}_cut = Xts_rnn[i]")
    exec(f"yTrain_rnn_ch{i} = Ytr_rnn[i]")
    exec(f"yTest_rnn_ch{i} = Yts_rnn[i]")


def conf_matrix(model, weights, xTest, yTest):
    model.load_weights(weights)
    predictions = model.predict(xTest, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(yTest, axis=1)
    array = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(array)
    return df_cm


le = LabelEncoder()

df_cm = []

for i in range(num_folders):
    df_cm.append(eval(f"conf_matrix(model_fc_branch, path_w + '/FCNN/' + 'weights_fc_ch{i}.hdf5', \
    xTest_fc_ch{i}, yTest_fc_ch{i})"))
    df_cm.append(eval(f"conf_matrix(model_cnn_branch, path_w + '/CNN/' + 'weights_cnn_ch{i}.hdf5', \
    xTest_cnn_ch{i}_cut, yTest_cnn_ch{i})"))
    df_cm.append(eval(f"conf_matrix(model_rnn_branch, path_w + '/RNN/' + 'weights_rnn_ch{i}.hdf5', \
    xTest_rnn_ch{i}_cut, yTest_rnn_ch{i})"))


# for i in range(len(df_cm)):
#     plt.figure(figsize=(10, 7))
#     sns.set(font_scale=1.4)  # for label size
#     sns.heatmap(df_cm[i], annot=True, annot_kws={"size": 16})  # font size


# %%
def print_prediction(array):
    predicted_vector = Model.predict_classes(array)
    predicted_class = le.inverse_transform(predicted_vector)
    print("The predicted class is:", predicted_class[0], '\n')

    predicted_proba_vector = Model.predict_proba(array)
    predicted_proba = predicted_proba_vector[0]
    for _ in range(len(predicted_proba)):
        category = le.inverse_transform(np.array([i]))
        print(category[0], "\t\t : ", format(predicted_proba[i], '.32f'))
    return category


# %%
def in_top(model, model_type, path, xTest, yTest, class_labels, j):
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
            # data = np.reshape(data, (1, np.shape(xTest)[1], np.shape(xTest)[2], np.shape(xTest)[3]))
            predicted_proba_vector = model.predict(data, verbose=0)
            predicted_proba = predicted_proba_vector[0]
        elif model_type == 'fc':
            data = np.reshape(data, (1, np.shape(xTest)[1]))
            predicted_proba_vector = model.predict(data, verbose=0)
            predicted_proba = predicted_proba_vector[0]
        else:
            # data = np.reshape(data, (1, np.shape(xTest)[1]))
            data = np.reshape(data, (1, np.shape(xTest)[1], np.shape(xTest)[2], np.shape(xTest)[3]))
            predicted_proba_vector = model.predict(data, verbose=0)
            predicted_proba = predicted_proba_vector[0]
        answer = []
        for _ in predicted_proba:
            answer.append(class_labels[np.argmax(predicted_proba)])
            predicted_proba[np.argmax(predicted_proba)] = -np.inf

        # print(answer[0:5])
        # print(class_labels[i[1]])

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

    CHANNEL = 'CH' + str(j)
    print('\n' + 40*'-' + CHANNEL + 40*'-')
    print(
      f'TOP1: {in_top1 / np.shape(yTest)[0]:.3f}, TOP2: {in_top2 / np.shape(yTest)[0]:.3f}, TOP3: {in_top3 / np.shape(yTest)[0]:.3f}, \
      TOP4: {in_top4 / np.shape(yTest)[0]:.3f}, TOP5: {in_top5 / np.shape(yTest)[0]:.3f}')

# %%
print("\nFCNN  TOP5")
for i in range(num_folders):
    exec(f"in_top(model_fc_branch, 'fc', path_w+'/FCNN/' + f'weights_fc_ch{i}.hdf5', \
    xTest_fc_ch{i}, yTest_fc_ch{i}, class_labels_fc, i)")

# %%
print("\n\nCNN  TOP5")
for i in range(num_folders):
    exec(f"in_top(model_cnn_branch, 'cnn', path_w+'/CNN/' + f'weights_cnn_ch{i}.hdf5',\
     xTest_cnn_ch{i}_cut, yTest_cnn_ch{i}, class_labels_cnn, i)")

# # %%
print("\n\nRNN  TOP5")
for i in range(num_folders):
    exec(f"in_top(model_rnn_branch, 'rnn', path_w+'/RNN/' + f'weights_rnn_ch{i}.hdf5', \
    xTest_rnn_ch{i}_cut, yTest_rnn_ch{i}, class_labels_rnn, i)")


# Обычное усреднение

def average_ensamble(model, weights, xTest):
    predictions = 0
    for _ in range(len(weights)):
        model.load_weights(weights[i])
        predictions += (1 / num_folders) * model.predict(xTest[i], verbose=0)

    return predictions


xTest_fc = xTest_cnn = xTest_cnn = weights_fc = weights_cnn = weights_rnn = []

weights_fc = [eval(f"path_w + '/FCNN/weights_fc_ch{i}.hdf5'") for i in range(num_folders)]
xTest_fc = [eval(f"xTest_fc_ch{i}") for i in range(num_folders)]
weights_cnn = [eval(f"path_w + '/CNN/weights_cnn_ch{i}.hdf5'") for i in range(num_folders)]
xTest_cnn = [eval(f"xTest_cnn_ch{i}_cut") for i in range(num_folders)]
weights_rnn = [eval(f"path_w + '/RNN/weights_rnn_ch{i}.hdf5'") for i in range(num_folders)]
xTest_rnn = [eval(f"xTest_rnn_ch{i}_cut") for i in range(num_folders)]


predictions_average_ensamble_fc = average_ensamble(model_fc_branch, weights_fc, xTest_fc)
predictions_average_ensamble_cnn = average_ensamble(model_cnn_branch, weights_cnn, xTest_cnn)
predictions_average_ensamble_rnn = average_ensamble(model_rnn_branch, weights_rnn, xTest_rnn)


# %%
def ensamble_top1_acc(predictions, ground_truth):
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(ground_truth, axis=1)
    top1_accuracy = (y_pred == y_true).sum() / np.shape(y_true)[0]
    return top1_accuracy


accuracy_fc = ensamble_top1_acc(predictions_average_ensamble_fc, yTest_fc_ch0)  # нет ошибки определение переменной в цикле выше
accuracy_cnn = ensamble_top1_acc(predictions_average_ensamble_cnn, yTest_cnn_ch0)  # нет ошибки определение переменной в цикле выше
accuracy_rnn = ensamble_top1_acc(predictions_average_ensamble_rnn, yTest_rnn_ch0)  # нет ошибки определение переменной в цикле выше

print('\n\nAVERAGE MODELS:')


# %%
def in_top_ansamble(predictions_all_ch, model, yTest, class_labels):
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

        answer = []
        for pp in predictions_all_ch[i[0]]:
            answer.append(class_labels[np.argmax(predictions_all_ch[i[0]])])
            predictions_all_ch[i[0]][np.argmax(predictions_all_ch[i[0]])] = -np.inf

        # print(answer[0:5])
        # print(class_labels[i[1]])

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
    print('\n' + model)
    print('\n' + 30*'-' + 'TOP1' + 30*'-')
    print(
        f'TOP1: {in_top1 / np.shape(yTest)[0]:.3f}, TOP2: {in_top2 / np.shape(yTest)[0]:.3f},'
        f'TOP3: {in_top3 / np.shape(yTest)[0]:.3f}, TOP4: {in_top4 / np.shape(yTest)[0]:.3f},'
        f' TOP5: {in_top5 / np.shape(yTest)[0]:.3f}'
        )

# %%
in_top_ansamble(predictions_average_ensamble_fc, 'FCNN', yTest_fc_ch0, class_labels_fc)  # нет ошибки определение переменной в цикле выше
in_top_ansamble(predictions_average_ensamble_cnn, 'CNN', yTest_cnn_ch0, class_labels_cnn)  # нет ошибки определение переменной в цикле выше
in_top_ansamble(predictions_average_ensamble_rnn, 'RNN', yTest_rnn_ch0, class_labels_rnn)  # нет ошибки определение переменной в цикле выше

print(f' \nFCNN TOP1: {accuracy_fc:.3f} \nCNN TOP1: {accuracy_cnn:.3f} \nRNN TOP1: {accuracy_rnn:.3f}')