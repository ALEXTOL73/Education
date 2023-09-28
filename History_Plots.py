#Графики обучения
from matplotlib import pyplot as plt

def plot_accuracy_and_loss(history):
    # Retrieve a list of list results on training and test data
    # sets for each training epoch
    # -----------------------------------------------------------
    acc_ch = []
    val_acc_ch = []

    loss_ch = []
    val_loss_ch = []

    for hist in history:
        acc_ch.append(hist.history['accuracy'])  # history.history is a dictionary with 'accuracy'
        val_acc_ch.append(hist.history['val_accuracy'])

        loss_ch.append(hist.history['loss'])
        val_loss_ch.append(hist.history['val_loss'])

    epochs = range(len(acc_ch[0]))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    fig = plt.figure(figsize=(12, 8))

    color = ['b--', 'r--', 'g--', 'y--']
    val_color = ['b', 'r', 'g', 'y']

    for i in range(len(acc_ch)):
        plt.plot(epochs, acc_ch[i], color[i], label=f'Training accuracy ch{i}')
        plt.plot(epochs, val_acc_ch[i], val_color[i], label=f'Validation accuracy ch{i}')

    plt.title('Training and validation accuracy')
    plt.legend()
    plt.show()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------

    fig = plt.figure(figsize=(12, 8))

    for i in range(len(loss_ch)):
        plt.plot(epochs, loss_ch[i], color[i], label=f'Training loss ch{i}')
        plt.plot(epochs, val_loss_ch[i], val_color[i], label=f'Validation loss ch{i}')

    plt.title('Training and validation loss')
    plt.legend()
    plt.show()