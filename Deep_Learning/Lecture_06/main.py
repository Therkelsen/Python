from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout
from livelossplot import PlotLossesKeras
import tensorflow.keras.datasets as ds
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np

if __name__ == '__main__':
    # Loads the data
    # (train_data, train_labels), (test_data, test_labels) = ds.mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = ds.fashion_mnist.load_data()
    print('data loaded')
    # Plots a single digit from the data
    # summarize loaded dataset
    # plot first few images
    x = random.randint(0, len(train_data - 6 * 6))
    for i in range(6 * 6):
        # define subplot
        plt.subplot(6, 6, i + 1)
        # plot raw pixel data
        plt.imshow(train_data[x + i], cmap=plt.get_cmap('gray'))
        # plot heatmap of pixel data
        # sns.heatmap(train_data[random.randint(0, len(train_data)), :, :])

    # show the figure
    plt.show(block=False)
    # Reshapes the data to work in a FFN
    # train_data = train_data.reshape((60000, 28 * 28)).astype('float32')
    train_data = train_data.reshape((60000, 28 * 28)).astype('float32')
    train_data /= 255
    # test_data = test_data.reshape((10000, 28 * 28)).astype('float32')
    test_data = test_data.reshape((10000, 28 * 28)).astype('float32')
    test_data /= 255

    print('train_data: X=%s, y=%s' % (train_data.shape, train_labels.shape))
    print('test_data: X=%s, y=%s' % (test_data.shape, test_labels.shape))

    num_classes = 10
    num_epochs = 100
    num_batches = 2 ** 10
    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    print('data reshaped to work in a ffn')
    print('train_data shape:', train_data.shape)
    print('test_data shape:', test_data.shape)

    # regularizers.L2(l2=1e-3)

    inputs = Input(shape=(28*28,))
    x = Dropout(0.1)(inputs)

    x = Dense(2 ** 8, activation='selu',
              kernel_regularizer =regularizers.L1L2(l1=1e-4, l2=1e-3))(x)
    x = Dropout(0.1)(x)

    x = Dense(2 ** 8, activation='selu',
              kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3))(x)
    x = Dropout(0.1)(x)

    x = Dense(2 ** 8, activation='selu',
              kernel_regularizer=regularizers.L1L2(l1=1e-4, l2=1e-3))(x)
    x = Dropout(0.1)(x)

    output = Dense(10, activation='softmax')(x)

    model = Model(inputs, output)
    print('layers added')

    # For a multi-class classification problem
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['Accuracy'])
    print('model compiled')

    history = model.fit(x=train_data, y=train_labels,  # Input and desired outcome
                        batch_size=num_batches,  # Number of samples per gradient update. If none, it defaults to 32
                        epochs=num_epochs,  # Number of runs over the complete x and y
                        verbose=0,  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                        callbacks=[PlotLossesKeras()],  # List of functions to call during training
                        validation_split=0.0,  # Part of dataset set aside for validating
                        validation_data=(test_data, test_labels),  # Validation dataset, tuple (x_val, y_val)
                        shuffle=True,  # shuffle the training data before each epoch
                        class_weight=None,  # Give some classes more/less weight
                        sample_weight=None,  # Give some samples more/less weight
                        )
    print('model fitted')

    # history_dict = history.history
    # loss_values = history_dict['loss']
    # val_loss_values = history_dict['val_loss']
    # acc_values = history_dict['accuracy']
    # val_acc_values = history_dict['val_accuracy']
    # epochs = range(1, len(loss_values) + 1)
    # # 'bo' is for blue dot, 'b' is for solid blue line
    #
    # plt.subplots(1, 2)
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, loss_values, 'r-', label='Training loss', alpha=0.6)
    # plt.plot(epochs, val_loss_values, 'm-', label='Validation loss', alpha=0.6)
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, acc_values, 'g-', label='Training acc', alpha=0.6)
    # plt.plot(epochs, val_acc_values, 'y-', label='Validation acc', alpha=0.6)
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    #
    # plt.show()

    model.summary()

    # model.save('func_dense_model.h5')