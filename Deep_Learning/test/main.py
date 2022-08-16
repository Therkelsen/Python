from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
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
    x = random.randint(0, len(train_data - 4 * 4))
    for i in range(4 * 4):
        # define subplot
        plt.subplot(4, 4, i + 1)
        # plot raw pixel data
        # plt.imshow(train_data[x + i], cmap=plt.get_cmap('gray'))
        # plot heatmap of pixel data
        sns.heatmap(train_data[x + i, :, :])

    # show the figure
    plt.subplots_adjust(wspace=0.5,
                        hspace=0.5)
    plt.show(block=False)

    num_classes = 10
    num_epochs = 10
    num_batches = 2 ** 7
    input_shape = (28, 28, 1)

    # Scale images to the [0, 1] range
    train_data = train_data.astype("float32") / 255
    test_data = test_data.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    train_data = np.expand_dims(train_data, -1)
    test_data = np.expand_dims(test_data, -1)
    print("train_data shape:", train_data.shape)
    print(train_data.shape[0], "train samples")
    print(test_data.shape[0], "test samples")

    train_labels = to_categorical(train_labels, num_classes)
    test_labels = to_categorical(test_labels, num_classes)

    print('data reshaped to work in a nn')
    print('train_data shape:', train_data.shape)
    print('test_data shape:', test_data.shape)

    model = tf.keras.Sequential()
    model.add(layers.Input(shape=input_shape))

    model.add(layers.Conv2D(2 ** 5, kernel_size=(3, 3), activation='selu',
                            kernel_regularizer=regularizers.L2(l2=1e-3)))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(2 ** 4, kernel_size=(3, 3), activation='selu',
                            kernel_regularizer=regularizers.L2(l2=1e-3)))

    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())

    model.add(layers.Dense(num_classes, activation='softmax'))

    print('layers added')

    # For a multi-class classification problem
    model.compile(optimizer=optimizers.Adam(),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['Accuracy'])
    print('model compiled')

    history = model.fit(x=train_data, y=train_labels,  # Input and desired outcome
                        batch_size=num_batches,  # Number of samples per gradient update. If none, it defaults to 32
                        epochs=num_epochs,  # Number of runs over the complete x and y
                        verbose=1,  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                        callbacks=None,  # List of functions to call during training
                        validation_split=0.0,  # Part of dataset set aside for validating
                        validation_data=(test_data, test_labels),  # Validation dataset, tuple (x_val, y_val)
                        shuffle=True,  # shuffle the training data before each epoch
                        class_weight=None,  # Give some classes more/less weight
                        sample_weight=None,  # Give some samples more/less weight
                        )
    print('model fitted')

    history_dict = history.history
    loss_values = history_dict['loss']
    val_loss_values = history_dict['val_loss']
    acc_values = history_dict['accuracy']
    val_acc_values = history_dict['val_accuracy']
    epochs = range(1, len(loss_values) + 1)
    # 'bo' is for blue dot, 'b' is for solid blue line

    plt.subplots(1, 2)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss_values, 'r-', label='Training loss', alpha=0.6)
    plt.plot(epochs, val_loss_values, 'm-', label='Validation loss', alpha=0.6)
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, acc_values, 'g-', label='Training acc', alpha=0.6)
    plt.plot(epochs, val_acc_values, 'y-', label='Validation acc', alpha=0.6)
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

    model.summary()

    # model.save('func_dense_model.h5')
