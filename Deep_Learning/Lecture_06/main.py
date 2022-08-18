##
from keras.utils import to_categorical
from keras import optimizers
from keras import regularizers
from keras.models import load_model
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.datasets import mnist, fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from livelossplot import PlotLossesKeras
from cleanlab.filter import find_label_issues
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import random
import numpy as np

if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    # Loads the data
    # (train_data, train_labels), (test_data, test_labels) = ds.mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()
    print('data loaded')
    # Plots a single digit from the data
    # summarize loaded dataset
    # plot first few images
    x = random.randint(0, len(train_data - 3 * 3))
    for i in range(3 * 3):
        # define subplot
        plt.subplot(3, 3, i + 1)
        # plot raw pixel data
        # plt.imshow(train_data[x + i], cmap=plt.get_cmap('gray'))
        # plot heatmap of pixel data
        sns.heatmap(train_data[x + i, :, :])

    # show the figure
    plt.subplots_adjust(wspace=0.5,
                        hspace=0.5)
    plt.show(block=False)

    num_classes = 10
    num_epochs = 3
    num_batches = 2 ** 10
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

    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory('data/train',
                                                        target_size=(150, 150),
                                                        batch_size=num_batches,
                                                        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory('data/validation',
                                                            target_size=(150, 150),
                                                            batch_size=num_batches,
                                                            class_mode='categorical')

    model = tf.keras.Sequential()
    model.add(Conv2D(2 ** 5, kernel_size=(3, 3), activation='selu', input_shape=input_shape,
                     kernel_regularizer=regularizers.L2(l2=1e-3)))
    #model.add(BatchNormalization())
    #model.add(Dropout(0.2))

    # model.add(Conv2D(2 ** 5, kernel_size=(3, 3), activation='selu',
    #                  kernel_regularizer=regularizers.L2(l2=1e-3)))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))
    #
    # model.add(Conv2D(2 ** 6, kernel_size=(3, 3), activation='selu',
    #                  kernel_regularizer=regularizers.L2(l2=1e-3)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))
    #
    # model.add(Conv2D(2 ** 7, kernel_size=(3, 3), activation='selu',
    #                  kernel_regularizer=regularizers.L2(l2=1e-3)))
    # model.add(BatchNormalization())
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.3))

    model.add(Flatten())

    # model.add(Dense(2 ** 9, activation='selu',
    #                 kernel_regularizer=regularizers.L2(l2=1e-3)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    # model.add(Dense(2 ** 7, activation='selu',
    #                 kernel_regularizer=regularizers.L2(l2=1e-3)))
    # model.add(BatchNormalization())
    # model.add(Dropout(0.3))

    model.add(Dense(10, activation='softmax'))
    print('layers added')

    # For a multi-class classification problem
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['Accuracy'])
    print('model compiled')
    history = model.fit(x=train_data, y=train_labels,  # Input and desired outcome
                        batch_size=num_batches,  # Number of samples per gradient update. If none, it defaults to 32
                        #steps_per_epoch=2000,
                        epochs=num_epochs,  # Number of runs over the complete x and y
                        verbose=2,  # Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
                        callbacks=[PlotLossesKeras()],  # List of functions to call during training
                        validation_split=0.0,  # Part of dataset set aside for validating
                        validation_data=(test_data, test_labels),  # Validation dataset, tuple (x_val, y_val)
                        #validation_steps=800,
                        shuffle=True,  # shuffle the training data before each epoch
                        class_weight=None,  # Give some classes more/less weight
                        sample_weight=None,  # Give some samples more/less weight
                        )
    print('model fitted')
    ##
    # history_dict = history.history
    # loss_values = history_dict['loss']
    # val_loss_values = history_dict['val_loss']
    # acc_values = history_dict['Accuracy']
    # val_acc_values = history_dict['val_Accuracy']
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
    # plt.subplots_adjust(wspace=0.5,
    #                     hspace=0.5)
    #
    # plt.show(block=False)

    model.summary()

    # model.save('func_dense_model.h5')
    predictions = model.predict(test_data)
    print('predictions shape: ', predictions.shape)
    print('test_labels shape: ', test_labels.shape)

    print('predictions[0]: ', predictions[0].max())
    print('test_labels[0]: ', test_labels[0].max())

    predictions = model.predict(test_data)
    predictions = predictions.argmax(axis=-1)
    wrong_data = test_data[predictions != test_labels]
    wrong_preds = predictions[predictions != test_labels]

    print(wrong_data)
    print(wrong_preds)
