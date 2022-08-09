from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.models import load_model
import tensorflow.keras.datasets as ds
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Loads the data
    # (train_data, train_labels), (test_data, test_labels) = ds.mnist.load_data()
    (train_data, train_labels), (test_data, test_labels) = ds.fashion_mnist.load_data()
    print('data loaded')
    # Plots a single digit from the data
    # summarize loaded dataset
    # plot first few images
    for i in range(8):
        # define subplot
        plt.subplot(330 + 1 + i)
        # plot raw pixel data
        plt.imshow(train_data[i], cmap=plt.get_cmap('gray'))
    # show the figure
    plt.subplot(330 + 1 + 8)
    sns.heatmap(train_data[1, :, :])
    plt.show()
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

    model = tf.keras.Sequential()
    model.add(layers.Dense(30, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dense(30, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    print('sequential dense layers added')

    # For a multi-class classification problem
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
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
    epochs = range(1, len(loss_values) + 1)
    # ’bo’ is for blue dot, ‘b’ is for solid blue line
    plt.plot(epochs, loss_values, 'bo', label='Training loss')
    plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # model.save('my_model.h5')