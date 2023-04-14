import numba as nb
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop
row, col, ch = 113, 113, 1


@nb.njit(fastmath=True)
def resize_image(image):
    return tf.image.resize(image, [56, 56])


def ModelCNN():
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.ZeroPadding2D(
            (1, 1), input_shape=(row, col, ch)))

        # Resize data within the neural network
        # Resize images to allow for easy computation
        model.add(tf.keras.layers.Lambda(resize_image))

        # CNN model - Building the model suggested in paper

        model.add(tf.keras.layers.Convolution2D(filters=32, kernel_size=(5, 5),
                                                strides=(2, 2), padding='same', name='conv1'))  # 96
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), name='pool1'))

        model.add(tf.keras.layers.Convolution2D(filters=64, kernel_size=(3, 3),
                                                strides=(1, 1), padding='same', name='conv2'))  # 256
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), name='pool2'))

        model.add(tf.keras.layers.Convolution2D(filters=128, kernel_size=(3, 3),
                                                strides=(1, 1), padding='same', name='conv3'))  # 256
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), name='pool3'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(512, name='dense1'))  # 1024
        # model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(256, name='dense2'))  # 1024
        model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.Dropout(0.5))

        model.add(tf.keras.layers.Dense(num_classes, name='output'))
        # softmax since output is within 50 classes
        model.add(tf.keras.layers.Activation('softmax'))

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return model
