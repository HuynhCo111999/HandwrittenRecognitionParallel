import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop
row, col, ch = 113, 113, 1
num_classes = 50


def resize_image(image):
    return tf.image.resize(image, [56, 56])


def ModelCNN():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(row, col, ch)))

    # Resise data within the neural network
    # resize images to allow for easy computation
    model.add(Lambda(resize_image))

    # CNN model - Building the model suggested in paper

    model.add(Convolution2D(filters=32, kernel_size=(5, 5),
              strides=(2, 2), padding='same', name='conv1'))  # 96
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool1'))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3),
              strides=(1, 1), padding='same', name='conv2'))  # 256
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool2'))

    model.add(Convolution2D(filters=128, kernel_size=(3, 3),
              strides=(1, 1), padding='same', name='conv3'))  # 256
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='pool3'))
    model.add(Flatten())
    model.add(Dropout(0.5))

    model.add(Dense(512, name='dense1'))  # 1024
    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(256, name='dense2'))  # 1024
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes, name='output'))
    # softmax since output is within 50 classes
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(), metrics=['accuracy'])

    return model
