import sys
import os
import numpy as np
import glob
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import Image
from keras.utils import to_categorical
from sklearn.utils import shuffle
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation, BatchNormalization
from keras.layers.convolutional import Convolution2D, Cropping2D, ZeroPadding2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam, RMSprop

batch_size = 8
num_classes = 50
row, col, ch = 113, 113, 1


def readFileForm():
    d = {}
    with open('dataset/result.txt') as f:
        for line in f:
            key = line.split(' ')[0]
            writer = line.split(' ')[1]
            d[key] = writer
    return d


def getImageFiles():
    tmp = []
    path_to_files = os.path.join('dataset/data_subset', '*')
    for filename in sorted(glob.glob(path_to_files)):
        tmp.append(filename)
    return tmp


def getImageTargets(dic):
    target_list = []
    path_to_files = os.path.join('dataset/data_subset', '*')
    for filename in sorted(glob.glob(path_to_files)):
        image_name = filename.split('/')[-1]
        file, ext = os.path.splitext(image_name)
        parts = file.split('-')
        form = parts[0] + '-' + parts[1]
        for key in dic:
            if key == form:
                target_list.append(str(dic[form]))
    return target_list


def encodeLabel(img_targets):
    encoder = LabelEncoder()
    encoder.fit(img_targets)
    encoded_Y = encoder.transform(img_targets)
    return encoded_Y

# Start with train generator shared in the class and add image augmentations


def generate_data(samples, target_files,  batch_size=batch_size, factor=0.1):
    num_samples = len(samples)
    while 1:  # Loop forever so the generator never terminates
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            batch_targets = target_files[offset:offset+batch_size]

            images = []
            targets = []
            for i in range(len(batch_samples)):
                batch_sample = batch_samples[i]
                batch_target = batch_targets[i]
                im = Image.open(batch_sample)
                cur_width = im.size[0]
                cur_height = im.size[1]

                # print(cur_width, cur_height)
                height_fac = 113 / cur_height

                new_width = int(cur_width * height_fac)
                size = new_width, 113

                # Resize so height = 113 while keeping aspect ratio
                imresize = im.resize((size), Image.ANTIALIAS)
                now_width = imresize.size[0]
                now_height = imresize.size[1]
                # Generate crops of size 113x113 from this resized image and keep random 10% of crops

                # total x start points are from 0 to width -113
                avail_x_points = list(range(0, now_width - 113))

                # Pick random x%
                pick_num = int(len(avail_x_points)*factor)

                # Now pick
                random_startx = sample(avail_x_points,  pick_num)

                for start in random_startx:
                    imcrop = imresize.crop((start, 0, start+113, 113))
                    images.append(np.asarray(imcrop))
                    targets.append(batch_target)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(targets)

            # reshape X_train for feeding in later
            X_train = X_train.reshape(X_train.shape[0], 113, 113, 1)
            # convert to float and normalize
            X_train = X_train.astype('float32')
            X_train /= 255

            # One hot encode y
            y_train = to_categorical(y_train, num_classes)
            yield shuffle(X_train, y_train)


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


def main() -> int:
    dic = readFileForm()
    img_files = np.asarray(getImageFiles())
    img_targets = np.asarray(getImageTargets(dic))

    # Encode to get label
    encoded_Y = encodeLabel(img_targets)

    print(encoded_Y.shape)

    train_files, rem_files, train_targets, rem_targets = train_test_split(
        img_files, encoded_Y, train_size=0.66, random_state=52, shuffle=True)

    validation_files, test_files, validation_targets, test_targets = train_test_split(
        rem_files, rem_targets, train_size=0.5, random_state=22, shuffle=True)

    print(train_files.shape, validation_files.shape, test_files.shape)
    print(train_targets.shape, validation_targets.shape, test_targets.shape)

    train_generator = generate_data(
        train_files, train_targets, batch_size=batch_size, factor=0.3)
    validation_generator = generate_data(
        validation_files, validation_targets, batch_size=batch_size, factor=0.3)
    test_generator = generate_data(
        test_files, test_targets, batch_size=batch_size, factor=0.1)

    modelCNN = ModelCNN()
    modelCNN.summary()

    return 0


if __name__ == '__main__':
    sys.exit(main())
