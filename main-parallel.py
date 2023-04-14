from numba import jit
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
from keras.callbacks import ModelCheckpoint
from random import sample

batch_size = 8
num_classes = 50
row, col, ch = 113, 113, 1
nb_epoch = 8
samples_per_epoch = 3268
nb_val_samples = 842


@jit(nopython=True)
def generate_data(samples, target_files, batch_size=batch_size, factor=0.1):
    num_samples = len(samples)
    while 1:
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

                # total x start points are from 0 to width -113
                avail_x_points = list(range(0, now_width - 113))

                # Pick random x%
                pick_num = int(len(avail_x_points)*factor)

                # Now pick
                random_startx = np.random.choice(
                    avail_x_points, pick_num, replace=False)

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
