from numba import jit, njit
import sys
import os
import numpy as np
import glob
import time
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image

# CONST VARIABLE
batch_size = 8
num_classes = 50
row, col, ch = 113, 113, 1
nb_epoch = 8
samples_per_epoch = 3268
nb_val_samples = 842


@jit
def readFileForm():
    d = {}
    f = open('dataset/result.txt')
    for line in f:
        key = line.split(' ')[0]
        writer = line.split(' ')[1]
        d[key] = writer
    return d


@jit
def getImageFiles():
    tmp = []
    path_to_files = os.path.join('dataset/data_subset', '')
    for filename in os.listdir(path_to_files):
        if filename.endswith('.png'):
            tmp.append(os.path.join(path_to_files, filename))
    return sorted(tmp)


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


@njit
def encodeLabel(img_targets):
    uniq_targets = np.unique(img_targets)
    target_map = {}
    for i in range(len(uniq_targets)):
        target_map[uniq_targets[i]] = i
    encoded_Y = np.zeros_like(img_targets, dtype=np.int32)
    for i in range(len(img_targets)):
        encoded_Y[i] = target_map[img_targets[i]]
    return encoded_Y


@jit(nopython=True)
def to_categorical(y, num_classes):
    return np.eye(num_classes, dtype='uint8')[y]


# Generate data
def load_image(filepath):
    image = np.load(filepath)
    return image


def resize_image(image, new_height):
    # Resize image while maintaining aspect ratio
    cur_width, cur_height = image.size
    height_fac = new_height / cur_height
    new_width = int(cur_width * height_fac)
    size = new_width, new_height
    resized_image = image.resize(size, Image.ANTIALIAS)
    return resized_image


def convert_image_to_array(image):
    # Convert PIL Image to NumPy array
    image_array = np.asarray(image)
    return image_array


def read_image(filepath):
    # Load image, resize it and convert it to NumPy array
    image = load_image(filepath)
    resized_image = resize_image(image, 113)
    image_array = convert_image_to_array(resized_image)
    return image_array


def generate_data(samples, target_files, batch_size, factor=0.1):
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
                im = read_image(batch_sample)
                now_width = im.shape[1]
                now_height = im.shape[0]
                # Generate crops of size 113x113 from this resized image and keep random 10% of crops

                # total x start points are from 0 to width -113
                avail_x_points = list(range(0, now_width - 113))

                # Pick random x%
                pick_num = int(len(avail_x_points)*factor)

                # Now pick
                random_startx = np.random.choice(
                    avail_x_points, pick_num, replace=False)

                for start in random_startx:
                    imcrop = im[:, start:start+113, :]
                    images.append(imcrop)
                    targets.append(batch_target)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(targets)

            # convert to float and normalize
            X_train = X_train.astype('float32')
            X_train /= 255

            # One hot encode y
            y_train = to_categorical(y_train, num_classes)
            np.random.shuffle(X_train)
            np.random.shuffle(y_train)
            yield X_train, y_train


# MAIN

def main() -> int:
    start_time = time.time()
    dic = readFileForm()
    img_files = np.asarray(getImageFiles())
    print(img_files)

    img_targets = np.asarray(getImageTargets(dic))
    print(img_targets)

    encoded_Y = encodeLabel(img_targets)
    print(encoded_Y)

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

    end_time = time.time()
    duration = end_time - start_time
    print("Thời gian chạy function là: ", duration, "giây")


if __name__ == '__main__':
    sys.exit(main())
