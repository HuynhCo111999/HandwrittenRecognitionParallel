from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.utils import shuffle
import numpy as np


class DataSet(object):
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.train_data_size = None
        self.test_data_size = None
        self._index_in_train_epoch = 0

        self.mnist_dataset_construct()

    def mnist_dataset_construct(self):
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

        assert train_images.shape[0] == train_labels.shape[0]
        assert test_images.shape[0] == test_labels.shape[0]

        self.train_data_size = train_images.shape[0]
        self.test_data_size = test_images.shape[0]

        train_images = train_images.reshape(
            train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(
            test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        train_images = train_images.astype(np.float32)
        train_images = np.multiply(train_images, 1.0 / 255.0)
        test_images = test_images.astype(np.float32)
        test_images = np.multiply(test_images, 1.0 / 255.0)

        order = np.arange(self.train_data_size)
        np.random.shuffle(order)
        self.train_data = train_images[order]
        self.train_label = train_labels[order]

        self.test_data = test_images
        self.test_label = test_labels

        self.train_data = self.train_data.reshape(-1, 28, 28, 1)
        self.test_data = self.test_data.reshape(-1, 28, 28, 1)

    def next_batch(self, batch_size):
        start = self._index_in_train_epoch
        self._index_in_train_epoch += batch_size
        if self._index_in_train_epoch > self.train_data_size:
            order = np.arange(self.train_data_size)
            np.random.shuffle(order)
            self.train_data = self.train_data[order]
            self.train_label = self.train_label[order]

            start = 0
            self._index_in_train_epoch = batch_size
            assert batch_size <= self.train_data_size
        end = self._index_in_train_epoch
        return self.train_data[start:end], self.train_label[start:end]
