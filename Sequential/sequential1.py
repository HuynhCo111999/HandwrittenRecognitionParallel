import numpy as np
import time
from keras.datasets import mnist
from keras.utils import np_utils
from convolutional_lib import *
from convolutional_nolib import *
from sigmoid import Sigmoid
from maxpool2d import MaxPoolingLayer
from reshape import Reshape
from fclayer import FCLayer
from softmax import Softmax
from network import train, predict
from loss import *

# extract 100 img of each class for testing.


def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]
    all_indices = np.hstack((zero_index, one_index, two_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    y = np_utils.to_categorical(y)
    y = y.reshape(len(y), 3, 1)
    return x, y


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 100)
x_test, y_test = preprocess_data(x_test, y_test, 100)

# CNN
network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    MaxPoolingLayer((5, 26, 26), 2, 2),
    Convolutional((5, 13, 13), 3, 5),
    Sigmoid(),
    Reshape((5, 11, 11), (5 * 11 * 11, 1)),
    FCLayer(5 * 11 * 11, 100),
    Sigmoid(),
    FCLayer(100, 3),
    Softmax()
]

start = time.time()
# train with CNN use scipy -> signal.correlate2D
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)
end = time.time()
print(f'Processing time: {end - start} s')

# test with network
true_label = 0
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    if np.argmax(output) == np.argmax(y):
        true_label += 1

# Score:
print(
    f"Accuracy: {true_label * 100/len(x_test)}% on predict true: {true_label} vs true: {len(x_test)}")
