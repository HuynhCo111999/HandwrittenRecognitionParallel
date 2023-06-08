import numpy as np
import numba
from numba import cuda


@cuda.jit()
def cal_cov_gradient(inputs, weights, gradient_loss_to_this_outputs, g_weights, g_biases, g_inputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    f_num = cuda.blockIdx.x

    for n in range(gradient_loss_to_this_outputs.shape[0]):
        for i in range(weights.shape[1]):
            for j in range(weights.shape[2]):
                for k in range(weights.shape[3]):
                    tmp1 = gradient_loss_to_this_outputs[n,
                                                         tx, ty, f_num] * weights[f_num, i, j, k]
                    tmp2 = gradient_loss_to_this_outputs[n,
                                                         tx, ty, f_num] * inputs[n, tx+i, ty+j, k]
                    g_inputs[n, tx+i, ty+j, k] += tmp1
                    g_weights[f_num, i, j, k] += tmp2
        g_biases[f_num] += gradient_loss_to_this_outputs[n, tx, ty, f_num]


@cuda.jit()
def cov(inputs, weights, biases, outputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    f_num = cuda.blockIdx.x

    for n in range(inputs.shape[0]):
        for i in range(weights.shape[1]):
            for j in range(weights.shape[2]):
                for k in range(weights.shape[3]):
                    outputs[n, tx, ty, f_num] += (
                        inputs[n, tx + i, ty + j, k] * weights[f_num, i, j, k])
        outputs[n, tx, ty, f_num] += biases[f_num]


class Conv2D(object):
    def __init__(self, in_channels, kernel_size, features):
        self.features = features
        self.ksize = kernel_size
        weights_scale = np.sqrt(kernel_size * kernel_size * in_channels / 2)
        self.weights = np.random.standard_normal(
            (features, kernel_size, kernel_size, in_channels)) / weights_scale
        self.biases = np.random.standard_normal(features) / weights_scale

        self.g_weights = None
        self.g_biases = None
        self.g_inputs = None
        self.inputs = None
        self.outputs = None

    def forward(self, inputs):
        self.inputs = np.zeros(shape=(inputs.shape[0], inputs.shape[1]+(self.ksize // 2)*2,
                                      inputs.shape[2] + (self.ksize // 2)*2, inputs.shape[3]), dtype=np.float32)
        self.inputs[:, self.ksize // 2: inputs.shape[1] + self.ksize // 2,
                    self.ksize // 2: inputs.shape[2] + self.ksize // 2, :] = inputs.copy()
        self.outputs = np.zeros(shape=(
            inputs.shape[0], inputs.shape[1], inputs.shape[2], self.features), dtype=np.float32)
        grid = (self.features)
        block = (inputs.shape[1], inputs.shape[2])
        cov[grid, block](self.inputs, self.weights, self.biases, self.outputs)
        return self.outputs

    def backward(self, gradient_loss_to_this_outputs):
        self.g_inputs = np.zeros(shape=self.inputs.shape, dtype=np.float32)
        self.g_weights = np.zeros(self.weights.shape, dtype=np.float32)
        self.g_biases = np.zeros(self.biases.shape, dtype=np.float32)
        grid = (self.features)
        block = (self.inputs.shape[1], self.inputs.shape[2])
        cal_cov_gradient[grid, block](self.inputs, self.weights, gradient_loss_to_this_outputs,
                                      self.g_weights, self.g_biases, self.g_inputs)

        self.g_inputs = self.g_inputs[:, self.ksize//2: self.g_inputs.shape[1] - self.ksize//2,
                                      self.ksize//2: self.g_inputs.shape[2] - self.ksize//2, :]
        return self.g_inputs

    def update_parameters(self, lr):
        self.weights -= self.g_weights * lr / self.inputs.shape[0]
        self.biases -= self.g_biases * lr / self.inputs.shape[0]
