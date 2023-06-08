import numpy as np
import numba
from numba import cuda


@cuda.jit()
def pool(inputs, outputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    d = cuda.blockIdx.x

    for i in range(inputs.shape[0]):
        outputs[i, tx, ty, d] = max(inputs[i, 2 * tx, 2 * ty, d], inputs[i, 2 * tx + 1, 2 * ty, d],
                                    inputs[i, 2 * tx, 2 * ty + 1, d], inputs[i, 2 * tx + 1, 2 * ty + 1, d])


@cuda.jit()
def cal_pool_gradient(inputs, outputs, gradient_to_outputs, grident_to_inputs):
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    d = cuda.blockIdx.x

    for k in range(outputs.shape[0]):
        for i in range(2):
            for j in range(2):
                if outputs[k, tx, ty, d] == inputs[k, 2 * tx + i, 2 * ty + j, d]:
                    grident_to_inputs[k, 2 * tx + i, 2 * ty +
                                      j, d] = gradient_to_outputs[k, tx, ty, d]


class MaxPooling2x2(object):
    def __init__(self):
        self.inputs = None
        self.outputs = None
        self.g_inputs = None

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.outputs = np.zeros(shape=(
            inputs.shape[0], inputs.shape[1]//2, inputs.shape[2]//2, inputs.shape[3]), dtype=np.float32)
        grid = (inputs.shape[3])
        block = (self.outputs.shape[1], self.outputs.shape[2])
        pool[grid, block](self.inputs, self.outputs)
        return self.outputs

    def backward(self, gradient_loss_this_outputs):
        self.g_inputs = np.zeros(shape=self.inputs.shape, dtype=np.float32)
        grid = (self.outputs.shape[3])
        block = (self.outputs.shape[1], self.outputs.shape[2])
        cal_pool_gradient[grid, block](
            self.inputs, self.outputs, gradient_loss_this_outputs, self.g_inputs)
        return self.g_inputs
