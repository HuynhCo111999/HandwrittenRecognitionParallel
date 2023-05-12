from layer import *
import numpy as np
from scipy import signal

def correlate(input, kernel, stride=1, padding='valid'):

  h_i, w_i  = input.shape
  h_k, w_k = kernel.shape

  s_h = stride
  s_w = stride

  if padding == 'valid':
    p_h = 0
    p_w = 0

  if padding == 'full':
    p_h= h_k - 1
    p_w= w_k -1

  input = np.pad(input, [(p_h, p_h), (p_w, p_w)],
                    mode='constant', constant_values=0)

  h_out = int((h_i - h_k + 2*p_h)/stride + 1)
  w_out = int((w_i - w_k + 2*p_w)/stride + 1)

  output_conv = np.zeros((h_out, w_out))

  for i in range(h_out):
        for j in range(w_out):
                output_conv[i, j] = np.sum(np.multiply(
                        input[
                            i*stride:h_k+i*stride,
                            j*stride:w_k+j*stride],
                        kernel), axis=(0,1))
  return output_conv

# def correlate_batch(input, kernel, stride=1, padding='valid'):

#   m_i,h_i, w_i  = input.shape
#   h_k, w_k = kernel.shape

#   s_h = stride
#   s_w = stride

#   if padding == 'valid':
#     p_h = 0
#     p_w = 0

#   if padding == 'full':
#     p_h= h_k - 1
#     p_w= w_k -1

#   input = np.pad(input, [(p_h, p_h), (p_w, p_w)],
#                     mode='constant', constant_values=0)

#   h_out = int((h_i - h_k + 2*p_h)/stride + 1)
#   w_out = int((w_i - w_k + 2*p_w)/stride + 1)

#   output_conv = np.zeros((h_out, w_out))

#   for i in range(h_out):
#         for j in range(w_out):
#                 output_conv[i, j] = np.sum(np.multiply(
#                         input[
#                             i*stride:h_k + i*stride,
#                             j*stride:w_k + j*stride],
#                         kernel), axis=(0,1))
#   return output_conv

#CNN su dung thu vien scipy -> signal.correlate2d
class Convolutional(Layer): 
    def __init__(self, input_shape, kernel_size, depth):
        '''
        input_shape: A tupple contain following:
        + input_depth: number of color channels
        + input_height: the height of input
        + input_width: the width of input

        kernel_size: size of kernel matrix: ex: 3 -> 3x3, 5 -> 5x5,...
        depth: number of different kernel, also the depth of output.

        '''
        
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        np.random.seed(0)
        self.kernels = np.random.randn(*self.kernels_shape)
        np.random.seed(0)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient
    
class CNNLayer(Layer): #CNN su dung ham correlate o tren
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        np.random.seed(0)
        self.kernels = np.random.randn(*self.kernels_shape)
        np.random.seed(0)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += correlate(self.input[j], self.kernels[i, j], 1, "valid")
                #self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)
        kernels_rot_180=np.rot90(self.kernels,2)
        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = correlate(self.input[j], output_gradient[i], 1, "valid")
                input_gradient[j] += correlate(output_gradient[i], kernels_rot_180[i, j], 1, "full")
                #kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                #input_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return input_gradient