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