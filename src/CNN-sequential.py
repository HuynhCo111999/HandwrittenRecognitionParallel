import numpy as np
from scipy import signal

# Base Layer
class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

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

# CNN su dung ham correlate o tren
class CNNLayer(Layer): 
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

# Pooling Layer
class MaxPoolingLayer(Layer):
    def __init__(self, input_shape, kernel_size, stride):
        #forward variable
        input_depth, input_height, input_width = input_shape
        self.depth = input_depth
        self.stride = stride
        self.input_shape = input_shape
        self.output_shape = (self.depth, int((input_height - kernel_size) / stride) + 1, int((input_width - kernel_size) / stride) + 1)
        self.kernels_size = kernel_size

        #backward variable
        self.prev_input = np.zeros(input_shape) # array have shape before pooling


    def get_pools(self, input, depth):
        # To store individual pools
        pools = []
        # Iterate over all row blocks (single block has `stride` rows)
        for i in np.arange(input.shape[0], step=self.stride):
            # Iterate over all column blocks (single block has `stride` columns)
            for j in np.arange(input.shape[1], step=self.stride):
            
                # Extract the current pool
                mat = input[i:i+self.kernels_size, j:j+self.kernels_size]
            
                # Make sure it's rectangular - has the shape identical to the pool size
                if mat.shape == (self.kernels_size, self.kernels_size):
                    # Append to the list of pools output.
                    pools.append(np.max(mat))
                    # store position in prev_input for backward propagation.
                    idx = np.where(input[i:i+self.kernels_size,j:j+self.kernels_size] == np.max(mat)) # find index where max element store in.
                    self.prev_input[depth][i + idx[0][0],j + idx[1][0]] = input[i + idx[0][0],j + idx[1][0]] # store in prev_input.
                        
        # Return all pools as a Numpy array with shape of output.
        tgt_shape = (self.output_shape[1], self.output_shape[2])
        return np.array(pools).reshape(tgt_shape)

    def max_pooling(self, input):
        # Total number of pools
        num_pools = input.shape[1]
        # Shape of the matrix after pooling - Square root of the number of pools
        # Cast it to int, as Numpy will return it as float
        # For example -> np.sqrt(16) = 4.0 -> int(4.0) = 4
        tgt_shape = (self.output_shape[1], self.output_shape[2])
        # To store the max values
        pooled = []
        
        # Iterate over all pools
        for pool in input:
            # Append the max value only
            pooled.append(np.max(pool))
            
        # Reshape to target shape
        return np.array(pooled).reshape(tgt_shape)

    def forward(self, input):
        self.input = input
        self.output = np.zeros(self.output_shape)
        for i in range(self.depth):
                # pooled = self.get_pools(self.input[i], i)
                self.output[i] = self.get_pools(self.input[i], i)

        return self.output
  
    def backward(self, output_gradient, learning_rate):
        #roll back shape of input 
        return self.prev_input
    
# Flatten Layer
class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)

# Fully Connected Layer
class FCLayer(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient

# Activation Layer
class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

# Softmax Layer
class Softmax(Layer):
    def forward(self, input):
        self.input = input
        tmp = np.exp(self.input)
        self.output = tmp / np.sum(tmp)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # This version is faster than the one presented in the video
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        # Original formula:
        # tmp = np.tile(self.output, n)
        # return np.dot(tmp * (np.identity(n) - np.transpose(tmp)), output_gradient)

#Loss function 

def reshape_labels(prediction,labels):
    label_reshape = np.zeros(prediction.shape)
    for i in range(0,len(prediction)):
        label_reshape[i][labels[i]] = 1
    return(label_reshape)

def loss_SSE(prediction, labels):
    loss_SSE = []
    labels_reshape = reshape_labels(prediction, labels)
    for i in range(0,len(prediction)):
        loss_SSE.append(np.sum((prediction[i] - labels_reshape[i])**2))
    return(np.mean(loss_SSE))

def grad_loss(prediction, labels):
    label_reshape = reshape_labels(prediction, labels)
    grad_loss_out = []
    for i in range(0,len(prediction)):
        grad_loss_out.append(2*(prediction[i] - label_reshape[i]))
    return(np.array(grad_loss_out))
    
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def grad_softmax(x):
    '''Compute the gradient of the output layer over each z, Output = softmax(z)
    Input -> Output layer you waht to do the gradient of softmax over
    Outpu -> Matrix of the gradient of each outpu w.r.t each z
    '''
    grad_softmax =[]
    for k in range(0,len(x)):
        jacobian =np.empty((x.shape[-1],x.shape[-1]))
        for i in range(0, jacobian.shape[0]):
            for j in range(0, jacobian.shape[1]):
                if i == j:
                    jacobian[i][j] =  x[k][i] * (1 - x[k][j])
                else:
                    jacobian[i][j] =  x[k][i] * (0 - x[k][j])
        grad_softmax.append(jacobian)            
    return(np.array(grad_softmax))

    #Step 1 - Calculate the Gradient of the loss w.r.t the outputs
    grad_loss_outputs = grad_loss(prediction, labels)

    #Step 2 - Calculate the Gradient of each output w.r.t each z, where output[i] = softmax(z[i])
    grad_outputs_z = grad_softmax(prediction)

# loss function and its derivative
def mse(y_true, y_pred):
    return np.mean(np.power(y_true-y_pred, 2));

def mse_prime(y_true, y_pred):
    return 2*(y_pred-y_true)/y_true.size;

def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))

def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)