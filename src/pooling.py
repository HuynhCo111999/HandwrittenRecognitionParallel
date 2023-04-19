from layer import *

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