import numpy as np


class FullyConnect(object):
    def __init__(self, inFeatureSzie, outFeatureSize):
        self.weights = np.random.standard_normal(
            (inFeatureSzie, outFeatureSize))/100
        self.biases = np.random.standard_normal(outFeatureSize)/100

        self.g_weights = None
        self.g_biases = None
        self.g_inputs = None

    def forward(self, inputs):
        self.inputs = inputs.copy()
        self.outputs = np.dot(self.inputs, self.weights) + self.biases
        return self.outputs

    def backward(self, gradient_loss_to_this_outputs):
        self.g_inputs = np.dot(
            gradient_loss_to_this_outputs, np.transpose(self.weights))
        self.g_weights = np.zeros(shape=self.weights.shape, dtype=np.float32)
        self.g_biases = np.zeros(shape=self.biases.shape, dtype=np.float32)
        for i in range(gradient_loss_to_this_outputs.shape[0]):
            self.g_weights += (np.dot(self.inputs[i][:, np.newaxis],
                               gradient_loss_to_this_outputs[i][np.newaxis, :]))
            self.g_biases += gradient_loss_to_this_outputs[i]
        return self.g_inputs

    def update_parameters(self, lr):
        self.weights -= self.g_weights * lr / self.inputs.shape[0]
        self.biases -= self.g_biases * lr / self.inputs.shape[0]
