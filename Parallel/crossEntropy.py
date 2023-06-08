import numpy as np


class CrossEntropy(object):
    def __init__(self):
        self.softmax = None
        self.labels = None
        self.loss = 0
        self.g_inputs = None

    def forward(self, inputs, labels):
        self.labels = labels
        self.loss = 0
        for i in range(inputs.shape[0]):
            self.loss += (np.sum(np.exp(inputs[i])) - inputs[i, labels[i]])
        self.loss = self.loss / inputs.shape[0]
        self.cal_softmax(inputs)
        return self.loss

    def cal_softmax(self, inputs):
        exp_prediction = np.zeros(inputs.shape, dtype=np.float32)
        self.softmax = np.zeros(inputs.shape, dtype=np.float32)
        for i in range(inputs.shape[0]):
            inputs[i, :] -= np.max(inputs[i, :])
            exp_prediction[i] = np.exp(inputs[i])
            self.softmax[i] = exp_prediction[i] / np.sum(exp_prediction[i])
        return self.softmax

    def backward(self):
        self.g_inputs = self.softmax.copy()
        for i in range(self.g_inputs.shape[0]):
            self.g_inputs[i, self.labels[i]] -= 1
        return self.g_inputs
