import numpy as np


class ReLU:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        grad = grad_output.copy()
        grad[self.input <= 0] = 0
        return grad


class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        return grad_output * (self.output * (1 - self.output))


class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp / np.sum(exp, axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        return grad_output