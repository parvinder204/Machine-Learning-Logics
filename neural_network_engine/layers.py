import numpy as np


class Dense:
    def __init__(self, in_features: int, out_features: int):
        limit = np.sqrt(6 / (in_features + out_features))
        self.weights = np.random.uniform(-limit, limit, (in_features, out_features))
        self.bias = np.zeros((1, out_features))

        self.d_weights = None
        self.d_bias = None

        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.d_weights = np.dot(self.input.T, grad_output)
        self.d_bias = np.sum(grad_output, axis=0, keepdims=True)

        return np.dot(grad_output, self.weights.T)

    def update(self, lr: float):
        self.weights -= lr * self.d_weights
        self.bias -= lr * self.d_bias