import numpy as np


class GradientDescent:
    """
    Standard Gradient Descent Optimizer
    """

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def step(self, weights: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        return weights - self.learning_rate * gradients


class MiniBatchIterator:

    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def generate(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        indices = np.random.permutation(n)

        for start_idx in range(0, n, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            yield X[batch_indices], y[batch_indices]
            
class Momentum:

    def __init__(self, learning_rate=0.01, beta=0.9):
        self.learning_rate = learning_rate
        self.beta = beta
        self.velocity = None

    def step(self, weights, gradients):
        if self.velocity is None:
            self.velocity = np.zeros_like(weights)

        self.velocity = self.beta * self.velocity + (1 - self.beta) * gradients
        return weights - self.learning_rate * self.velocity