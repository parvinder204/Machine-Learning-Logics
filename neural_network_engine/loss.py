import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.predictions = None
        self.targets = None

    def forward(self, predictions, targets):
        self.predictions = predictions
        self.targets = targets

        m = targets.shape[0]
        loss = -np.sum(targets * np.log(predictions + 1e-15)) / m
        return loss

    def backward(self):
        m = self.targets.shape[0]
        return (self.predictions - self.targets) / m