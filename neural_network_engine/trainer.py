import numpy as np


class Trainer:
    def __init__(self, model, loss_fn, lr=0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr

    def train_batch(self, x, y):
        predictions = self.model.forward(x)

        loss = self.loss_fn.forward(predictions, y)

        grad = self.loss_fn.backward()
        self.model.backward(grad)

        self.model.update(self.lr)

        return loss

    def evaluate(self, x, y):
        predictions = self.model.forward(x)
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)

        accuracy = np.mean(pred_classes == true_classes)
        return accuracy