import numpy as np
from losses import MSELoss
from optimizers import GradientDescent, MiniBatchIterator


class LinearRegression:
    def __init__(
        self,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        batch_size: int = None,
        early_stopping: bool = False,
        tolerance: float = 1e-6,
        l2_lambda: float = 0.0
    ):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.tolerance = tolerance
        self.l2_lambda = l2_lambda

        self.weights = None
        self.bias = None
        self.loss_history = []

    def _initialize_parameters(self, n_features: int):
        self.weights = np.zeros((n_features, 1))
        self.bias = 0.0

    def _add_bias_column(self, X: np.ndarray) -> np.ndarray:
        return np.hstack((X, np.ones((X.shape[0], 1))))

    def fit(self, X: np.ndarray, y: np.ndarray):
        y = y.reshape(-1, 1)

        X_bias = self._add_bias_column(X)
        n_samples, n_features = X_bias.shape

        self._initialize_parameters(n_features - 1)

        optimizer = GradientDescent(self.learning_rate)
        loss_fn = MSELoss()

        weights = np.vstack((self.weights, [[self.bias]]))

        prev_loss = float("inf")

        for epoch in range(self.epochs):

            if self.batch_size:
                batch_iterator = MiniBatchIterator(self.batch_size)

                for X_batch, y_batch in batch_iterator.generate(X_bias, y):
                    y_pred = X_batch @ weights
                    # gradients = loss_fn.backward(X_batch, y_batch, y_pred)
                    gradients = loss_fn.backward(
                        X_batch,
                        y_batch,
                        y_pred,
                        weights,
                        self.l2_lambda
                    )
                    weights = optimizer.step(weights, gradients)

            else:
                y_pred = X_bias @ weights
                # gradients = loss_fn.backward(X_bias, y, y_pred)
                gradients = loss_fn.backward(
                    X_bias,
                    y,
                    y_pred,
                    weights,
                    self.l2_lambda
                )
                weights = optimizer.step(weights, gradients)

            y_pred_full = X_bias @ weights
            # loss = loss_fn.forward(y, y_pred_full)
            loss = loss_fn.forward(
                y,
                y_pred_full,
                weights,
                self.l2_lambda
            )
            self.loss_history.append(loss)

            if self.early_stopping:
                if abs(prev_loss - loss) < self.tolerance:
                    break
                prev_loss = loss

        self.weights = weights[:-1]
        self.bias = weights[-1][0]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias