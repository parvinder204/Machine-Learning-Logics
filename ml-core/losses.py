import numpy as np

class MSELoss:

    @staticmethod
    def forward(y_true, y_pred, weights=None, l2_lambda=0.0):
        mse = np.mean((y_pred - y_true) ** 2)

        if weights is not None:
            l2_term = l2_lambda * np.sum(weights ** 2)
            return mse + l2_term

        return mse

    @staticmethod
    def backward(X, y_true, y_pred, weights=None, l2_lambda=0.0):
        n = y_true.shape[0]
        gradient = (2 / n) * X.T @ (y_pred - y_true)

        if weights is not None and l2_lambda > 0:
            regularization = 2 * l2_lambda * weights
            regularization[-1] = 0  
            gradient += regularization

        return gradient


class BinaryCrossEntropy:

    @staticmethod
    def forward(y_true, y_pred, weights=None, l2_lambda=0.0):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        bce = -np.mean(
            y_true * np.log(y_pred) +
            (1 - y_true) * np.log(1 - y_pred)
        )

        if weights is not None:
            l2_term = l2_lambda * np.sum(weights ** 2)
            return bce + l2_term

        return bce

    @staticmethod
    def backward(X, y_true, y_pred, weights=None, l2_lambda=0.0):
        n = y_true.shape[0]
        gradient = (1 / n) * X.T @ (y_pred - y_true)

        if weights is not None and l2_lambda > 0:
            regularization = 2 * l2_lambda * weights
            regularization[-1] = 0  # exclude bias
            gradient += regularization

        return gradient