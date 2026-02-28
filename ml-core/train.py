import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as SklearnLR

from linear_regression import LinearRegression
from metrics import r2_score

from sklearn.linear_model import LogisticRegression as SklearnLogistic
from sklearn.datasets import make_classification

from logistic_regression import LogisticRegression
from metrics import accuracy, precision, recall


def generate_data(n_samples=200):
    np.random.seed(42)
    X = 2 * np.random.rand(n_samples, 1)
    true_w = 3.5
    true_b = 1.2
    noise = np.random.randn(n_samples, 1)
    y = true_w * X + true_b + noise
    return X, y, true_w, true_b


def train_linear():
    X, y, true_w, true_b = generate_data()

    model = LinearRegression(
        learning_rate=0.05,
        epochs=500,
        batch_size=32,
        early_stopping=True,
        l2_lambda=0.001
    )
    
    model.fit(X, y)
    y_pred = model.predict(X)

    print("Custom Model Weights:", model.weights.flatten())
    print("Custom Model Bias:", model.bias)
    print("R2 Score:", r2_score(y, y_pred))

    sklearn_model = SklearnLR()
    sklearn_model.fit(X, y)

    print("Sklearn Weights:", sklearn_model.coef_)
    print("Sklearn Bias:", sklearn_model.intercept_)

    print("Weight difference:", abs(model.weights.flatten()[0] - sklearn_model.coef_[0][0]))
    print("Bias difference:", abs(model.bias - sklearn_model.intercept_[0]))

    plt.plot(model.loss_history)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.show()


def generate_classification_data():
    X, y = make_classification(
        n_samples=500,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        random_state=42
    )
    return X, y


def train_logistic():
    X, y = generate_classification_data()

    model = LogisticRegression(
        learning_rate=0.05,
        epochs=500,
        batch_size=32,
        early_stopping=True,
        l2_lambda=0.001
    )

    model.fit(X, y)
    y_pred = model.predict(X)

    print("Custom Accuracy:", accuracy(y, y_pred))
    print("Custom Precision:", precision(y, y_pred))
    print("Custom Recall:", recall(y, y_pred))

    sklearn_model = SklearnLogistic()
    sklearn_model.fit(X, y)

    print("Sklearn Accuracy:", sklearn_model.score(X, y))

    plt.plot(model.loss_history)
    plt.title("Logistic Regression Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross Entropy")
    plt.show()


if __name__ == "__main__":
    train_linear()
    train_logistic()