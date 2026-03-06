from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def load_mnist():
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="liac-arff"
    )

    X = mnist.data.astype(np.float32) / 255.0
    y = mnist.target.astype(int)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    return X, y