import numpy as np
from neural_network_engine.layers import Dense
from neural_network_engine.activations import ReLU, Softmax
from neural_network_engine.loss import CrossEntropyLoss
from neural_network_engine.model import Sequential
from neural_network_engine.trainer import Trainer
from data.mnist_loader import load_mnist


def main():

    X, y = load_mnist()

    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]

    model = Sequential([
        Dense(784, 128),
        ReLU(),
        Dense(128, 64),
        ReLU(),
        Dense(64, 10),
        Softmax()
    ])

    loss_fn = CrossEntropyLoss()
    trainer = Trainer(model, loss_fn, lr=0.1)

    epochs = 10
    batch_size = 64

    for epoch in range(epochs):
        losses = []

        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i:i+batch_size]
            y_batch = y_train[i:i+batch_size]

            loss = trainer.train_batch(x_batch, y_batch)
            losses.append(loss)

        accuracy = trainer.evaluate(X_test, y_test)
        print(f"Epoch {epoch+1} | Loss: {np.mean(losses):.4f} | Test Acc: {accuracy:.4f}")


if __name__ == "__main__":
    main()