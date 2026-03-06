## Neural Network Engine

This project implements a fully functional neural network engine from scratch using only NumPy.

No deep learning frameworks were used.
No automatic differentiation.
All forward and backward propagation logic is implemented manually.

The objective of this project is to deeply understand how modern deep learning frameworks like PyTorch and TensorFlow work internally.

## What Has Been Implemented
# Core Components

Dense (Fully Connected) Layer
ReLU Activation
Sigmoid Activation
Softmax Activation
Cross Entropy Loss
Manual Backpropagation
Mini-batch Gradient Descent
Xavier Weight Initialization

# Engine Capabilities
1. Each layer supports:
Forward pass
Caching intermediate values
Backward pass
Gradient computation
Parameter update

2. The engine supports:
Sequential model building
Modular layer stacking
Batch training
Evaluation mode


Architecture Used (MNIST)

Fully connected neural network:

Input (784)
    ↓
Dense (128)
    ↓
ReLU
    ↓
Dense (64)
    ↓
ReLU
    ↓
Dense (10)
    ↓
Softmax

Loss: Cross Entropy
Optimizer: Stochastic Gradient Descent
Batch Size: 64
Epochs: 10

## Training Results (MNIST)

Example Output:

Epoch 1 | Loss: 0.3468 | Test Acc: 0.9418
Epoch 5 | Loss: 0.0697 | Test Acc: 0.9719
Epoch 10 | Loss: 0.0268 | Test Acc: 0.9767

Final Test Accuracy: ~97%

This confirms:

Correct forward propagation
Correct gradient computation
Stable optimization
Proper softmax + cross entropy derivative implementation

## Mathematical Foundations Implemented

# Forward Pass

For Dense layer: Z = XW + b
ReLU: f(x) = max(0, x)
Softmax: softmax(x_i) = exp(x_i) / Σ exp(x_j)
Cross Entropy Loss: L = - Σ y log(y_hat)
Backpropagation (Softmax + CrossEntropy Simplification): dL/dZ = (y_hat - y) / m
Weight Gradients: dW = X^T * dZ, db = Σ dZ


## How To Run

Run training:
python -m neural_network_engine.train


## Why This Project Matters

This project demonstrates:

Understanding of matrix calculus
Implementation of chain rule manually
Deep knowledge of gradient flow
Ability to build ML systems without frameworks
Engineering-level understanding of neural network internals
This is not model usage.
This is model construction.