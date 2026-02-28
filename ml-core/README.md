# ML Core – Linear & Logistic Regression From Scratch

This project implements Linear Regression and Logistic Regression from scratch using NumPy, without relying on high-level ML frameworks.

The goal is to understand and implement the mathematical foundations of supervised learning algorithms, including loss computation, gradient calculation, and optimization.


## Implemented Algorithms

### 1. Linear Regression

Mathematical Model: y = XW + b

Loss Function: L = (1/n) * Σ (y - ŷ)²

Gradient: ∂L/∂W = (2/n) Xᵀ (XW - y)

Features Implemented:

- Vectorized forward pass
- Mean Squared Error (MSE)
- Analytical gradient computation
- Batch Gradient Descent
- Mini-batch Gradient Descent
- Early stopping
- R² metric
- Loss curve plotting
- Verification against sklearn


### 2. Logistic Regression

Mathematical Model:

p = sigmoid(XW + b)

Sigmoid: σ(z) = 1 / (1 + e⁻ᶻ)

Binary Cross Entropy Loss: L = -(1/n) Σ [ y log(p) + (1 - y) log(1 - p) ]

Gradient: ∂L/∂W = (1/n) Xᵀ (p - y)

Features Implemented:

- Sigmoid activation
- Binary Cross Entropy
- Vectorized gradients
- Mini-batch training
- Early stopping
- Accuracy
- Precision
- Recall
- Decision boundary learning
- Sklearn comparison


## Installation

Install dependencies: pip install numpy matplotlib scikit-learn

## How To Run

python train.py

(you can customize main function in train.py)

## Expected Output

Linear Regression:
- Decreasing MSE loss curve
- Learned weights close to sklearn
- High R² score

Logistic Regression:
- Decreasing BCE loss curve
- Accuracy comparable to sklearn
- Small performance difference


## Engineering Design Decisions

- Fully vectorized computations
- Modular loss and optimizer separation
- Clean parameter initialization
- Bias handled via augmented feature column
- Stable BCE using clipping
- Reproducible data generation


## Learning Objectives

By completing this project, you have:

- Implemented supervised learning from first principles
- Derived and coded analytical gradients
- Built your own training loop
- Understood optimization behavior
- Validated correctness against sklearn