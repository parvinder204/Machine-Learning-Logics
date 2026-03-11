# PyTorch Deep Learning Pipeline
A modular and production-style PyTorch training pipeline built from scratch for learning and experimentation.

This project demonstrates how to structure a **real-world deep learning training workflow** including:

- Config-driven training
- Modular model architecture
- Dataset pipelines
- Training loops
- Evaluation scripts
- Learning rate scheduling
- Early stopping
- TensorBoard logging
- Checkpointing

The project trains neural networks on the **MNIST dataset** using both:

- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)

---

# Features

This pipeline includes many components used in real ML systems.

### Modular Model Design
Models are defined using `torch.nn.Module` and can easily be swapped through the configuration file.

# Supported models:

- MLP
- CNN

# Architecture:

Input (784)
↓
Linear
↓
ReLU
↓
Linear
↓
10 Classes

Expected accuracy
~97%

# CNN (Convolutional Neural Network)
A convolutional model that learns spatial patterns in images.
Architecture:

Conv2D
↓
ReLU
↓
MaxPool
↓
Conv2D
↓
ReLU
↓
MaxPool
↓
Fully Connected
↓
10 Classes

Expected accuracy:
~99%

# Dataset

This project uses the MNIST handwritten digit dataset.
Dataset details:

Images: 70,000
Training: 60,000
Testing: 10,000
Resolution: 28x28
Classes: 10 digits

# What This Project Demonstrates
This repository demonstrates the core components of a deep learning training system:
Dataset pipeline
Model definition
GPU training
Training loop
Loss computation
Optimization
Evaluation
Experiment logging
Model checkpointing
These concepts form the foundation of most modern deep learning pipelines.

# Future Improvements
Possible extensions for this project:
Train CNN on CIFAR-10
Add data augmentation
Implement ResNet architectures
Add mixed precision training
Add experiment tracking tools (Weights & Biases / MLflow)
Hyperparameter tuning


# Contents:
torch
torchvision
tensorboard
pyyaml
