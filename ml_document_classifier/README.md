# ML Document Classifier – Production ML SaaS Feature

## Overview

This project implements a **production-style Machine Learning service** that classifies support tickets or documents into predefined categories such as:
* `account_issue`
* `billing_issue`
* `technical_issue`

The system is designed following **real-world ML engineering architecture**, where the model is not just trained but also **served via an API, tracked with experiment tools, versioned, containerized, and retrained when new data arrives**.

This project demonstrates the **complete lifecycle of a machine learning feature inside a SaaS platform**.

# Project Goals
The objective of this project is to simulate how an ML team would integrate a **machine learning capability into a production backend system**.

Key goals:

* Build a **data pipeline**
* Train a **machine learning model**
* Track experiments
* Save and version models
* Serve predictions through an **API**
* Containerize the service
* Enable **model retraining**
* Add basic **monitoring**


# Architecture Explanation

## 1. Data Layer

**Directory:** `data/`
Responsible for loading and managing the dataset used during model training.
Components:

* `raw/` – Contains the dataset.
* `dataset_loader.py` – Loads the dataset for the training pipeline.

# 2. Training Pipeline

**Directory:** `training/`
This module contains the complete **machine learning training pipeline**.

Responsibilities:

* Text preprocessing
* Feature engineering
* Model training
* Model evaluation
* Logging metrics

### Training Process

1. Load dataset
2. Convert text to numerical features using **TF-IDF**
3. Split data into training and test sets
4. Train **Logistic Regression classifier**
5. Evaluate model performance
6. Log metrics to MLflow
7. Save trained model

# 3. Model Registry

**Directory:** `model_registry/`

This directory stores trained artifacts:

* `model.pkl` – trained classification model
* `vectorizer.pkl` – TF-IDF feature encoder

These files are loaded during **API inference**.


# 4. Experiment Tracking

The project integrates **MLflow** for experiment tracking.
MLflow records:
* training runs
* evaluation metrics
* experiment history
* artifacts

Run MLflow UI: mlflow ui
Open in browser: http://localhost:5000


# 5. Model Serving API

**Directory:** `api/`
The trained model is exposed using **FastAPI**, allowing other services or applications to request predictions.

Endpoint: POST /predict

Example request:
{
  "text": "I forgot my password"
}


Example response:
{
  "category": "account_issue",
  "confidence": 0.92
}

This API simulates how ML models are integrated into **real SaaS backend services**.


# 6. Monitoring

**Directory:** `monitoring/`
Logs model predictions for future analysis.

Purpose:
* debug predictions
* detect potential model drift
* analyze production behavior

Prediction logs are stored in: monitoring/predictions.log


# 7. Retraining Pipeline

**Directory:** `retraining/`
Provides a script to retrain the model when new data becomes available.
Run retraining: python retraining/retrain_pipeline.py

Future automation options:
* Cron jobs
* Airflow pipelines
* CI/CD retraining pipelines


# 8. Containerization

The project includes a **Dockerfile** to containerize the ML service.
Benefits:
* reproducible environments
* easy deployment
* cloud-ready packaging

Build container: docker build -t doc-classifier -f docker/Dockerfile .

Run container: docker run -p 8000:8000 doc-classifier

# Installation

Clone or create the project and navigate to the root directory.

Create virtual environment: python -m venv venv

Activate environment:
### Windows
venv\Scripts\activate

### Linux / Mac
source venv/bin/activate

Install dependencies: pip install -r requirements.txt

# Running the Project

## Step 1 — Train the Model
python training/train.py

Expected output: Accuracy: 0.83

Saved artifacts:
model_registry/model.pkl
model_registry/vectorizer.pkl

## Step 2 — Start MLflow

mlflow ui
Open: http://localhost:5000

## Step 3 — Start API Server

uvicorn api.main:app --reload

Server: http://127.0.0.1:8000

Swagger UI: http://127.0.0.1:8000/docs

## Step 4 — Test Prediction

Send request: POST /predict
Body:
{
  "text": "I cannot login to my account"
}

Response:
{
  "category": "account_issue",
  "confidence": 0.91
}

# Technologies Used

* Python
* Scikit-learn
* FastAPI
* MLflow
* Docker
* Pandas
* NumPy


# Future Improvements

Possible next enhancements:
1. Replace TF-IDF with **Transformer embeddings**
2. Integrate **MLflow Model Registry**
3. Implement **data drift detection**
4. Add **automated retraining**
5. Deploy on **Kubernetes**
6. Implement **CI/CD for ML pipelines**


# Conclusion

This project simulates the **full lifecycle of deploying a machine learning feature inside a SaaS system**. It demonstrates how ML models move from **training environments to production APIs**, while maintaining experiment tracking, monitoring, and retraining capabilities.

This architecture forms a strong foundation for building **scalable production ML systems**.
