# Intelligent Support Ticket Classifier

A **production-grade NLP system** that automatically classifies incoming support tickets using fine-tuned DistilBERT. Built with the same architecture used in real SaaS companies — from raw data all the way to a containerized REST API with experiment tracking.


## What It Does

Given a support ticket like:

> *"My payment failed but money got deducted"*

The system returns:

{
  "label": "billing",
  "confidence": 0.9421,
  "all_scores": {
    "billing": 0.9421,
    "refund": 0.0312,
    "technical_issue": 0.0187,
    "account_access": 0.0053,
    "feature_request": 0.0027
  },
  "is_confident": true,
  "latency_ms": 18.4
}

**Supported categories:** `billing` · `technical_issue` · `account_access` · `refund` · `feature_request`


## Architecture

┌─────────────────────────────────────────────────────────────┐
│  DATA LAYER                                                  │
│  Raw CSV  ──►  preprocessing.py  ──►  Tokenized Tensors     │
│               (clean, tokenize,       (train / val / test)  │
│                split, save)                                  │
└────────────────────────────┬────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────┐
│  TRAINING LAYER                                              │
│  DistilBERT backbone  +  2-layer classifier head            │
│  AdamW  +  OneCycleLR  +  early stopping                    │
│  MLflow tracks: loss, F1, accuracy per epoch                │
└────────────────────────────┬────────────────────────────────┘
                             │
                    experiments/best_model/
                             │
┌────────────────────────────▼────────────────────────────────┐
│  INFERENCE LAYER                                             │
│  predictor.py  ──►  FastAPI  ──►  Docker container          │
│  (load once)       POST /predict   (production serving)     │
└─────────────────────────────────────────────────────────────┘

## Quick Start

### Prerequisites

- Python 3.11+
- 4 GB+ RAM (8 GB recommended for training)
- GPU optional but speeds training up ~10x

### 1. Clone and set up environment

git clone <your-repo-url>
cd ticket-ml-system

python -m venv .venv
source .venv/bin/activate       
pip install -r requirements.txt

### 2. Generate data

python scripts/generate_dataset.py
# Creates data/raw/tickets.csv — 1,500 tickets across 5 classes

### 3. Preprocess
python -m training.preprocessing

### 4. Train
python -m training.train

### 5. Serve the API
uvicorn inference.api:app --reload --port 800

### 6. Full pipeline in one command
chmod +x scripts/train.sh
./scripts/train.sh

## API Reference
### `POST /predict`
Classify a single ticket.

**Request**
{ "text": "My account is locked after too many login attempts" }

**Response**
{
  "label": "account_access",
  "confidence": 0.9634,
  "all_scores": {
    "account_access": 0.9634,
    "technical_issue": 0.0221,
    "billing": 0.0089,
    "refund": 0.0034,
    "feature_request": 0.0022
  },
  "is_confident": true,
  "latency_ms": 21.3
}

### `POST /predict/batch`
Classify up to 64 tickets in one call.

**Request**
{
  "texts": [
    "How do I change my password?",
    "Refund not processed after 14 days",
    "Please add dark mode"
  ]
}

**Response**
{
  "predictions": [
    { "label": "account_access",  "confidence": 0.9411, "is_confident": true },
    { "label": "refund",          "confidence": 0.9872, "is_confident": true },
    { "label": "feature_request", "confidence": 0.9103, "is_confident": true }
  ],
  "total_latency_ms": 34.7,
  "count": 3
}

### `GET /health`
Liveness + readiness probe for Docker/Kubernetes.

{ "status": "healthy", "model_loaded": true }

## Experiment Tracking with MLflow
mlflow ui --backend-store-uri experiments/mlflow

Every training run automatically logs:
- Hyperparameters (model name, LR, batch size, dropout, epochs)
- Per-epoch metrics (train/val loss, accuracy, F1)
- Final test set metrics
- Model artifact (downloadable from the UI)

## Docker
### Build and run
docker build -t ticket-classifier .
docker run -p 8000:8000 \
  -v $(pwd)/experiments:/app/experiments \
  -v $(pwd)/data:/app/data \
  ticket-classifier


## Tech Stack
| Component | Library | Purpose |
| Model backbone | `transformers` (HuggingFace) | Pretrained DistilBERT |
| Training | `PyTorch` | Custom training loop |
| Data processing | `pandas`, `numpy` | CSV handling, splits |
| Metrics | `scikit-learn` | F1, confusion matrix |
| Experiment tracking | `MLflow` | Metrics, params, artifacts |
| API | `FastAPI` + `uvicorn` | REST endpoints |
| Validation | `Pydantic v2` | Request/response schemas |
| Containerization | `Docker` | Production deployment |
