# BERT Sentiment Analysis (Transformer Fine-Tuning)

This project demonstrates how to fine-tune a pretrained BERT model for sentiment analysis using PyTorch and Hugging Face Transformers.

The goal of the project is to understand the internal training workflow of transformer models instead of relying on high-level training abstractions.

The project includes:
• Manual training loop implementation  
• Learning rate scheduler  
• F1 score evaluation  
• Model saving and loading  
• FastAPI inference API  


## Dataset

The project uses the IMDB movie review dataset.
Classes:

0 → Negative  
1 → Positive

Dataset is automatically downloaded using Hugging Face Datasets.


## Model

The model architecture is based on:
BERT Base (bert-base-uncased)

Architecture:

Input Text
   ↓
BERT Encoder
   ↓
Dropout
   ↓
Linear Classifier
   ↓
Sentiment Prediction



## Training the Model

Run the training script: python training/train.py


This will:
    • Load dataset  
    • Tokenize text  
    • Fine-tune BERT  
    • Apply learning rate scheduler  
    • Save trained model

Saved model: saved_models/bert_sentiment.pt



## Model Evaluation

To evaluate model performance: python training/evaluate.py
Output example: F1 Score: 0.92


## Running API Server

Start the FastAPI server:

uvicorn inference.api:app --reload

API endpoint: POST /predict

Request example:
    {
    "text": "This movie was amazing"
    }

Response:
    {
    "sentiment": "positive"
    }


## Concepts Implemented

This project demonstrates the following machine learning concepts:

• Transformer fine-tuning
• Tokenization
• Manual PyTorch training loop
• Optimizer (AdamW)
• Learning rate scheduling
• F1 score evaluation
• Model persistence (save/load)
• API deployment with FastAPI

## Future Improvements

Possible improvements:
• Mixed precision training (FP16)
• Gradient accumulation
• Early stopping
• Experiment tracking (MLflow)
• Docker deployment
• Batch inference optimization

# Learning Outcome

By completing this project, you gain hands-on experience with:

• Transformer architecture usage
• BERT fine-tuning workflow
• PyTorch training pipeline
• NLP model evaluation
• Building ML inference APIs
