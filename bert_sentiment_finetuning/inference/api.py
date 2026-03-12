from fastapi import FastAPI
from pydantic import BaseModel

from inference.predict import predict

app = FastAPI()

class Review(BaseModel):
    text: str

@app.post("/predict")

def predict_sentiment(review: Review):
    sentiment = predict(review.text)
    return {
        "sentiment": sentiment
    }