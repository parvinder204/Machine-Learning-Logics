from fastapi import FastAPI
import joblib

from api.schema import DocumentRequest

app = FastAPI()

model = joblib.load("model_registry/model.pkl")
vectorizer = joblib.load("model_registry/vectorizer.pkl")


@app.post("/predict")
def predict(request: DocumentRequest):
    X = vectorizer.transform([request.text])
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X).max()
    return {
        "category": prediction,
        "confidence": float(prob)
    }