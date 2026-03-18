import mlflow
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from data.dataset_loader import load_dataset
from training.preprocess import build_vectorizer


def train():
    texts, labels = load_dataset("data/raw/tickets.csv")
    vectorizer = build_vectorizer()
    X = vectorizer.fit_transform(texts)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2
    )
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy", acc)
    joblib.dump(model, "model_registry/model.pkl")
    joblib.dump(vectorizer, "model_registry/vectorizer.pkl")
    print("Accuracy:", acc)

if __name__ == "__main__":
    train()