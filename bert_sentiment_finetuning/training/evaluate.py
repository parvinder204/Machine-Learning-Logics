import torch
from torch.utils.data import DataLoader

from data.dataset_loader import load_imdb_dataset, prepare_dataset
from model.bert_classifier import BertClassifier
from utils.metrics import compute_f1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate():
    _, test_data = load_imdb_dataset()
    test_data = prepare_dataset(test_data)
    loader = DataLoader(test_data, batch_size=8)
    model = BertClassifier()
    model.load_state_dict(torch.load("saved_models/bert_sentiment.pt"))
    model.to(DEVICE)
    model.eval()

    preds = []
    labels_list = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = model(input_ids, attention_mask)
            preds.append(outputs.cpu())
            labels_list.append(labels.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels_list).numpy()
    f1 = compute_f1(preds, labels)
    print("F1 Score:", f1)


if __name__ == "__main__":
    evaluate()