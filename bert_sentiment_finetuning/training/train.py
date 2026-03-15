import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from data.dataset_loader import load_imdb_dataset, prepare_dataset
from model.bert_classifier import BertClassifier
from utils.metrics import compute_f1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    train_data, test_data = load_imdb_dataset()
    train_data = prepare_dataset(train_data)
    test_data = prepare_dataset(test_data)
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=8)
    model = BertClassifier().to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * 3
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()

    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss {total_loss/len(train_loader)}")
    torch.save(model.state_dict(), "saved_models/bert_sentiment.pt")
    print("Model Saved")


if __name__ == "__main__":
    train()