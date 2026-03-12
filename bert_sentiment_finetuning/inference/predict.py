import torch
from transformers import BertTokenizer
from model.bert_classifier import BertClassifier

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertClassifier()
model.load_state_dict(torch.load("saved_models/bert_sentiment.pt"))
model.to(DEVICE)
model.eval()

def predict(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        pred = torch.argmax(outputs, dim=1).item()
    return "positive" if pred == 1 else "negative"