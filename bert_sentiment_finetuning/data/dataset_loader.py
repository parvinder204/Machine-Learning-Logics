from datasets import load_dataset
from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def load_imdb_dataset():
    dataset = load_dataset("imdb")
    train_data = dataset["train"]
    test_data = dataset["test"]
    return train_data, test_data


def tokenize(example):
    return tokenizer(
        example["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )


def prepare_dataset(dataset):
    dataset = dataset.map(tokenize, batched=True)
    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"]
    )
    return dataset