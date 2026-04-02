import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def build_label_map(labels: list[str]) -> tuple[dict, dict]:
    """Returns (label→id, id→label) from a sorted unique list."""
    unique = sorted(set(labels))
    label2id = {lbl: i for i, lbl in enumerate(unique)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label2id, id2label


def tokenize_texts(
    texts: list[str],
    tokenizer: AutoTokenizer,
    max_length: int = 128,
) -> dict[str, torch.Tensor]:
    
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }



def run_preprocessing(config: dict) -> None:
    
    raw_path = Path(config["data"]["raw_path"])
    out_dir = Path(config["data"]["processed_path"])
    out_dir.mkdir(parents=True, exist_ok=True)

    model_name = config["model"]["name"]
    max_length = config["model"]["max_length"]
    val_split = config["training"]["val_split"]
    test_split = config["training"]["test_split"]
    seed = config["training"]["seed"]

    logger.info(f"Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path)

    assert "text" in df.columns and "label" in df.columns, \
        "CSV must have 'text' and 'label' columns"

    df["text"] = df["text"].apply(clean_text)
    df = df.dropna(subset=["text", "label"])
    logger.info(f"  Rows after cleaning: {len(df)}")

    label2id, id2label = build_label_map(df["label"].tolist())
    df["label_id"] = df["label"].map(label2id)
    logger.info(f"  Classes: {label2id}")
    
    train_df, test_df = train_test_split(
        df, test_size=test_split, stratify=df["label_id"], random_state=seed
    )
    effective_val = val_split / (1 - test_split)
    train_df, val_df = train_test_split(
        train_df, test_size=effective_val, stratify=train_df["label_id"], random_state=seed
    )

    logger.info(
        f"  Split sizes → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        logger.info(f"  Tokenizing {split_name}...")
        tokens = tokenize_texts(split_df["text"].tolist(), tokenizer, max_length)
        labels = torch.tensor(split_df["label_id"].tolist(), dtype=torch.long)

        torch.save(
            {
                "input_ids": tokens["input_ids"],
                "attention_mask": tokens["attention_mask"],
                "labels": labels,
            },
            out_dir / f"{split_name}.pt",
        )

    meta = {
        "label2id": label2id,
        "id2label": id2label,
        "model_name": model_name,
        "max_length": max_length,
        "num_labels": len(label2id),
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    tokenizer.save_pretrained(out_dir / "tokenizer")

    logger.info(f"Preprocessing complete. Artifacts saved to {out_dir}")


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    run_preprocessing(cfg)