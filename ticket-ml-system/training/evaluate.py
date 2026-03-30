import logging
from typing import Optional

import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

logger = logging.getLogger(__name__)


def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    id2label: Optional[dict] = None,
    verbose: bool = False,
) -> dict:
   
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    f1 = f1_score(all_labels, all_preds, average="weighted")
    metrics = {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "f1": f1,
    }

    if verbose:
        target_names = (
            [id2label[i] for i in sorted(id2label)] if id2label else None
        )
        report = classification_report(all_labels, all_preds, target_names=target_names)
        cm = confusion_matrix(all_labels, all_preds)
        logger.info(f"\nClassification Report:\n{report}")
        logger.info(f"\nConfusion Matrix:\n{cm}")
        metrics["report"] = report
        metrics["confusion_matrix"] = cm.tolist()

    return metrics