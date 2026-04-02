import json
import logging
import random
import sys
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

from training.dataset import get_dataloaders
from training.evaluate import evaluate
from training.model import TicketClassifier, count_parameters


class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "time": self.formatTime(record),
            "level": record.levelname,
            "msg": record.getMessage(),
        })


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("trainer")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    scaler,
) -> dict:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for batch in tqdm(loader, desc="  train", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "elapsed": time.time() - t0,
    }


def train(config: dict) -> None:
    set_seed(config["training"]["seed"])

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(
        processed_dir=config["data"]["processed_path"],
        batch_size=config["training"]["batch_size"],
    )

    model = TicketClassifier(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
        dropout=config["model"]["dropout"],
    ).to(device)
    logger.info(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=config["training"]["weight_decay"],
    )

    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = OneCycleLR(
        optimizer,
        max_lr=float(config["training"]["learning_rate"]),
        total_steps=total_steps,
        pct_start=config["training"]["warmup_steps"] / total_steps,
        anneal_strategy="cos",
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]
    best_model_path = Path("experiments/best_model")
    best_model_path.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            "model": config["model"]["name"],
            "epochs": config["training"]["epochs"],
            "batch_size": config["training"]["batch_size"],
            "lr": config["training"]["learning_rate"],
            "dropout": config["model"]["dropout"],
            "max_length": config["model"]["max_length"],
        })

        for epoch in range(1, config["training"]["epochs"] + 1):
            logger.info(f"Epoch {epoch}/{config['training']['epochs']}")

            train_metrics = train_one_epoch(
                model, train_loader, optimizer, scheduler, criterion, device, scaler
            )
            val_metrics = evaluate(model, val_loader, criterion, device)

            logger.info(
                f"  train loss={train_metrics['loss']:.4f} acc={train_metrics['accuracy']:.4f} | "
                f"val loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f}"
            )

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
            }, step=epoch)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                torch.save(model.state_dict(), best_model_path / "model.pt")
                logger.info(f" New best model saved (val_loss={best_val_loss:.4f})")
            else:
                patience_counter += 1
                logger.info(f" No improvement. Patience {patience_counter}/{patience}")
                if patience_counter >= patience:
                    logger.info(" Early stopping triggered.")
                    break

        model.load_state_dict(torch.load(best_model_path / "model.pt", weights_only=True))
        test_metrics = evaluate(model, test_loader, criterion, device)
        logger.info(
            f"Test → loss={test_metrics['loss']:.4f} acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}"
        )

        mlflow.log_metrics({
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["f1"],
        })

        mlflow.pytorch.log_model(model, "model")
    logger.info("Training complete.")

if __name__ == "__main__":
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    train(cfg)