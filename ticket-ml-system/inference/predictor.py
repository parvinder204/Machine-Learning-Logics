import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml
from transformers import AutoTokenizer

from training.model import TicketClassifier

logger = logging.getLogger(__name__)


@dataclass
class Prediction:
    label: str
    confidence: float
    all_scores: dict[str, float]
    is_confident: bool  


class TicketPredictor:
    def __init__(
        self,
        model_path: str | Path,
        processed_dir: str | Path,
        config: dict,
        device: Optional[str] = None,
    ):
        self.config = config
        self.threshold = config["inference"]["confidence_threshold"]

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available()
                else "cpu"
            )
        logger.info(f"Predictor using device: {self.device}")

        meta_path = Path(processed_dir) / "metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)

        self.label2id: dict = meta["label2id"]
        self.id2label: dict = {int(k): v for k, v in meta["id2label"].items()}
        self.max_length: int = meta["max_length"]

    
        tokenizer_path = Path(processed_dir) / "tokenizer"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        logger.info("Tokenizer loaded.")

        self.model = TicketClassifier(
            model_name=meta["model_name"],
            num_labels=meta["num_labels"],
            dropout=0.0, 
        )
        state_dict = torch.load(
            Path(model_path) / "model.pt",
            map_location=self.device,
            weights_only=True,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        logger.info(f"Model loaded from {model_path}")

    def _tokenize(self, texts: list[str]) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].to(self.device),
            "attention_mask": encoded["attention_mask"].to(self.device),
        }

    def predict(self, text: str) -> Prediction:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str]) -> list[Prediction]:
        tokens = self._tokenize(texts)

        with torch.no_grad():
            logits = self.model(**tokens)
            probs = torch.softmax(logits, dim=-1).cpu()

        predictions = []
        for prob_row in probs:
            all_scores = {
                self.id2label[i]: round(float(prob_row[i]), 4)
                for i in range(len(self.id2label))
            }
            top_idx = int(prob_row.argmax())
            top_label = self.id2label[top_idx]
            confidence = float(prob_row[top_idx])

            predictions.append(Prediction(
                label=top_label,
                confidence=round(confidence, 4),
                all_scores=all_scores,
                is_confident=confidence >= self.threshold,
            ))

        return predictions

def load_predictor() -> TicketPredictor:
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)

    return TicketPredictor(
        model_path=config["inference"]["model_path"],
        processed_dir=config["data"]["processed_path"],
        config=config,
    )