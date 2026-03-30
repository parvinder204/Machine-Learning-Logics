import torch
import torch.nn as nn
from transformers import AutoModel


class TicketClassifier(nn.Module):
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 5,
        dropout: float = 0.3,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)

        hidden_size = self.encoder.config.hidden_size  
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_labels),
        )

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_embedding = outputs.last_hidden_state[:, 0, :]

        logits = self.classifier(cls_embedding)
        return logits

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns softmax probabilities — useful at inference time."""
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
        return torch.softmax(logits, dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)