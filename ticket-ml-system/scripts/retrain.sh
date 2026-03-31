#!/bin/bash
# scripts/retrain.sh
# Retraining pipeline — runs when new labelled data arrives.
# In production: trigger via cron, Airflow, or GitHub Actions.
set -e

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "=== Retraining started at $TIMESTAMP ==="

if [ -d "experiments/best_model" ]; then
    cp -r experiments/best_model "experiments/backup_$TIMESTAMP"
    echo "  Backed up current model → experiments/backup_$TIMESTAMP"
fi

echo "=== Preprocess new data ==="
python -m training.preprocessing

echo "=== Train new model ==="
python -m training.train

echo "=== Evaluate on test set ==="
python -c "
import yaml, torch, json
from pathlib import Path
from training.model import TicketClassifier
from training.dataset import get_dataloaders
from training.evaluate import evaluate
import torch.nn as nn

with open('configs/config.yaml') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cpu')
_, _, test_loader = get_dataloaders(cfg['data']['processed_path'], batch_size=32)
meta = json.load(open('data/processed/metadata.json'))

model = TicketClassifier(cfg['model']['name'], cfg['model']['num_labels'], dropout=0.0)
model.load_state_dict(torch.load('experiments/best_model/model.pt', weights_only=True, map_location=device))
model.to(device)

metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device,
                   id2label=meta['id2label'], verbose=True)
print(f'Test F1: {metrics[\"f1\"]:.4f}  Accuracy: {metrics[\"accuracy\"]:.4f}')
"

echo "=== Retraining complete at $(date) ==="