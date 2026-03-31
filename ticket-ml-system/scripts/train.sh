#!/bin/bash
# scripts/train.sh
# Full pipeline: generate data → preprocess → train
set -e

echo "=== Step 1: Generate dataset ==="
python scripts/generate_dataset.py

echo "=== Step 2: Preprocess & tokenize ==="
python -m training.preprocessing

echo "=== Step 3: Train model ==="
python -m training.train

echo "=== Pipeline complete! ==="
echo "Start the API with: uvicorn inference.api:app --reload"