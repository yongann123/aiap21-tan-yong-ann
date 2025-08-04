#!/bin/bash

PYTHON="C:/Python312/python.exe"

echo "===== Starting End-to-End ML Pipeline ====="

echo ">> [1/4] Loading data..."
"$PYTHON" src/data_loader.py

echo ">> [2/4] Preprocessing data..."
"$PYTHON" src/preprocessing.py

echo ">> [3/4] Feature engineering..."
"$PYTHON" src/feature_engineering.py

echo ">> [4/4] Training models..."
"$PYTHON" src/train_model.py

echo ">> Evaluation:"
"$PYTHON" src/evaluate_model.py

echo "===== Pipeline Complete ====="
