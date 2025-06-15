import os
from pathlib import Path
import joblib
import torch
import torch.nn as nn

# Путь к артефактам
MODEL_DIR   = Path(__file__).parent.parent / "model"
MODEL_PATH  = MODEL_DIR / "artifacts/mlp_regressor.pth"
SCALER_PATH = MODEL_DIR / "processer/scaler.pkl"
ENCODERS_PATH = MODEL_DIR / "processer/label_encoders.pkl"

# 1) Определяем архитектуру
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.model(x)

label_encoders = joblib.load(ENCODERS_PATH)

scaler = joblib.load(SCALER_PATH)

INPUT_DIM = scaler.n_features_in_
model = MLPRegressor(input_dim=INPUT_DIM)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

def preprocess(df):
    for col, le in label_encoders.items():
        df[col] = le.transform(df[col])
    X = scaler.transform(df.values)
    return X
