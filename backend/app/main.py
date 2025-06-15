import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import torch

from .schemas import RealEstateFeatures
from .model_utils import model, preprocess

app = FastAPI(title="Real Estate Pricing API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict_price(features: RealEstateFeatures):
    """
    Получает JSON с признаками, возвращает { price_rub: float }.
    """
    try:
        df = pd.DataFrame([features.dict()])
        X = preprocess(df)
        with torch.no_grad():
            pred = model(torch.from_numpy(X).float()).item()
        return {"price_rub": pred}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
