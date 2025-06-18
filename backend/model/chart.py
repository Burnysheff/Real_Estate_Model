import torch
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Предполагаем, что MLPRegressor определён в model_utils.py
from model_utils import MLPRegressor, preprocess

# 1) Загрузим модель и препроцессоры
MODEL_PATH = Path("artifacts/mlp_regressor.pth")
SCALER_PATH = Path("models/scaler.pkl")
ENCODERS_PATH = Path("models/label_encoders.pkl")

# Загружаем препроцессоры
enc = joblib.load(ENCODERS_PATH)
scaler = joblib.load(SCALER_PATH)

# Загружаем модель
# Определяем входную размерность по scaler
input_dim = scaler.n_features_in_
model = MLPRegressor(input_dim=input_dim)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# 2) Определяем базовую точку (медиана/мода из обучающего датасета)
df_train = pd.read_csv("data/real_estate_7.csv", parse_dates=["date_listed"])

# числовые и категориальные столбцы
cat_cols = enc['cat_cols']
num_cols = enc['num_cols']
ord_enc = enc.get('ordinal_encoder')

baseline = {}
# медианы числовых
for c in num_cols:
    baseline[c] = df_train[c].median()
# моды категориальных
for c in cat_cols:
    baseline[c] = df_train[c].mode().iloc[0]
# если есть ordinal для renovation
if ord_enc:
    baseline['renovation'] = df_train['renovation'].mode().iloc[0]

# 3) Выбираем признак для исследования
feature = "area_sqm"   # можно заменить на любой числовой
# создаём сетку значений от мин до макс
grid = np.linspace(df_train[feature].min(), df_train[feature].max(), 100)

# 4) Формируем DataFrame для предсказаний
records = []
for val in grid:
    rec = baseline.copy()
    rec[feature] = val
    records.append(rec)
df_grid = pd.DataFrame(records)

# 5) Предобработка и инференс
X_grid = preprocess(df_grid)
with torch.no_grad():
    preds_log = model(torch.from_numpy(X_grid).float()).numpy().flatten()

# если модель обучена на log1p(price), возвращаем рубли
preds_price = np.expm1(preds_log)

# 6) Строим график
plt.plot(grid, preds_price)
plt.xlabel(feature)
plt.ylabel("Predicted price, ₽")
plt.title(f"Dependence of predicted price on '{feature}'")
plt.tight_layout()
plt.show()
