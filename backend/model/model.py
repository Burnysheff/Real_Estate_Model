import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap
from lime.lime_tabular import LimeTabularExplainer

from dataset import (
    load_data, collect_macro_features, merge_macro,
    impute_missing, remove_outliers, encode_and_scale,
    RealEstateDataset, MACRO_CACHE
)

# -----------------------------------------------------------
# Конфиг
# -----------------------------------------------------------
DATA_PATH = "backend/model/artifacts/real_estate_8.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Модель
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# Обучение / оценка
# -----------------------------------------------------------
def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion,
    optimizer,
    num_epochs: int = 80,
    patience: int = 10,
    scheduler=None,       # добавили параметр
    device: str = 'cpu'
) -> nn.Module:
    best_loss, epochs_no_improve = float('inf'), 0
    best_wts = model.state_dict()

    for epoch in range(1, num_epochs+1):
        logger.info(f"Epoch {epoch}/{num_epochs}")
        for phase in ['train','val']:
            model.train() if phase=='train' else model.eval()
            running = 0.0

            for X, y in loaders[phase]:
                X, y = X.to(device), y.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    preds = model(X)
                    loss = criterion(preds, y)
                    if phase=='train':
                        loss.backward()
                        optimizer.step()
                        # шаг scheduler после каждого батча
                        if scheduler is not None:
                            scheduler.step()

                running += loss.item() * X.size(0)

            epoch_loss = running / len(loaders[phase].dataset)
            logger.info(f"{phase} Loss: {epoch_loss:.4f}")

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_wts = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info("Early stopping")
                    model.load_state_dict(best_wts)
                    return model

    model.load_state_dict(best_wts)
    return model

def evaluate(model: nn.Module, loader: DataLoader, device: str='cpu'):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X,y in loader:
            X = X.to(device)
            out = model(X).cpu().numpy().flatten()
            preds.extend(out)
            trues.extend(y.numpy().flatten())
    mae = mean_absolute_error(trues,preds)
    mse = mean_squared_error(trues,preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(trues,preds)
    return {'MAE':mae,'MSE':mse,'RMSE':rmse,'R2':r2}

# -----------------------------------------------------------
# Интерпретация
# -----------------------------------------------------------
def shap_analysis(model, X: np.ndarray, feature_names: List[str]):
    expl = shap.DeepExplainer(model, torch.tensor(X[:100]).float())
    vals = expl.shap_values(torch.tensor(X).float())
    shap.summary_plot(vals, X, feature_names=feature_names)

def lime_analysis(model, X: np.ndarray, feature_names: List[str]):
    expl = LimeTabularExplainer(X, feature_names=feature_names, mode='regression')
    exp = expl.explain_instance(X[0], model.forward, num_features=10)
    exp.show_in_notebook()

# -----------------------------------------------------------
# Запуск
# -----------------------------------------------------------
def main():
    # 1) Загрузка
    df = load_data(DATA_PATH)
    # dates = df['date_listed'].dt.to_pydatetime().tolist()
    # if os.path.exists(MACRO_CACHE):
    #     macro = pd.read_csv(MACRO_CACHE, index_col='date', parse_dates=['date'])
    # else:
    #     macro = collect_macro_features(dates)
    # df = merge_macro(df, macro)
    #
    # # 2) Предобработка
    # df = impute_missing(df)
    # num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # df = remove_outliers(df, num_cols)

    # 3) Фичи/таргет
    X, y = encode_and_scale(df)

    print(df.head())

    # 4) Сплит
    Xtr, Xt, ytr, yt = train_test_split(X,y,test_size=0.3,random_state=42)
    Xv, Xte, yv, yte = train_test_split(Xt,yt,test_size=0.5,random_state=42)

    # 5) Datasets & Loaders
    tr_ds = RealEstateDataset(Xtr,ytr)
    v_ds  = RealEstateDataset(Xv, yv)
    te_ds = RealEstateDataset(Xte,yte)
    loaders = {
        'train': DataLoader(tr_ds, batch_size=64, shuffle=True),
        'val':   DataLoader(v_ds,  batch_size=64, shuffle=False)
    }

    # 6) Модель
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLPRegressor(input_dim=X.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # 7) Тренировка
    total_steps = 150 * len(loaders['train'])
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,  # максимум lr
        total_steps=total_steps,
        pct_start=0.3,  # 30% шагов на "разогрев"
        div_factor=10,  # стартовый lr = max_lr/div_factor = 1e-4
        final_div_factor=100,  # финальный lr = max_lr/final_div_factor = 1e-5
    )

    model = train_model(
        model, loaders, criterion, optimizer,
        num_epochs=80, patience=10,
        scheduler=scheduler,  # передаём scheduler
        device=device
    )

    # 8) Оценка
    metrics = evaluate(model, DataLoader(te_ds, batch_size=64), device)
    logger.info(f"Test metrics: {metrics}")

    # 9) Сохранение
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'mlp_regressor_2.pth'))

if __name__ == '__main__':
    main()
