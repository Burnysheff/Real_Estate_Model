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

# Импортируем всё, что нужно для работы с данными и моделью
from dataset import (
    load_data,               # чтение исходного CSV или парсинг
    collect_macro_features,  # сбор макроэкономических признаков
    merge_macro,             # объединение с макроданными
    impute_missing,          # заполнение пропусков
    remove_outliers,         # фильтрация выбросов
    encode_and_scale,        # кодирование категорий и масштабирование
    RealEstateDataset,       # PyTorch Dataset для табличных данных
    MACRO_CACHE              # путь к файлу-кэшу макроданных
)

# -----------------------------------------------------------
# Конфигурация: пути и логирование
# -----------------------------------------------------------

# Путь к вашему CSV с уже сгенерированными данными
DATA_PATH = "backend/model/artifacts/real_estate_8.csv"

# Папка, в которую сохраняем веса обученной модели
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Настраиваем логирование (INFO → выводим ход обучения и метрики)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Определение архитектуры нейросети
# -----------------------------------------------------------

class MLPRegressor(nn.Module):
    """
    Многослойный персептрон для регрессии:
      - Вход: размерность = число признаков (input_dim)
      - Скрытые слои: 128→64→32→16 нейронов
      - Активации: ReLU для первых трёх, Tanh перед выходом
      - Dropout 0.2 после первого ReLU
      - BatchNorm на втором слое для стабилизации
      - Выход: 1 нейрон, линейная активация (прогноз цены)
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),  # первый линейный слой
            nn.ReLU(),                   # ReLU
            nn.Dropout(0.2),             # Dropout для регуляризации
            nn.Linear(128, 64),          # второй линейный слой
            nn.ReLU(),                   # ReLU
            nn.BatchNorm1d(64),          # BatchNorm для ускорения сходимости
            nn.Linear(64, 32),           # третий линейный слой
            nn.ReLU(),                   # ReLU
            nn.Linear(32, 16),           # четвёртый линейный слой
            nn.Tanh(),                   # Tanh для сглаживания
            nn.Linear(16, 1)             # выходной слой: 1 предсказание
        )

    def forward(self, x):
        """
        Прямой проход: передаём вход через весь Sequential-блок.
        """
        return self.model(x)

# -----------------------------------------------------------
# Функции обучения и оценки модели
# -----------------------------------------------------------

def train_model(
    model: nn.Module,
    loaders: Dict[str, DataLoader],
    criterion,
    optimizer,
    num_epochs: int = 80,
    patience: int = 10,
    scheduler=None,
    device: str = 'cpu'
) -> nn.Module:
    """
    Тренировочный цикл с early stopping и опциональным scheduler:
      - loaders: словарь {'train': train_loader, 'val': val_loader}
      - criterion: функция потерь (например, L1Loss)
      - optimizer: оптимизатор (Adam)
      - num_epochs: макс. число эпох
      - patience: сколько эпох без улучшения val_loss ждём перед остановкой
      - scheduler: lr_scheduler (например, OneCycleLR) или None
      - device: 'cpu' или 'cuda'
    """
    best_loss = float('inf')
    epochs_no_improve = 0
    best_wts = model.state_dict()  # сохраним лучшие веса

    # Проходим по эпохам
    for epoch in range(1, num_epochs + 1):
        logger.info(f"Epoch {epoch}/{num_epochs}")

        # Два режима: обучение и валидация
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()   # включаем dropout и градиенты
            else:
                model.eval()    # выключаем dropout, фиксируем BN

            running_loss = 0.0

            # Итерируем по батчам
            for X_batch, y_batch in loaders[phase]:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()

                # Прямой проход + вычисление loss
                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)

                    # Только в train фазы делаем обратный проход
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # Шаг lr scheduler после optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

                # Accumulate loss over the epoch
                running_loss += loss.item() * X_batch.size(0)

            # Средний loss за эпоху
            epoch_loss = running_loss / len(loaders[phase].dataset)
            logger.info(f"{phase} Loss: {epoch_loss:.4f}")

            # Для валидации проверяем улучшение и early stopping
            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_wts = model.state_dict()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                # Останавливаем, если не улучшается дольше patience
                if epochs_no_improve >= patience:
                    logger.info("Early stopping")
                    model.load_state_dict(best_wts)
                    return model

    # После всех эпох возвращаем лучшие веса
    model.load_state_dict(best_wts)
    return model


def evaluate(model: nn.Module, loader: DataLoader, device: str = 'cpu') -> Dict[str, float]:
    """
    Оцениваем модель на тестовом DataLoader:
      - Собираем предсказания и истинные y
      - Вычисляем MAE, MSE, RMSE, R2
      - Возвращаем dict метрик
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            out = model(X_batch).cpu().numpy().flatten()
            preds.extend(out)
            trues.extend(y_batch.numpy().flatten())

    mae  = mean_absolute_error(trues, preds)
    mse  = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    r2   = r2_score(trues, preds)

    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}

# -----------------------------------------------------------
# Функции для интерпретации модели
# -----------------------------------------------------------

def shap_analysis(model, X: np.ndarray, feature_names: List[str]):
    """
    SHAP DeepExplainer:
      - берём первые 100 примеров как фон
      - считаем shap-значения для всей X
      - рисуем summary_plot
    """
    explainer = shap.DeepExplainer(model, torch.tensor(X[:100]).float())
    shap_values = explainer.shap_values(torch.tensor(X).float())
    shap.summary_plot(shap_values, X, feature_names=feature_names)


def lime_analysis(model, X: np.ndarray, feature_names: List[str]):
    """
    LIME Tabular:
      - создаём explainer на всех X
      - объясняем первое предсказание
      - выводим результат в ноутбуке
    """
    explainer = LimeTabularExplainer(
        X, feature_names=feature_names, mode='regression'
    )
    exp = explainer.explain_instance(X[0], model.forward, num_features=10)
    exp.show_in_notebook()

# -----------------------------------------------------------
# Точка входа скрипта
# -----------------------------------------------------------

def main():
    # 1) Загрузка данных из CSV
    df = load_data(DATA_PATH)

    # Если нужно было бы добавить макроэкономику:
    # dates = df['date_listed'].dt.to_pydatetime().tolist()
    # if os.path.exists(MACRO_CACHE):
    #     macro = pd.read_csv(MACRO_CACHE, index_col='date', parse_dates=['date'])
    # else:
    #     macro = collect_macro_features(dates)
    # df = merge_macro(df, macro)
    #
    # # 2) Предобработка
    # df = impute_missing(df)
    # numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # df = remove_outliers(df, numeric_cols)

    # 3) Кодирование признаков и масштабирование (лог1p для таргета)
    X, y = encode_and_scale(df)
    print(df.head())

    # 4) Разбиваем на train / val / test: сначала 70/30, потом 50/50 внутри тестовой
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    # 5) Собираем PyTorch Datasets и DataLoaders
    train_ds = RealEstateDataset(X_tr, y_tr)
    val_ds   = RealEstateDataset(X_val, y_val)
    test_ds  = RealEstateDataset(X_te, y_te)

    loaders = {
        'train': DataLoader(train_ds, batch_size=64, shuffle=True),
        'val':   DataLoader(val_ds,   batch_size=64, shuffle=False)
    }

    # 6) Инициализируем модель, loss, optimizer
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    model     = MLPRegressor(input_dim=X.shape[1]).to(device)
    criterion = nn.L1Loss()  # MAE
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-3)

    # Настройка OneCycleLR: старт 1e-4 → до 1e-3 → вниз к 1e-5
    total_steps = 150 * len(loaders['train'])  # epochs * batches
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100
    )

    # 7) Тренируем модель
    model = train_model(
        model,
        loaders,
        criterion,
        optimizer,
        num_epochs=80,
        patience=10,
        scheduler=scheduler,
        device=device
    )

    # 8) Оцениваем на тесте
    test_loader = DataLoader(test_ds, batch_size=64)
    metrics = evaluate(model, test_loader, device)
    logger.info(f"Test metrics: {metrics}")

    # 9) Сохраняем финальные веса
    torch.save(
        model.state_dict(),
        os.path.join(MODEL_DIR, 'mlp_regressor_2.pth')
    )

if __name__ == '__main__':
    main()
