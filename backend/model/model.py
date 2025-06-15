import os
import time
import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import requests
import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import shap
from lime.lime_tabular import LimeTabularExplainer

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# -----------------------------------------------------------
# Общая Конфигурация
# -----------------------------------------------------------
API_KEYS = {
    'CBR': os.getenv('CBR_API_KEY', ''),
    'MOSBIRZHA': os.getenv('MOSBIRZHA_API_KEY', ''),
    'OIL': os.getenv('OIL_API_KEY', ''),
    'ROSSTAT': os.getenv('ROSSTAT_API_KEY', ''),
    'KEYRATE': os.getenv("KEYRATE_API_KEY"),
    'API_CRED': os.getenv("PMI_KEY"),
    'API_CRED_CPI': os.getenv("CPI_KEY")
}

DATA_DIR = 'data/'
CIAN_CSV = os.path.join(DATA_DIR, 'cian_listings.csv')
DOMKLIK_CSV = os.path.join(DATA_DIR, 'domklik_listings.csv')
MACRO_CACHE = os.path.join(DATA_DIR, 'macro_cache.csv')
MODEL_DIR = 'models/'

PER_PAGE = 150

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------
# Парсинг сайтов
# -----------------------------------------------------------

def fetch_cian_listings(base_url: str, pages: int = 5) -> pd.DataFrame:
    """
    Получение данных парсингом ЦИАН (оказалось самым сложным)
    """
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    records = []

    for p in range(1, pages + 1):
        url = base_url.format(page=p)
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        cards = soup.select('div.c6e8ba5398--item')

        for c in cards:
            try:
                price_text = c.select_one('span.c6e8ba5398--price').text
                price = float(price_text.replace('\u2009', '').replace('₽', '').replace(' ', ''))

                params = c.select_one('div.c6e8ba5398--params').text.strip()
                rooms_match = re.search(r'(\d+)[‑\s]комн', params)
                rooms = int(rooms_match.group(1)) if rooms_match else None
                area_match = re.search(r'([\d\.]+)\s*м²', params)
                area_sqm = float(area_match.group(1)) if area_match else None
                floor_match = re.search(r'(\d+)\s*/\s*(\d+)\s*эт', params)
                if floor_match:
                    floor = int(floor_match.group(1))
                    total_floors = int(floor_match.group(2))
                else:
                    floor = total_floors = None

                house_type = 'новостройка' if 'новостройка' in params.lower() else 'вторичка'

                district_elem = c.select_one('a.c6e8ba5398--region')
                district = district_elem.text.strip() if district_elem else None

                metro_elem = c.select_one('div.c6e8ba5398--geo-icon ~ div')
                metro_distance_km = None
                if metro_elem:
                    dist_text = metro_elem.text.strip()
                    if 'м' in dist_text:
                        val = float(dist_text.replace('м', '').strip())
                        metro_distance_km = val / 1000
                    elif 'км' in dist_text:
                        metro_distance_km = float(dist_text.replace('км', '').strip())

                link_elem = c.select_one('a.c6e8ba5398--link')
                link = link_elem['href'] if link_elem else None

                records.append({
                    'price_rub': price,
                    'area_sqm': area_sqm,
                    'rooms': rooms,
                    'floor': floor,
                    'total_floors': total_floors,
                    'house_type': house_type,
                    'district': district,
                    'metro_distance_km': metro_distance_km,
                    'link': link,
                    'date_listed': datetime.today().strftime('%Y-%m-%d')
                })
            except Exception:
                continue

    driver.quit()
    df = pd.DataFrame(records)
    df.to_csv('ciан_listings_extended.csv', index=False)
    return df


def fetch_domklik_listings(api_url: str) -> pd.DataFrame:
    """
    Получение данных парсингом ДомКлик
    """
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json()['listings']
    records = []
    for item in data:
        rec = {
            'price_rub': item.get('price'),
            'area': item.get('area'),
            'rooms': item.get('rooms'),
            'year_built': item.get('yearBuilt'),
            'date_listed': pd.to_datetime(item.get('listedDate'))
        }
        records.append(rec)
    df = pd.DataFrame(records)
    df.to_csv(DOMKLIK_CSV, index=False)
    return df


# -----------------------------------------------------------
# Сбор Макроэкономических Данных
# -----------------------------------------------------------

def fetch_fx_rate(pair: str, date: datetime) -> float:
    """
    Для получения курсов валют
    """
    url = f'https://api.exchangeratesapi.io/{date.strftime("%Y-%m-%d")}'
    params = {'symbols': pair.split('_')[0], 'base': pair.split('_')[1]}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()['rates'][pair.split('_')[0]]


def fetch_oil_price(date: datetime) -> float:
    """
    Цена нефти
    """
    url = f'https://api.oilpriceapi.com/v1/historical/spot'
    params = {'date': date.strftime('%Y-%m-%d'), 'symbol': 'BRENT'}
    headers = {'Authorization': f'Token {API_KEYS["OIL"]}'}
    resp = requests.get(url, params=params, headers=headers)
    resp.raise_for_status()
    return float(resp.json()['data']['price'])


def fetch_mos_index(date: datetime) -> float:
    """
    Индекс мосбиржи
    """
    url = f'https://iss.moex.com/iss/engines/stock/markets/index/boards/SIS/securities.json'
    params = {'date': date.strftime('%Y-%m-%d')}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()['marketdata']['data']
    return float(data[0][data[0].index('LAST')])


def load_key_rate_series(
        from_date: str = "2020-01-01",
        per_page: int = PER_PAGE
) -> pd.Series:
    """
    Скачивает все записи ключевой ставки ЦБ РФ (сервисный метод)
    """
    records = []
    page = 1
    while True:
        resp = requests.get(
            API_KEYS['KEYRATE'],
            params={
                "from_date": from_date,
                "page": page,
                "per_page": per_page
            }
        )
        resp.raise_for_status()
        js = resp.json()
        data = js.get("data", [])
        print("emit")
        time.sleep(0.1)
        print("time")
        if not data:
            break
        for rec in data:
            dt = pd.to_datetime(rec["date"]).date()
            rate = float(rec["rate"])
            records.append((dt, rate))
        total_pages = js.get("total_pages", 1)
        if page >= total_pages:
            break
        page += 1

    records.sort(key=lambda x: x[0])
    dates, rates = zip(*records)
    return pd.Series(rates, index=dates)


_key_rate_series = load_key_rate_series()
print(_key_rate_series)


def get_key_rate(ts: pd.Timestamp) -> Optional[float]:
    """
    Ключевая ставка.
    """
    d = ts.date()
    if d in _key_rate_series.index:
        return _key_rate_series.loc[d]
    prev = _key_rate_series.index[_key_rate_series.index <= d]
    if len(prev):
        print(_key_rate_series.loc[prev.max()])
        return _key_rate_series.loc[prev.max()]
    print("A")
    return None

def fetch_pmi(date: datetime) -> float:
    """
    PMI для России
    """
    url = (
        "https://api.tradingeconomics.com/historical/"
        "country/russia/indicator/manufacturing%20pmi"
        f"?c={API_KEYS['API_CRED']}&f=json"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df = df[df['DateTime'].dt.date <= date.date()]
    if df.empty:
        raise KeyError(f"No PMI data available on or before {date.date()}")
    latest = df.sort_values('DateTime', ascending=False).iloc[0]
    return float(latest['Close'])

def fetch_cpi(date: datetime) -> float:
    """
    Индекс потребительских цен (CPI)
    """
    url = (
        "https://api.tradingeconomics.com/historical/"
        "country/russia/indicator/consumer%20price%20index"
        f"?c={API_KEYS['API_CRED']}&f=json"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df = df[df['DateTime'].dt.date <= date.date()]
    if df.empty:
        raise KeyError(f"No CPI data available on or before {date.date()}")

    latest = df.sort_values('DateTime', ascending=False).iloc[0]
    return float(latest['Close'])

def fetch_unemployment_rate(date: datetime) -> float:
    """
    Уровень безработицы
    """
    url = (
        "https://api.tradingeconomics.com/historical/"
        "country/russia/indicator/unemployment%20rate"
        f"?c={API_KEYS['API_CRED']}&f=json"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df = df[df['DateTime'].dt.date <= date.date()]
    if df.empty:
        raise KeyError(f"No unemployment data available on or before {date.date()}")

    latest = df.sort_values('DateTime', ascending=False).iloc[0]
    return float(latest['Close'])

def fetch_gdp_growth(date: datetime) -> float:
    """
    Темп прироста ВВП
    """
    url = (
        "https://api.tradingeconomics.com/historical/"
        "country/russia/indicator/gdp%20growth"
        f"?c={API_KEYS['API_CRED']}&f=json"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df = df[df['DateTime'].dt.date <= date.date()]
    if df.empty:
        raise KeyError(f"No GDP growth data available on or before {date.date()}")

    latest = df.sort_values('DateTime', ascending=False).iloc[0]
    return float(latest['Close'])

def fetch_household_debt_to_gdp(date: datetime) -> float:
    """
    Уровень закредитованности населения
    """
    url = (
        "https://api.tradingeconomics.com/historical/"
        "country/russia/indicator/households%20debt%20to%20gdp"
        f"?c={API_KEYS['API_CRED']}&f=json"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df = df[df['DateTime'].dt.date <= date.date()]
    if df.empty:
        raise KeyError(f"No Households Debt to GDP data on or before {date.date()}")

    latest = df.sort_values('DateTime', ascending=False).iloc[0]
    return float(latest['Close'])  # возвращаем процент от ВВП

def fetch_mortgage_rate(date: datetime) -> float:
    """
    Процентную ставка по ипотечным кредитам
    """
    url = (
        "https://api.tradingeconomics.com/historical/"
        "country/russia/indicator/mortgage%20rate"
        f"?c={API_KEYS['API_CRED']}&f=json"
    )
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()

    df = pd.DataFrame(data)
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    df = df[df['DateTime'].dt.date <= date.date()]
    if df.empty:
        raise KeyError(f"No mortgage rate data available on or before {date.date()}")

    latest = df.sort_values('DateTime', ascending=False).iloc[0]
    return float(latest['Close'])


def collect_macro_features(dates: List[datetime]) -> pd.DataFrame:
    records = []
    for dt in sorted(set(dates)):
        try:
            rec = {
                'date': dt,
                'dollar_rate': fetch_fx_rate('USD_RUB', dt), # Цена доллара
                'euro_rate': fetch_fx_rate('EUR_RUB', dt), # Цена евро
                'oil_price': fetch_oil_price(dt), # Цена на нефть
                'key_rate': get_key_rate(dt), # Ключевая ставка ЦБ
                'moex_index': fetch_mos_index(dt), # Индекс мосбиржи


                'pmi': fetch_pmi(dt), # Индекс деловой активности
                'cpi': fetch_cpi(dt), # Индекс потребительских цен
                'unemployment': fetch_unemployment_rate(dt), # Уровень безработицы
                'gdp_growth': fetch_gdp_growth(dt), # Уровень роста ВВП
                'mortgage_rate': fetch_mortgage_rate(dt), # Годовая процентная ставка на покупку жилья
                'households_debt_to_gdp': fetch_household_debt_to_gdp(dt) # Уровень закредитованности населения
            }
            records.append(rec)
        except Exception as e:
            logger.warning(f"Macro fetch fail {dt}: {e}")
    df = pd.DataFrame(records).set_index('date')
    df.to_csv(MACRO_CACHE)
    return df


# -----------------------------------------------------------
# Загрузка данных
# -----------------------------------------------------------
def load_aggregator_data() -> pd.DataFrame:
    if not os.path.exists(CIAN_CSV):
        fetch_cian_listings('https://www.cian.ru/cat.php?deal_type=sale&engine_version=2&page={page}', pages=5)
    if not os.path.exists(DOMKLIK_CSV):
        fetch_domklik_listings('https://api.domclick.ru/v1/listings')
    df1 = pd.read_csv(CIAN_CSV, parse_dates=['date_listed'])
    df2 = pd.read_csv(DOMKLIK_CSV, parse_dates=['date_listed'])
    df = pd.concat([df1, df2], ignore_index=True)
    return df


def load_data() -> pd.DataFrame:
    df = load_aggregator_data()
    return df


# -----------------------------------------------------------
# Предобработка данных
# -----------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=['date_listed'])
    return df


def merge_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(macro, how='left', left_on='date_listed', right_index=True)
    return df


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object', 'bool', 'category']).columns

    imputer_num = KNNImputer(n_neighbors=5)
    df[num_cols] = imputer_num.fit_transform(df[num_cols])

    imputer_cat = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = imputer_cat.fit_transform(df[cat_cols])
    return df


def remove_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask_iqr = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)

    z_scores = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask_z = ~(z_scores > 3).any(axis=1)

    df_clean = df[mask_iqr & mask_z].reset_index(drop=True)
    return df_clean


def encode_and_scale(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    y = df['price_rub'].values
    X = df.drop(columns=['price_rub', 'some_id', 'date_listed'])

    cat_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    if 'repair_level' in cat_cols:
        ordinal_map = ['none', 'cosmetic', 'capital', 'designer']
        ord_enc = OrdinalEncoder(categories=[ordinal_map])
        X[['repair_level']] = ord_enc.fit_transform(X[['repair_level']])
        cat_cols.remove('repair_level')

    onehot = OneHotEncoder(sparse=False, handle_unknown='ignore')
    oh = onehot.fit_transform(X[cat_cols])
    oh_cols = onehot.get_feature_names_out(cat_cols)
    df_oh = pd.DataFrame(oh, columns=oh_cols, index=X.index)

    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(X[num_cols])

    X_final = np.hstack([X_num, df_oh.values])
    return X_final, y


# -----------------------------------------------------------
# Собственно сама модель
# -----------------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super(MLPRegressor, self).__init__()
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
# Обучающий цикл
# -----------------------------------------------------------
def train_model(model: nn.Module,
                dataloaders: Dict[str, DataLoader],
                criterion,
                optimizer,
                num_epochs: int = 80,
                patience: int = 10,
                device: str = 'cpu') -> nn.Module:
    best_loss = float('inf')
    best_model_wts = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        logger.info(f'Epoch {epoch + 1}/{num_epochs}')
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            for inputs, targets in dataloaders[phase]:
                inputs = inputs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            logger.info(f'{phase} Loss: {epoch_loss:.4f}')

            if phase == 'val':
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = model.state_dict().copy()
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    logger.info('Early stopping')
                    model.load_state_dict(best_model_wts)
                    return model
    model.load_state_dict(best_model_wts)
    return model


def evaluate(model: nn.Module, dataloader: DataLoader, device: str = 'cpu') -> Dict[str, float]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs).cpu().numpy().flatten()
            preds.extend(outputs)
            trues.extend(targets.numpy().flatten())
    mae = mean_absolute_error(trues, preds)
    mse = mean_squared_error(trues, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(trues, preds)
    return {'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'R2': r2}


# -----------------------------------------------------------
# Интерпретируем (смотрим как что влияет на целевую переменную)
# -----------------------------------------------------------
def shap_analysis(model: nn.Module, X: np.ndarray, feature_names: List[str]):
    explainer = shap.DeepExplainer(model, torch.tensor(X[:100]).float())
    shap_values = explainer.shap_values(torch.tensor(X).float())
    shap.summary_plot(shap_values, X, feature_names=feature_names)


def lime_analysis(model: nn.Module, X: np.ndarray, feature_names: List[str]):
    explainer = LimeTabularExplainer(
        X, feature_names=feature_names, mode='regression'
    )
    exp = explainer.explain_instance(X[0], model.forward, num_features=10)
    exp.show_in_notebook()

# -----------------------------------------------------------
# Сервисная функция
# -----------------------------------------------------------
class RealEstateDataset(Dataset):
def __init__(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).float().unsqueeze(1)  # делаем (N,1)

        self.X = X
        self.y = y

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        # Возвращает кортеж (features, target) для индекса idx
        return self.X[idx], self.y[idx]

# -----------------------------------------------------------
# Вызов всего-всего
# -----------------------------------------------------------
def main():
    df = load_data("data/real_estate_7.csv")
    macro_dates = df['date_listed'].dt.to_pydatetime().tolist()
    if os.path.exists(MACRO_CACHE):
        macro_df = pd.read_csv(MACRO_CACHE, index_col='date', parse_dates=['date'])
    else:
        macro_df = collect_macro_features(macro_dates)
    df = merge_macro(df, macro_df)

    df = impute_missing(df)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = remove_outliers(df, numeric_cols)

    X, y = encode_and_scale(df)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42)

    train_ds = RealEstateDataset(X_train, y_train)
    val_ds = RealEstateDataset(X_val, y_val)
    test_ds = RealEstateDataset(X_test, y_test)

    dataloaders = {
        'train': DataLoader(train_ds, batch_size=64, shuffle=True),
        'val': DataLoader(val_ds, batch_size=64, shuffle=False)
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MLPRegressor(input_dim=X.shape[1]).to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)

    model = train_model(model, dataloaders, criterion, optimizer, num_epochs=80, patience=10, device=device)
    metrics = evaluate(model, DataLoader(test_ds, batch_size=64), device)
    logger.info(f'Test metrics: {metrics}')

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, 'artifacts/mlp_regressor.pth'))

    # shap_analysis(model, X_test, list(df.drop(columns=['price_rub']).columns))
    # lime_analysis(model, X_test, list(df.drop(columns=['price_rub']).columns))


if __name__ == '__main__':
    main()
