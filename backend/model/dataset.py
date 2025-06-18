import os
import time
import re
import torch
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.impute import KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from torch.utils.data import Dataset

# -----------------------------------------------------------
# Конфигурация и пути
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

DATA_DIR = 'data'
CIAN_CSV = os.path.join(DATA_DIR, 'cian_listings.csv')
DOMKLIK_CSV = os.path.join(DATA_DIR, 'domklik_listings.csv')
MACRO_CACHE = os.path.join(DATA_DIR, 'macro_cache.csv')
PER_PAGE = 150


# -----------------------------------------------------------
# Утилиты конвертации и унификации
# -----------------------------------------------------------
def convert_distance(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip().lower()
    if 'мин' in s:
        num = float(s.replace('мин', '').strip())
        return num * 0.06
    if 'м' in s and 'км' not in s:
        num = float(s.replace('м', '').strip())
        return num / 1000
    if 'км' in s:
        return float(s.replace('км', '').strip())
    try:
        return float(s)
    except:
        return np.nan


synonyms = {
    'material': {
        'панельный': 'панель', 'панель': 'панель',
        'монолитный': 'монолит', 'монолит': 'монолит',
        'кирпичный': 'кирпич', 'кирпич': 'кирпич',
        'деревянный': 'дерево', 'дерево': 'дерево',
        'газобетонный': 'газобетон', 'газобетон': 'газобетон',
        'керамзитобетонный': 'керамзитобетон', 'керамзитобетон': 'керамзитобетон',
        'брус': 'брус'
    },
    'house_type': {'новостройка': 'новостройка', 'вторичка': 'вторичка', 'бу': 'вторичка'},
    'renovation_num': {
        'без': 'без', 'без ремонта': 'без',
        'косметический': 'косметический', 'косметический ремонт': 'косметический',
        'капитальный': 'капитальный', 'капитальный ремонт': 'капитальный',
        'дизайнерский': 'дизайнерский', 'дизайнерский ремонт': 'дизайнерский'
    },
    'layout': {'изолированная': 'изолированные', 'изолированные': 'изолированные',
               'смежная': 'смежные', 'смежные': 'смежные'},
    'view': {'во двор': 'во двор', 'на улицу': 'на улицу', 'панорамный': 'панорамный'}
}


# -----------------------------------------------------------
# Парсинг объявлений
# -----------------------------------------------------------
def fetch_cian_listings(base_url: str, pages: int = 5) -> pd.DataFrame:
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
                price = float(
                    c.select_one('span.c6e8ba5398--price').text
                        .replace('\u2009', '').replace('₽', '').replace(' ', '')
                )
                params = c.select_one('div.c6e8ba5398--params').text.strip()
                rooms = int(re.search(r'(\d+)[-\s]комн', params).group(1)) \
                    if re.search(r'(\d+)[-\s]комн', params) else None
                area_sqm = float(re.search(r'([\d\.]+)\s*м²', params).group(1)) \
                    if re.search(r'([\d\.]+)\s*м²', params) else None
                fm = re.search(r'(\d+)\s*/\s*(\d+)\s*эт', params)
                floor, total_floors = (int(fm.group(1)), int(fm.group(2))) if fm else (None, None)
                house_type = 'новостройка' if 'новостройка' in params.lower() else 'вторичка'
                district_elem = c.select_one('a.c6e8ba5398--region')
                district = district_elem.text.strip() if district_elem else None
                metro_elem = c.select_one('div.c6e8ba5398--geo-icon ~ div')
                metro_distance_km = None
                if metro_elem:
                    dt = metro_elem.text.strip()
                    metro_distance_km = convert_distance(dt)
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
    df.to_csv(CIAN_CSV, index=False)
    return df


def fetch_domklik_listings(api_url: str) -> pd.DataFrame:
    resp = requests.get(api_url)
    resp.raise_for_status()
    data = resp.json().get('listings', [])
    records = []
    for item in data:
        records.append({
            'price_rub': item.get('price'),
            'area_sqm': item.get('area'),
            'rooms': item.get('rooms'),
            'year_built': item.get('yearBuilt'),
            'date_listed': pd.to_datetime(item.get('listedDate')).strftime('%Y-%m-%d')
        })
    df = pd.DataFrame(records)
    df.to_csv(DOMKLIK_CSV, index=False)
    return df


# -----------------------------------------------------------
# Макроэкономика
# -----------------------------------------------------------
def fetch_fx_rate(pair: str, date: datetime) -> float:
    url = f'https://api.exchangeratesapi.io/{date.strftime("%Y-%m-%d")}'
    base, sym = pair.split('_')[1], pair.split('_')[0]
    resp = requests.get(url, params={'symbols': sym, 'base': base})
    resp.raise_for_status()
    return resp.json()['rates'][sym]


def fetch_oil_price(date: datetime) -> float:
    url = 'https://api.oilpriceapi.com/v1/historical/spot'
    resp = requests.get(
        url,
        params={'date': date.strftime('%Y-%m-%d'), 'symbol': 'BRENT'},
        headers={'Authorization': f'Token {API_KEYS["OIL"]}'}
    )
    resp.raise_for_status()
    return float(resp.json()['data']['price'])


def fetch_mos_index(date: datetime) -> float:
    url = 'https://iss.moex.com/iss/engines/stock/markets/index/boards/SIS/securities.json'
    resp = requests.get(url, params={'date': date.strftime('%Y-%m-%d')})
    resp.raise_for_status()
    data = resp.json()['marketdata']['data']
    return float(data[0][data[0].index('LAST')])


def load_key_rate_series(from_date="2020-01-01", per_page=PER_PAGE) -> pd.Series:
    records, page = [], 1
    url = 'https://api.oilpriceapi.com/v1/historical/spot'
    while True:
        resp = requests.get(API_KEYS['KEYRATE'], params={
            "from_date": from_date, "page": page, "per_page": per_page
        })
        resp.raise_for_status()
        js = resp.json()
        data = js.get("data", [])
        if not data:
            break
        for rec in data:
            records.append((pd.to_datetime(rec["date"]).date(), float(rec["rate"])))
        if page >= js.get("total_pages", 1):
            break
        page += 1
    records.sort(key=lambda x: x[0])
    dates, rates = zip(*records)
    return pd.Series(rates, index=dates)


# _key_rate_series = load_key_rate_series()

# def get_key_rate(ts: datetime) -> Optional[float]:
#     d = ts.date()
#     if d in _key_rate_series.index:
#         return _key_rate_series.loc[d]
#     prev = _key_rate_series.index[_key_rate_series.index<=d]
#     return _key_rate_series.loc[prev.max()] if len(prev) else None

def fetch_indicator(name: str, date: datetime) -> float:
    url = f"https://api.tradingeconomics.com/historical/country/russia/indicator/{name}"
    resp = requests.get(f"{url}?c={API_KEYS['API_CRED']}&f=json")
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df[df['DateTime'].dt.date <= date.date()]
    if df.empty:
        raise KeyError(f"No {name} on or before {date.date()}")
    return float(df.sort_values('DateTime', ascending=False).iloc[0]['Close'])


def fetch_pmi(date): return fetch_indicator("manufacturing%20pmi", date)


def fetch_cpi(date): return fetch_indicator("consumer%20price%20index", date)


def fetch_unemployment_rate(d): return fetch_indicator("unemployment%20rate", d)


def fetch_gdp_growth(d): return fetch_indicator("gdp%20growth", d)


def fetch_household_debt_to_gdp(d): return fetch_indicator("households%20debt%20to%20gdp", d)


def fetch_mortgage_rate(d):return fetch_indicator("mortgage%20rate", d)


def collect_macro_features(dates: List[datetime]) -> pd.DataFrame:
    recs = []
    for dt in sorted(set(dates)):
        try:
            recs.append({
                'date': dt,
                'dollar_rate': fetch_fx_rate('USD_RUB', dt),
                'euro_rate': fetch_fx_rate('EUR_RUB', dt),
                'oil_price': fetch_oil_price(dt),
                # 'key_rate':    get_key_rate(dt),
                'moex_index': fetch_mos_index(dt),
                'pmi': fetch_pmi(dt),
                'cpi': fetch_cpi(dt),
                'unemployment': fetch_unemployment_rate(dt),
                'gdp_growth': fetch_gdp_growth(dt),
                'mortgage_rate': fetch_mortgage_rate(dt),
                'households_debt_to_gdp': fetch_household_debt_to_gdp(dt)
            })
        except Exception as e:
            print(f"Macro fetch fail {dt}: {e}")
    df = pd.DataFrame(recs).set_index('date')
    df.to_csv(MACRO_CACHE)
    return df


# -----------------------------------------------------------
# Загрузка и объединение данных
# -----------------------------------------------------------
def load_aggregator_data() -> pd.DataFrame:
    if not os.path.exists(CIAN_CSV):
        fetch_cian_listings('https://www.cian.ru/cat.php?deal_type=sale&page={page}', 5)
    if not os.path.exists(DOMKLIK_CSV):
        fetch_domklik_listings('https://api.domclick.ru/v1/listings')
    df1 = pd.read_csv(CIAN_CSV, parse_dates=['date_listed'])
    df2 = pd.read_csv(DOMKLIK_CSV, parse_dates=['date_listed'])
    return pd.concat([df1, df2], ignore_index=True)


def load_data(path: str = None) -> pd.DataFrame:
    return load_aggregator_data() if path is None else pd.read_csv(path, parse_dates=['date_listed'])


def merge_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    return df.merge(macro, how='left', left_on='date_listed', right_index=True)


# -----------------------------------------------------------
# Предобработка
# -----------------------------------------------------------
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Удаляем строки с >30% пропусков
    thresh = int(0.7 * df.shape[1])
    df = df.dropna(thresh=thresh).copy()
    # Приводим дистанции
    for c in ['distance_to_center_km', 'metro_distance_km']:
        if c in df:
            df[c] = df[c].apply(convert_distance)
    # Категории: унификация + мода по региону
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.difference(['date_listed'])
    for c in cat_cols:
        if c in synonyms:
            df[c] = df[c].astype(str).str.lower().map(synonyms[c]).fillna(df[c])
        df[c] = df.groupby('region')[c] \
            .apply(lambda x: x.fillna(x.mode().iat[0] if not x.mode().empty else np.nan))
        df[c].fillna(df[c].mode().iat[0], inplace=True)
    # KNN для численных
    num = df.select_dtypes(include=[np.number]).columns.difference(['price_rub'])
    df[num] = KNNImputer(n_neighbors=5).fit_transform(df[num])
    return df


def remove_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    df = df[(df.get('area_sqm', 0) >= 10) & (df.get('price_rub', 0) <= 200e6)].copy()
    Q1, Q3 = df[numeric_cols].quantile(0.25), df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    mask_iqr = ~((df[numeric_cols] < (Q1 - 1.5 * IQR)) | (df[numeric_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
    z = np.abs((df[numeric_cols] - df[numeric_cols].mean()) / df[numeric_cols].std())
    mask_z = ~(z > 3).any(axis=1)
    return df[mask_iqr & mask_z].reset_index(drop=True)


def encode_and_scale(df: pd.DataFrame):
    df = df.copy()
    y = np.log1p(df['price_rub'].values)
    X = df.drop(columns=['price_rub', 'date_listed', 'link'], errors='ignore')
    # Категории
    cat_oh = ['region', 'house_type', 'material', 'layout', 'view']
    cat_ord = ['renovation_num']
    num = [c for c in X.columns if c not in cat_oh + cat_ord and X[c].dtype != 'object']
    # Ordinal
    ord_enc = OrdinalEncoder(categories=[['без', 'косметический', 'капитальный', 'дизайнерский']])
    # X['renovation_num'] = ord_enc.fit_transform(X[['renovation_num']])
    # # Log для skew
    # if 'income_level' in X:
    #     X['income_level'] = np.log1p(X['income_level'])
    # OneHot
    oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    oh_arr = oh.fit_transform(X[cat_oh])
    oh_cols = oh.get_feature_names_out(cat_oh)
    df_oh = pd.DataFrame(oh_arr, columns=oh_cols, index=X.index)
    # MinMax
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(X[num + cat_ord])
    # Собираем
    X_final = np.hstack([X_num, df_oh.values])

    return X_final, y


class RealEstateDataset(Dataset):
    """
    PyTorch Dataset для табличных данных недвижимости.
    Гарантированно возвращает FloatTensor входы и FloatTensor цели.
    """
    def __init__(self, X, y):
        # X: numpy.ndarray или torch.Tensor формы (N, D)
        # y: numpy.ndarray или torch.Tensor формы (N,) или (N,1)
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        else:
            X = X.float()
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).float().unsqueeze(1)
        else:
            y = y.float().unsqueeze(1) if y.dim()==1 else y.float()
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
