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
# Конфигурация API-ключей и путей к файлам данных
# -----------------------------------------------------------
API_KEYS = {
    'CBR': os.getenv('CBR_API_KEY', ''),            # ключ для API Центробанка
    'MOSBIRZHA': os.getenv('MOSBIRZHA_API_KEY', ''),# ключ для MOEX
    'OIL': os.getenv('OIL_API_KEY', ''),            # ключ для API нефти
    'ROSSTAT': os.getenv('ROSSTAT_API_KEY', ''),    # ключ для Росстата
    'KEYRATE': os.getenv("KEYRATE_API_KEY"),        # ключ для ключевой ставки
    'API_CRED': os.getenv("PMI_KEY"),               # ключ для PMI и проч.
    'API_CRED_CPI': os.getenv("CPI_KEY")            # ключ для CPI
}

DATA_DIR     = 'data'                             # корневая папка для CSV
CIAN_CSV     = os.path.join(DATA_DIR, 'cian_listings.csv')
DOMKLIK_CSV  = os.path.join(DATA_DIR, 'domklik_listings.csv')
MACRO_CACHE  = os.path.join(DATA_DIR, 'macro_cache.csv')
PER_PAGE     = 150                                # страниц за раз для пагинации API

# -----------------------------------------------------------
# Утилиты: приведение расстояний и унификация категорий
# -----------------------------------------------------------
def convert_distance(val):
    """
    Приводит входные строки:
      - '10 мин' → километры по скорости 1 м/с
      - '800 м'  → километры
      - '1.2 км' → километры
    Возвращает NaN, если не распознано.
    """
    if pd.isna(val):
        return np.nan
    if isinstance(val, (int, float)):
        return val
    s = str(val).strip().lower()
    if 'мин' in s:
        # 1 мин ~ 60 м → 0.06 км
        num = float(s.replace('мин','').strip())
        return num * 0.06
    if 'м' in s and 'км' not in s:
        # метры → километры
        num = float(s.replace('м','').strip())
        return num / 1000
    if 'км' in s:
        return float(s.replace('км','').strip())
    try:
        return float(s)
    except:
        return np.nan

# Словари для выравнивания разных вариантов одного и того же значения
synonyms = {
    'material': {
        'панельный':'панель','панель':'панель',
        'монолитный':'монолит','монолит':'монолит',
        'кирпичный':'кирпич','кирпич':'кирпич',
        'деревянный':'дерево','дерево':'дерево',
        'газобетонный':'газобетон','газобетон':'газобетон',
        'керамзитобетонный':'керамзитобетон','керамзитобетон':'керамзитобетон',
        'брус':'брус'
    },
    'house_type': {'новостройка':'новостройка','вторичка':'вторичка','бу':'вторичка'},
    'renovation_num': {
        'без':'без','без ремонта':'без',
        'косметический':'косметический','косметический ремонт':'косметический',
        'капитальный':'капитальный','капитальный ремонт':'капитальный',
        'дизайнерский':'дизайнерский','дизайнерский ремонт':'дизайнерский'
    },
    'layout': {'изолированная':'изолированные','изолированные':'изолированные',
               'смежная':'смежные','смежные':'смежные'},
    'view': {'во двор':'во двор','на улицу':'на улицу','панорамный':'панорамный'}
}

# -----------------------------------------------------------
# Функции парсинга объявлений из ЦИАН и ДомКлик
# -----------------------------------------------------------
def fetch_cian_listings(base_url: str, pages: int = 5) -> pd.DataFrame:
    """
    Парсим ЦИАН с помощью Selenium:
      - открываем каждую страницу,
      - извлекаем цену, параметры (комнатность, этажи, площадь),
      - определяем тип жилья, район, расстояние до метро,
      - сохраняем в CSV и возвращаем DataFrame.
    """
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    records = []

    for p in range(1, pages+1):
        driver.get(base_url.format(page=p))
        time.sleep(3)  # даём JS загрузиться
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        cards = soup.select('div.c6e8ba5398--item')

        for c in cards:
            try:
                # 1) Цена: убираем нечисловые символы
                raw = c.select_one('span.c6e8ba5398--price').text
                price = float(raw.replace('\u2009','').replace('₽','').replace(' ',''))

                # 2) Параметры: комнаты, площадь, этаж/этажность
                params = c.select_one('div.c6e8ba5398--params').text.strip()
                rooms = int(re.search(r'(\d+)[-\s]комн', params).group(1)) \
                        if re.search(r'(\d+)[-\s]комн', params) else None
                area_sqm = float(re.search(r'([\d\.]+)\s*м²', params).group(1)) \
                           if re.search(r'([\d\.]+)\s*м²', params) else None
                fm = re.search(r'(\d+)\s*/\s*(\d+)\s*эт', params)
                floor, total_floors = (int(fm.group(1)), int(fm.group(2))) if fm else (None,None)

                # 3) Тип жилья: новостройка или вторичка
                house_type = 'новостройка' if 'новостройка' in params.lower() else 'вторичка'

                # 4) Район
                district_elem = c.select_one('a.c6e8ba5398--region')
                district = district_elem.text.strip() if district_elem else None

                # 5) Расстояние до метро
                metro_elem = c.select_one('div.c6e8ba5398--geo-icon ~ div')
                metro_distance_km = convert_distance(metro_elem.text.strip()) if metro_elem else None

                # 6) Ссылка на объект
                link_elem = c.select_one('a.c6e8ba5398--link')
                link = link_elem['href'] if link_elem else None

                # 7) Собираем все поля
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
                # Пропускаем карточки с неожиданным форматом
                continue

    driver.quit()
    df = pd.DataFrame(records)
    df.to_csv(CIAN_CSV, index=False)
    return df

def fetch_domklik_listings(api_url: str) -> pd.DataFrame:
    """
    Запрос к API ДомКлик, где JSON→DataFrame.
    Сохраняет area, rooms, год постройки и дату листинга.
    """
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
# Функции для сбора макроэкономических данных через разные API
# -----------------------------------------------------------
def fetch_fx_rate(pair: str, date: datetime) -> float:
    """
    Курс валют: GET от exchangeratesapi.io
    pair = 'USD_RUB' → base='RUB', symbols='USD'
    """
    url = f'https://api.exchangeratesapi.io/{date.strftime("%Y-%m-%d")}'
    base, sym = pair.split('_')[1], pair.split('_')[0]
    resp = requests.get(url, params={'symbols': sym, 'base': base})
    resp.raise_for_status()
    return resp.json()['rates'][sym]

def fetch_oil_price(date: datetime) -> float:
    """
    Цена нефти Brent через OilPriceAPI.
    """
    url = 'https://api.oilpriceapi.com/v1/historical/spot'
    resp = requests.get(url,
                        params={'date': date.strftime('%Y-%m-%d'),'symbol':'BRENT'},
                        headers={'Authorization': f'Token {API_KEYS["OIL"]}'})
    resp.raise_for_status()
    return float(resp.json()['data']['price'])

def fetch_mos_index(date: datetime) -> float:
    """
    MOEX INDEX: парсинг JSON с сайта Мосбиржи.
    """
    url = 'https://iss.moex.com/iss/engines/stock/markets/index/boards/SIS/securities.json'
    resp = requests.get(url, params={'date': date.strftime('%Y-%m-%d')})
    resp.raise_for_status()
    data = resp.json()['marketdata']['data']
    # столбец 'LAST' содержит последний торгованный курс
    return float(data[0][data[0].index('LAST')])

def fetch_indicator(name: str, date: datetime) -> float:
    """
    Общая функция для индикаторов TradingEconomics:
    name — URL-код индикатора, например 'manufacturing%20pmi'.
    Возвращает последнее значение on_or_before date.
    """
    url = f"https://api.tradingeconomics.com/historical/country/russia/indicator/{name}"
    resp = requests.get(f"{url}?c={API_KEYS['API_CRED']}&f=json")
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    # Оставляем даты ≤ requested
    df = df[df['DateTime'].dt.date <= date.date()]
    if df.empty:
        raise KeyError(f"No {name} on or before {date.date()}")
    # Берём самое свежее
    return float(df.sort_values('DateTime',ascending=False).iloc[0]['Close'])

# Варианты индикаторов
fetch_pmi                = lambda d: fetch_indicator("manufacturing%20pmi", d)
fetch_cpi                = lambda d: fetch_indicator("consumer%20price%20index", d)
fetch_unemployment_rate  = lambda d: fetch_indicator("unemployment%20rate", d)
fetch_gdp_growth         = lambda d: fetch_indicator("gdp%20growth", d)
fetch_household_debt_to_gdp = lambda d: fetch_indicator("households%20debt%20to%20gdp", d)
fetch_mortgage_rate      = lambda d: fetch_indicator("mortgage%20rate", d)

def collect_macro_features(dates: List[datetime]) -> pd.DataFrame:
    """
    Сбор макрофич: для каждого уникального date_listed
    делает запросы к API, собирает все в один DataFrame,
    сохраняет в CSV-кэш и возвращает.
    """
    recs = []
    for dt in sorted(set(dates)):
        try:
            recs.append({
                'date': dt,
                'dollar_rate':   fetch_fx_rate('USD_RUB', dt),
                'euro_rate':     fetch_fx_rate('EUR_RUB', dt),
                'oil_price':     fetch_oil_price(dt),
                'moex_index':    fetch_mos_index(dt),
                'pmi':           fetch_pmi(dt),
                'cpi':           fetch_cpi(dt),
                'unemployment':  fetch_unemployment_rate(dt),
                'gdp_growth':    fetch_gdp_growth(dt),
                'mortgage_rate': fetch_mortgage_rate(dt),
                'households_debt_to_gdp': fetch_household_debt_to_gdp(dt)
            })
        except Exception as e:
            print(f"Macro fetch fail {dt}: {e}")
    df = pd.DataFrame(recs).set_index('date')
    df.to_csv(MACRO_CACHE)
    return df

# -----------------------------------------------------------
# Сбор и объединение сырых данных
# -----------------------------------------------------------
def load_aggregator_data() -> pd.DataFrame:
    """
    Если CSV нет на диске — парсим заново,
    затем читаем и объединяем ЦИАН + ДомКлик.
    """
    if not os.path.exists(CIAN_CSV):
        fetch_cian_listings('https://www.cian.ru/cat.php?deal_type=sale&page={page}',5)
    if not os.path.exists(DOMKLIK_CSV):
        fetch_domklik_listings('https://api.domclick.ru/v1/listings')
    df1 = pd.read_csv(CIAN_CSV, parse_dates=['date_listed'])
    df2 = pd.read_csv(DOMKLIK_CSV, parse_dates=['date_listed'])
    return pd.concat([df1,df2], ignore_index=True)

def load_data(path: str=None) -> pd.DataFrame:
    """
    Если path=None — грузим агрегированные данные,
    иначе читаем по переданному пути.
    """
    return load_aggregator_data() if path is None else pd.read_csv(path, parse_dates=['date_listed'])

def merge_macro(df: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """ Left-join по дате листинга и макрофичам """
    return df.merge(macro, how='left', left_on='date_listed', right_index=True)

# -----------------------------------------------------------
# Предобработка: пропуски, выбросы, кодирование и масштабирование
# -----------------------------------------------------------
def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    1) Удаляет строки с >30% пропусков.
    2) Конвертирует колонки расстояний.
    3) Унифицирует категории синонимами + заполняет мода по региону, затем глобально.
    4) Заполняет числовые пропуски KNNImputer(k=5).
    """
    thresh = int(0.7 * df.shape[1])
    df = df.dropna(thresh=thresh).copy()

    # Приводим дистанции к км
    for c in ['distance_to_center_km','metro_distance_km']:
        if c in df:
            df[c] = df[c].apply(convert_distance)

    # Категории
    cat_cols = df.select_dtypes(include=['object','category']).columns.difference(['date_listed'])
    for c in cat_cols:
        if c in synonyms:
            df[c] = df[c].astype(str).str.lower().map(synonyms[c]).fillna(df[c])
        # по региону сначала
        df[c] = df.groupby('region')[c]\
                 .apply(lambda x: x.fillna(x.mode().iat[0] if not x.mode().empty else np.nan))
        # потом глобально
        df[c].fillna(df[c].mode().iat[0], inplace=True)

    # Числовые KNN
    num = df.select_dtypes(include=[np.number]).columns.difference(['price_rub'])
    df[num] = KNNImputer(n_neighbors=5).fit_transform(df[num])

    return df

def remove_outliers(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    1) Убирает аномалии: area<10 или price>200e6.
    2) Фильтрует по IQR и Z-оценке (>3σ).
    """
    df = df[(df.get('area_sqm',0)>=10) & (df.get('price_rub',0)<=200e6)].copy()
    Q1,Q3 = df[numeric_cols].quantile(0.25), df[numeric_cols].quantile(0.75)
    IQR = Q3-Q1
    mask_iqr = ~((df[numeric_cols]<(Q1-1.5*IQR))|(df[numeric_cols]>(Q3+1.5*IQR))).any(axis=1)
    z = np.abs((df[numeric_cols]-df[numeric_cols].mean())/df[numeric_cols].std())
    mask_z   = ~(z>3).any(axis=1)
    return df[mask_iqr & mask_z].reset_index(drop=True)

def encode_and_scale(df: pd.DataFrame):
    """
    1) Лог1p таргета.
    2)
    OrdinalEncoder для renovation_num.
    3) OneHotEncoder для остальных категорий.
    4) MinMaxScaler для всех числовых.
    Возвращает X_final и y.
    """
    df = df.copy()
    y = np.log1p(df['price_rub'].values)
    X = df.drop(columns=['price_rub','date_listed','link'], errors='ignore')

    # разделяем колонки
    cat_oh  = ['region','house_type','material','layout','view']
    cat_ord = ['renovation_num']
    num     = [c for c in X.columns if c not in cat_oh+cat_ord and X[c].dtype!='object']

    # ordinal для ремонта
    ord_enc = OrdinalEncoder(categories=[['без','косметический','капитальный','дизайнерский']])
    X['renovation_num'] = ord_enc.fit_transform(X[['renovation_num']])

    # one-hot
    oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    oh_arr  = oh.fit_transform(X[cat_oh])
    oh_cols = oh.get_feature_names_out(cat_oh)
    df_oh   = pd.DataFrame(oh_arr, columns=oh_cols, index=X.index)

    # min-max
    scaler = MinMaxScaler()
    X_num = scaler.fit_transform(X[num+cat_ord])

    # объединяем все признаки
    X_final = np.hstack([X_num, df_oh.values])
    return X_final, y

class RealEstateDataset(Dataset):
    """
    PyTorch Dataset для табличных данных.
    Конвертирует X, y в FloatTensor и возвращает кортеж (X, y).
    """
    def __init__(self, X, y):
        # если numpy → в Tensor
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X).float()
        else:
            X = X.float()
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).float().unsqueeze(1)  # shape (N,1)
        else:
            y = y.float().unsqueeze(1) if y.dim()==1 else y.float()
        self.X = X
        self.y = y

    def __len__(self):
        # число образцов
        return self.X.size(0)

    def __getitem__(self, idx):
        # возвращает один образец
        return self.X[idx], self.y[idx]
