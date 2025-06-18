# prepare_preprocessors.py

import os

import joblib
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler

from dataset import (
    load_data,
    collect_macro_features,
    merge_macro,
    impute_missing,
    remove_outliers,
)

DATA_PATH = 'artifacts/real_estate_8.csv'

df = load_data(DATA_PATH)

macro_dates = df['date_listed'].dt.to_pydatetime().tolist()
macro_df = collect_macro_features(macro_dates)

df = merge_macro(df, macro_df)
df = impute_missing(df)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df = remove_outliers(df, numeric_cols)

FEATURES = df.drop(columns=['price_rub', 'some_id', 'date_listed'])
y = df['price_rub'].values

cat_cols = FEATURES.select_dtypes(include=['object','bool','category']).columns.tolist()
num_cols = FEATURES.select_dtypes(include=[np.number]).columns.tolist()
if 'repair_level' in cat_cols:
    ord_enc = OrdinalEncoder(categories=[['none','cosmetic','capital','designer']])
    ord_enc.fit(FEATURES[['repair_level']])
    FEATURES['repair_level'] = ord_enc.transform(FEATURES[['repair_level']])
    cat_cols.remove('repair_level')
else:
    ord_enc = None

oh_enc = OneHotEncoder(sparse=False, handle_unknown='ignore')
oh_enc.fit(FEATURES[cat_cols])

scaler = MinMaxScaler()
scaler.fit(FEATURES[num_cols])

os.makedirs('models', exist_ok=True)

encoders_artifact = {
    'ordinal_encoder': ord_enc,
    'onehot_encoder': oh_enc,
    'cat_cols': cat_cols,
    'num_cols': num_cols
}
joblib.dump(encoders_artifact, 'models/label_encoders.pkl')

joblib.dump(scaler, 'models/scaler.pkl')
