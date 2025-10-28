import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split

def load_and_fix_data(data_path):
    """Загружает и исправляет "склеенный" CSV."""
    with open(data_path, 'r', encoding='utf-8') as f:
        raw = f.read().strip()
    fixed = re.sub(r'(\d)([A-Z])', r'\1\n\2', raw)
    return pd.read_csv(pd.compat.StringIO(fixed), header=None)

def preprocess_data(df):
    """Преобразует колонки, обрабатывает пропуски, создаёт log_msrp."""
    columns = [
        'make', 'model', 'year', 'engine_fuel_type', 'engine_hp', 'engine_cylinders',
        'transmission_type', 'driven_wheels', 'number_of_doors', 'market_category',
        'vehicle_size', 'vehicle_style', 'highway_mpg', 'city_mpg', 'popularity', 'msrp'
    ]
    df.columns = columns

    # Числовые колонки
    numeric_cols = ['year', 'engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity', 'msrp', 'number_of_doors']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Строковые колонки
    str_cols = df.select_dtypes(include='object').columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.lower().str.replace(' ', '_')

    # Удаляем строки с некорректной ценой
    df = df.dropna(subset=['msrp'])
    df['log_msrp'] = np.log1p(df['msrp'])

    # Заполняем пропуски
    df['engine_hp'] = df['engine_hp'].fillna(df['engine_hp'].median())
    df['engine_cylinders'] = df['engine_cylinders'].fillna(df['engine_cylinders'].median())
    df['number_of_doors'] = df['number_of_doors'].fillna(df['number_of_doors'].median())
    df['engine_fuel_type'] = df['engine_fuel_type'].fillna(df['engine_fuel_type'].mode()[0])
    df['market_category'] = df['market_category'].fillna('unknown')

    return df

def split_data(df, test_size=0.2, val_size=0.25, random_state=2):
    """Разделяет данные на train/val/test."""
    df_full_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state)
    df_train, df_val = train_test_split(df_full_train, test_size=val_size, random_state=random_state)
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True), df_test.reset_index(drop=True)