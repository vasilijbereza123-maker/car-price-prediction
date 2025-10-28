import pandas as pd
import numpy as np

def prepare_X(df):
    df = df.copy()
    base = ['engine_hp', 'engine_cylinders', 'highway_mpg', 'city_mpg', 'popularity']
    features = base.copy()
    
    # Возраст
    df['age'] = 2017 - df['year']
    features.append('age')
    
    # Двери
    for v in [2, 3, 4]:
        feature = f'num_doors_{v}'
        df[feature] = (df['number_of_doors'] == v).astype(int)
        features.append(feature)
    
    # Топ-5 марок
    top_makes = ['ford', 'chevrolet', 'toyota', 'nissan', 'honda']
    for make in top_makes:
        feature = f'is_make_{make}'
        df[feature] = (df['make'] == make).astype(int)
        features.append(feature)
    
    # Тип топлива
    fuel_types = ['regular unleaded', 'premium unleaded (required)', 'premium unleaded (recommended)', 'flex-fuel (unleaded/e85)', 'diesel']
    for fuel in fuel_types:
        clean_name = fuel.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
        feature = f'is_fuel_{clean_name}'
        df[feature] = (df['engine_fuel_type'] == fuel).astype(int)
        features.append(feature)
    
    # Трансмиссия
    transmissions = ['automatic', 'manual', 'automated_manual']
    for trans in transmissions:
        feature = f'is_trans_{trans}'
        df[feature] = (df['transmission_type'] == trans).astype(int)
        features.append(feature)
    
    # Привод
    drives = ['front wheel drive', 'rear wheel drive', 'all wheel drive', 'four wheel drive']
    for drive in drives:
        clean_drive = drive.replace(' ', '_')
        feature = f'is_drive_{clean_drive}'
        df[feature] = (df['driven_wheels'] == drive).astype(int)
        features.append(feature)
    
    # Размер
    sizes = ['compact', 'midsize', 'large']
    for size in sizes:
        feature = f'is_size_{size}'
        df[feature] = (df['vehicle_size'] == size).astype(int)
        features.append(feature)
    
    # Стиль кузова
    styles = ['sedan', '4dr_suv', 'crew_cab_pickup', 'coupe', '4dr_hatchback']
    for style in styles:
        feature = f'is_style_{style}'
        df[feature] = (df['vehicle_style'] == style).astype(int)
        features.append(feature)
    
    df_num = df[features].fillna(0)
    return df_num.values