import pickle
import numpy as np
import pandas as pd
from .features import prepare_X

def predict_price(car_dict, model_path='../models/model_weights.pkl'):
    """
    Предсказывает цену автомобиля по его характеристикам.
    """
    # Загружаем веса модели
    with open(model_path, 'rb') as f:
        weights = pickle.load(f)
    w0, w = weights['w0'], weights['w']
    
    # Создаём датафрейм из словаря
    df_input = pd.DataFrame([car_dict])
    
    # Подготавливаем признаки
    X_input = prepare_X(df_input)
    
    # Предсказание в лог-масштабе
    log_price_pred = w0 + X_input.dot(w)
    
    # Обратное преобразование в доллары
    price_pred = np.expm1(log_price_pred[0])
    
    return price_pred  