# Car Price Prediction

Проект предсказания стоимости подержанных автомобилей на основе технических характеристик и марки.  
Реализованы линейная и логистическая регрессии с ручной реализацией обучения и регуляризации.  


## Основные возможности

- Предсказание цены автомобиля в долларах (линейная регрессия)
- Классификация на 3 ценовых диапазона: дешёвый / средний / дорогой (логистическая регрессия)
- Полностью воспроизводимый pipeline: от загрузки "склеенного" CSV до предсказания
- Сравнение с `sklearn.linear_model.Ridge` — полное совпадение результатов
- Сохранение модели и функция предсказания для нового автомобиля

## Структура проекта
car-price-prediction/
├── data/ # исходные данные (data.csv)
├── notebooks/ # Jupyter-ноутбуки по шагам
│ ├── 01_exploratory_data_analysis.ipynb
│ ├── 02_feature_engineering.ipynb
│ └── 03_model_tuning.ipynb
├── src/ # переиспользуемый код
│ ├── init.py
│ ├── data.py
│ ├── features.py
│ ├── model.py
│ └── predict.py
├── models/ # сохранённые веса модели (model_weights.pkl)
├── reports/figures/ # графики (опционально)
├── README.md
├── requirements.txt
└── .gitignore

## Как запустить

1. Клонируйте репозиторий:
   ```bash
   git clone https://github.com/vasilijbereza123-maker/car-price-prediction.git
   cd car-price-prediction

2. Установите зависимости:
    '''bash
    - pip install -r requirements.txt

3. Откройте ноутбук:
    ```bash
    - jupyter notebook notebooks/03_model_tuning.ipynb

4. Используйте функцию predict_price() для нового автомобиля:
    - new_car = {
    'make': 'toyota',
    'model': 'rav4',
    'year': 2017,
    'engine_fuel_type': 'regular unleaded',
    'engine_hp': 176,
    'engine_cylinders': 4,
    'transmission_type': 'automatic',
    'driven_wheels': 'all wheel drive',
    'number_of_doors': 4,
    'market_category': 'crossover',
    'vehicle_size': 'midsize',
    'vehicle_style': '4dr suv',
    'highway_mpg': 28,
    'city_mpg': 22,
    'popularity': 2031
}
price = predict_price(new_car)
print(f"Предсказанная цена: ${price:,.0f}")

Результаты:
Линейная регрессия: RMSE = $40,564 на валидации
Логистическая регрессия: Accuracy = 75.5% (3 класса)
Лучший параметр регуляризации: r = 1e-05
Взаимодействия признаков не улучшили качество → исключены из финальной модели

Автор: @vasilijbereza123-maker
