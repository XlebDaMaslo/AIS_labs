import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

X = train_data.drop('SalePrice', axis=1)
y = train_data['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

cat_features = [col for col in X_train.columns if X_train[col].dtype == 'object']

# Замена NaN на строку "NaN"
for col in cat_features:
    X_train[col] = X_train[col].astype(str).fillna("NaN")
    X_val[col] = X_val[col].astype(str).fillna("NaN")
    test_data[col] = test_data[col].astype(str).fillna("NaN")

# Pool для CatBoost
train_pool = Pool(X_train, y_train, cat_features=cat_features)
val_pool = Pool(X_val, y_val, cat_features=cat_features)

# Инициализация модели CatBoost
model = CatBoostRegressor(
    iterations=2000,  # Количество итераций
    learning_rate=0.1,  # Скорость обучения
    depth=6,  # Глубина деревьев
    loss_function='RMSE',  # Функция потерь
    eval_metric='RMSE',  # Метрика для оценки
    cat_features=cat_features,  # Категориальные признаки
    verbose=100,  # Вывод информации каждые 100 итераций
    random_seed=42  # Seed
)

# Обучение модели
model.fit(train_pool, eval_set=val_pool, use_best_model=True)

# Прогнозирование на валидационном наборе
y_val_pred = model.predict(val_pool)

# Вычисление RMSE на валидационном наборе
val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
print(f'Validation RMSE: {val_rmse}')

relative_rmse = val_rmse / y_train.mean()
print(f'Relative RMSE: {relative_rmse}')

# Прогнозирование на тестовом наборе
test_pool = Pool(test_data, cat_features=cat_features)
y_test_pred = model.predict(test_pool)

# Сохранение прогнозов
submission = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': y_test_pred})
submission.to_csv('submission.csv', index=False)
