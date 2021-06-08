# ---Загрузка и подготовка данных + подготовка среды выполнения---

# Подключение необходимых библиотек
import sys
import pandas as pd
import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Загрузка набора данных (указывается название файла с набором данных)
try:
    dataset = pd.read_excel('dataset4.xlsx')
except:
    print("Ошибка: Указанный набор данных не обнаружен\n")
    sys.exit()

# Номера столбцов с данными,
# если таблица с данными содержит не только необходимые данные
# (начало с 0)
MFR_obs = 5
DD = 7
MFR_err = 8
DD_err = 10

# Номер строки таблицы, с которой начинаются непосредственно данные (начало с 0)
row_start = 3


# Извлечение данных (здесь необходимо выбрать, какие данные корректировать - MFR_err или DD_err)
try:
    X = dataset.iloc[row_start:, [MFR_obs, DD]].values.astype(float)
    y = dataset.iloc[row_start:, [MFR_err]].values.astype(float)
    # y = dataset.iloc[row_start:, [DD_err]].values.astype(float)
except:
    print("Ошибка: Указаны некорректные номера столбцов с данными\n")
    sys.exit()

# Разделение исходного набора данных на тренировочный и тестовый
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Нормализация данных
scaler_X = StandardScaler().fit(X_train)
scaler_y = StandardScaler().fit(y_train)
X_train_scaled = scaler_X.transform(X_train)
y_train_scaled = scaler_y.transform(y_train).ravel()
X_scaled = scaler_X.transform(X)


# Создаем модель преобразования
poly = PolynomialFeatures(3)

# Создаем новые признаки - многочлены до 3 степени включительно
X_train_extended = poly.fit_transform(X_train_scaled)
X_extended = poly.fit_transform(X_scaled)


# ---Подбор параметров и коррекция исходных данных---

# Подбор параметров
C_values = 2 ** np.linspace(-4, 7, num=31)
best_params = {'C': None}
mae = 100
k = y_train_scaled.size

for C in C_values:
    model = svm.LinearSVR(C=C, max_iter=10000000)

    # Кросс-валидация
    arr_mae = []

    for i in range(k):
        val_X = X_train_extended[i: i + 1]
        val_y = y_train_scaled[i: i + 1]

        partial_X_train = np.concatenate([X_train_extended[:i], X_train_extended[i + 1:]], axis=0)
        partial_y_train = np.concatenate([y_train_scaled[:i], y_train_scaled[i + 1:]], axis=0)
        
        model.fit(partial_X_train, partial_y_train)
        arr_mae.append(metrics.mean_absolute_error(val_y, model.predict(val_X)))
    
    mae_tmp = np.mean(arr_mae)

    if mae_tmp < mae:
        mae = mae_tmp
        best_params['C'] = C


# Создание модели с подобранными параметрами
model = svm.LinearSVR(C=best_params['C'], max_iter=10000000)
model.fit(X_train_extended, y_train_scaled)
y_predicted = scaler_y.inverse_transform(model.predict(X_extended))

# Вывод подобранных параметров и значения ошибки
print("Подобранные параметры:", best_params)
print("MAE:", metrics.mean_absolute_error(y, y_predicted), '\n')

# Вывод скорректированных данных
print("Скорректированные значения:")
for i in y_predicted:
    print(i)