import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


import xgboost as xgb

# Wczytaj dane
X_train = np.loadtxt( 'X_train.txt')
y_train = np.loadtxt( 'y_train.txt')
X_test = np.loadtxt( 'X_test.txt')
y_test = np.loadtxt('y_test.txt')



# Inicjalizacja modelu ADABoost z domyślnymi parametrami
ada_boost_model = AdaBoostClassifier()

# Trenowanie modelu na danych treningowych
ada_boost_model.fit(X_train, y_train)

# Prognozowanie na danych testowych
y_pred = ada_boost_model.predict(X_test)

# Oblicz dokładność modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu ADABoost (z domyślnymi parametrami): {accuracy}')

# ---------------------------------------------

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Inicjalizacja modelu XGBoost z domyślnymi parametrami
xgb_model = xgb.XGBClassifier()

# Trenowanie modelu na danych treningowych
xgb_model.fit(X_train, y_train)

# Prognozowanie na danych testowych
y_pred = xgb_model.predict(X_test)

# Oblicz dokładność modelu
accuracy = accuracy_score(y_test, y_pred)
print(f'Dokładność modelu XGBoost (z domyślnymi parametrami): {accuracy}')