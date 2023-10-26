from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import os
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pandas as pd

base_path = os.getcwd()

# Załadowanie danych
X_train = np.loadtxt(base_path + '/X_train.txt')
y_train = np.loadtxt(base_path + '/y_train.txt')
X_test = np.loadtxt(base_path + '/X_test.txt')
y_test = np.loadtxt(base_path + '/y_test.txt')

# Definicja klasyfikatorów
rf_clf = RandomForestClassifier(random_state=42, n_estimators=3,max_depth=1)
ada_clf = AdaBoostClassifier(random_state=42, n_estimators=3)
gb_clf = GradientBoostingClassifier(random_state=42, n_estimators=3, max_depth=1)
bagging_clf = BaggingClassifier(estimator=rf_clf, n_estimators=3, random_state=4)

# Konfiguracja głosowania
ensemble_clf = VotingClassifier(estimators=[
    ('Random Forest', rf_clf),
    ('AdaBoost', ada_clf),
    ('Gradient Boosting', gb_clf),
    ('Bagging', bagging_clf)
], voting='hard')  # 'hard' oznacza głosowanie większościowe

# Trenowanie modelu ensemble
ensemble_clf.fit(X_train, y_train)

# Przewidywanie na zbiorze testowym
y_pred = ensemble_clf.predict(X_test)

# Obliczanie dokładności
accuracy = accuracy_score(y_test, y_pred)
print("Dokładność modelu ensemble: ", accuracy)

# Obliczanie Recall (czułość)
recall = recall_score(y_test, y_pred, average='weighted')
print("Recall (Czułość):", recall)

# Obliczanie F1
f1 = f1_score(y_test, y_pred, average='weighted')
print("F1:", f1)

# Wynik kroswalidacji
cross_val_scores = cross_val_score(ensemble_clf, X_train, y_train, cv=5)
print("Wyniki kroswalidacji (5 podprób):", cross_val_scores)

# Tworzenie ramki danych z wynikami
results = pd.DataFrame({
    'Metric': ['ACC (Dokładność)', 'Recall (Czułość)', 'F1'],
    'Value': [accuracy, recall, f1]
})

# Zapisanie wyników do pliku Excel (xlsx)
results.to_excel('aggregating.xlsx', index=False)