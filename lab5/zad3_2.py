import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from scipy.stats import mode
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import os

base_path = os.getcwd()

# Wczytaj dane treningowe i testowe
X_train = np.loadtxt(base_path+'/lab5/X_train.txt')
y_train = np.loadtxt(
    base_path+'/lab5/y_train.txt')
X_test = np.loadtxt(
    base_path+'/lab5/X_test.txt')
y_test = np.loadtxt(
    base_path+'/lab5/y_test.txt')

# Konfiguracja zespołu
classifiers = [
    # RandomForestClassifier(n_estimators=10),
    # GradientBoostingClassifier(n_estimators=1),
    SVC(kernel='linear', C=1.0, cache_size=1536)
]

# Trenowanie każdego klasyfikatora na pełnym zbiorze treningowym
for classifier in classifiers:
    classifier.fit(X_train, y_train)

# Klasyfikacja na danych testowych
predictions = [classifier.predict(X_test) for classifier in classifiers]

# Ustalanie wyniku głosowania większościowego
ensemble_predictions = np.round(np.mean(predictions, axis=0))

# Obliczanie dokładności klasyfikacji na danych testowych
accuracy = accuracy_score(y_test, ensemble_predictions)
print("Dokładność na danych testowych:", accuracy)

# -----------------------------------------------------

# Obliczamy wypadkową ocenę na podstawie zasady większości głosów
ensemble_predictions, _ = mode(predictions, axis=0)

# Obliczanie dokładności klasyfikacji na danych testowych
accuracy = accuracy_score(y_test, ensemble_predictions)
print("Dokładność na danych testowych (z zasadą większości głosów):", accuracy)

# ------------------------------------------------------


# Kroswalidacja na danych treningowych
cv_scores = []
for classifier in classifiers:
    cv_score = cross_val_score(
        classifier, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(cv_score)

# Trenowanie każdego klasyfikatora na pełnym zbiorze treningowym
for classifier in classifiers:
    classifier.fit(X_train, y_train)

# Klasyfikacja na danych testowych
predictions = [classifier.predict(X_test) for classifier in classifiers]

# Ustalanie wyniku głosowania większościowego
ensemble_predictions = np.round(np.mean(predictions, axis=0))

# Obliczanie dokładności klasyfikacji na danych testowych
accuracy_test = accuracy_score(y_test, ensemble_predictions)

# Obliczanie średniej dokładności z kroswalidacji
average_cv_accuracy = np.mean(cv_scores, axis=1)

print("Wynik kroswalidacji (średnia dokładność):", average_cv_accuracy)
print("Dokładność na danych testowych:", accuracy_test)


# ---------------------------------


# Inicjalizacja list do przechowywania wyników
results = []

# Obliczanie metryk
for i, classifier in enumerate(classifiers):
    acc = accuracy_score(y_test, predictions[i])
    recall = recall_score(y_test, predictions[i], average='macro')
    f1 = f1_score(y_test, predictions[i], average='macro')
    auc = roc_auc_score(y_test, predictions[i], multi_class='ovr')

    results.append({'Classifier': f'Classifier {i + 1}',
                   'ACC': acc, 'Recall': recall, 'F1': f1})

# Tworzenie ramki danych z wynikami
results_df = pd.DataFrame(results)

# Zapis wyników do pliku Excel
results_df.to_csv('aggregating_results.csv', index=False)
