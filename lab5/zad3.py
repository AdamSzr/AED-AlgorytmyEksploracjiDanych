import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import pandas as pd


# Wczytaj dane treningowe i testowe
X_train = np.loadtxt('X_train.txt')
y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')

# Podziel dane na zestaw treningowy i zestaw testowy
X_train, X_test, y_train, y_test = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42)

# Przykładowe klasyfikatory (możesz wybrać inne)
classifier1 = DecisionTreeClassifier()
classifier2 = RandomForestClassifier()
classifier3 = SVC()

# Tworzenie klasyfikatora zespołowego z agregacją klasyfikatorów
bagging_classifier = BaggingClassifier(
    estimator=DecisionTreeClassifier(), n_estimators=3, random_state=42)


# Trenowanie klasyfikatora zespołowego na całym zbiorze treningowym
bagging_classifier.fit(X_train, y_train)

# Klasyfikacja na danych testowych
predictions = bagging_classifier.predict(X_test)

# Wypisz wynik
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy using bagging ensemble classifier: {accuracy}')

# -----------------------------------------------------------

# Kroswalidacja
cv_scores = cross_val_score(bagging_classifier, X_train, y_train, cv=5)

# Wypisz wyniki
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy using bagging ensemble classifier on test set: {accuracy}')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')


# ------------------------------------------------


# Oblicz ACC (dokładność)
accuracy = accuracy_score(y_test, predictions)

# Oblicz Recall
recall = recall_score(y_test, predictions, average='macro')

# Oblicz F1-score
f1 = f1_score(y_test, predictions, average='macro')

# Oblicz AUC
probs = bagging_classifier.predict_proba(
    X_test)[:, 1]  # Prawdopodobieństwo klasy pozytywnej
# Oblicz AUC
auc = roc_auc_score(y_test, probs, multi_class='ovr', average='weighted')


print(f'Accuracy (ACC): {accuracy}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'AUC: {auc}')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')


# ------------------------------------------------------

# Tworzenie DataFrame z wynikami
results = pd.DataFrame({
    'Metric': ['Accuracy (ACC)', 'Recall', 'F1 Score', 'AUC', 'Mean CV Score'],
    'Value': [accuracy, recall, f1, auc, cv_scores.mean()]
})

# Zapis wyników do pliku Excel
results.to_excel('aggregating.xlsx', index=False)
