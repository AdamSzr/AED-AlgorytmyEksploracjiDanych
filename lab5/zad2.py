from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
import pandas as pd


# Wczytaj dane treningowe i testowe
X_train = np.loadtxt('Train/X_train.txt')
y_train = np.loadtxt('Train/y_train.txt')
X_test = np.loadtxt('Test/X_test.txt')
y_test = np.loadtxt('Test/y_test.txt')

# Przykładowe klasyfikatory (możesz wybrać inne)
classifier1 = DecisionTreeClassifier()
classifier2 = RandomForestClassifier()
classifier3 = SVC()

# Tworzenie klasyfikatora zespołowego
ensemble_classifier = VotingClassifier(estimators=[
    ('decision_tree', classifier1),
    ('random_forest', classifier2),
    ('svm', classifier3)
], voting='hard')  # Ustawienie 'hard' oznacza zasadę większości głosów


# Kroswalidacja
cv_scores = cross_val_score(ensemble_classifier, X_train, y_train, cv=5)

# Trenowanie na całym zbiorze treningowym
ensemble_classifier.fit(X_train, y_train)

# Klasyfikacja na danych testowych
predictions = ensemble_classifier.predict(X_test)

# Oblicz ACC (dokładność)
accuracy = accuracy_score(y_test, predictions)

# Oblicz Recall
recall = recall_score(y_test, predictions)

# Oblicz F1-score
f1 = f1_score(y_test, predictions)

# Oblicz AUC
# Dla obliczenia AUC musisz uzyskać prawdopodobieństwa przynależności do klasy pozytywnej zamiast przewidywanych etykiet
probs = ensemble_classifier.predict_proba(
    X_test)[:, 1]  # Prawdopodobieństwo klasy pozytywnej
auc = roc_auc_score(y_test, probs)

print(f'Accuracy using ensemble classifier on test set: {accuracy}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'AUC: {auc}')
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean cross-validation score: {cv_scores.mean()}')

# Tworzenie DataFrame z wynikami
results = pd.DataFrame({
    'Metric': ['Accuracy', 'Recall', 'F1 Score', 'AUC', 'Mean CV Score'],
    'Value': [accuracy, recall, f1, auc, cv_scores.mean()]
})

# Zapis wyników do pliku Excel
results.to_excel('ensemble_learning.xlsx', index=False)
