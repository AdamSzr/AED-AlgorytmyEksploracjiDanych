import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score

# Wczytaj dane
X_train = np.loadtxt('X_train.txt')
y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')

# Inicjalizacja klasyfikatorów
rf_classifier = RandomForestClassifier(max_depth=3, n_estimators=100)
gb_classifier = GradientBoostingClassifier(max_depth=1, n_estimators=5)
bagging_classifier = BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=5, random_state=42)

print('Inicjalizacja klasyfikatorów -> DONE')

# Uczymy modele
rf_classifier.fit(X_train, y_train)
print('Uczymy modele - rf -> DONE')
gb_classifier.fit(X_train, y_train)
print('Uczymy modele - gb -> DONE')
bagging_classifier.fit(X_train, y_train)
print('Uczymy modele - bagging -> DONE')

# Przewidujemy na zbiorze testowym
rf_predictions = rf_classifier.predict(X_test)
gb_predictions = gb_classifier.predict(X_test)
bagging_predictions = bagging_classifier.predict(X_test)
print('Przewidujemy na zbiorze testowym-> DONE')

# Oblicz metryki
rf_acc = accuracy_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions, average='weighted')
rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
# rf_auc = roc_auc_score(y_test, rf_classifier.predict_proba(X_test)[:, 1])
print('Oblicz metryki 1.-> DONE')

gb_acc = accuracy_score(y_test, gb_predictions)
gb_recall = recall_score(y_test, gb_predictions, average='weighted')
gb_f1 = f1_score(y_test, gb_predictions, average='weighted')
# gb_auc = roc_auc_score(y_test, gb_classifier.predict_proba(X_test)[:, 1])
print('Oblicz metryki 2.-> DONE')

bagging_acc = accuracy_score(y_test, bagging_predictions)
bagging_recall = recall_score(y_test, bagging_predictions, average='weighted')
bagging_f1 = f1_score(y_test, bagging_predictions, average='weighted')
# bagging_auc = roc_auc_score(y_test, bagging_classifier.predict_proba(X_test)[:, 1])
print('Oblicz metryki 3.-> DONE')


# Tworzymy tabelę wyników
results = pd.DataFrame({
    'Method': ['Random Forest', 'Gradient Boosting', 'Bagging'],
    'ACC': [rf_acc, gb_acc, bagging_acc],
    'Recall': [rf_recall, gb_recall, bagging_recall],
    'F1': [rf_f1, gb_f1, bagging_f1],
    # 'AUC': [rf_auc, gb_auc, bagging_auc]
})
print('Tworzymy tabelę wyników-> DONE')

# Wyświetlamy wyniki
print(results)


# Zapisanie wyników do pliku Excel (xlsx)
results.to_excel('comparison.xlsx', index=False)
