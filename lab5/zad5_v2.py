import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier, ExtraTreesClassifier
import lightgbm
import catboost
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

# Wczytaj dane
X_train = np.loadtxt('X_train.txt')
y_train = np.loadtxt('y_train.txt')
X_test = np.loadtxt('X_test.txt')
y_test = np.loadtxt('y_test.txt')

# Przygotuj modele
rf_clf = RandomForestClassifier(random_state=42, n_estimators=20)
ada_clf = AdaBoostClassifier(random_state=42, n_estimators=20)
gb_clf = GradientBoostingClassifier(random_state=42, n_estimators=20)
bagging_clf = BaggingClassifier(base_estimator=rf_clf, n_estimators=20, random_state=42)
extra_trees_clf = ExtraTreesClassifier(random_state=42, n_estimators=20)
lgbm_clf = lightgbm.LGBMClassifier(random_state=42, n_estimators=20)
catboost_clf = catboost.CatBoostClassifier(random_state=42, verbose=0, n_estimators=20)
hist_gb_clf = HistGradientBoostingClassifier(random_state=42)

# Zdefiniuj funkcję do oceny modelu i zwrócenia wyników
def evaluate_model(clf, X_train, y_train, X_test, y_test):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred, )
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return accuracy, recall, f1

# Ocena wyników dla różnych modeli
models = [rf_clf, ada_clf, gb_clf, bagging_clf, extra_trees_clf, lgbm_clf, catboost_clf, hist_gb_clf]
results = {'Method': [], 'ACC': [], 'Recall': [], 'F1': []}

for model in models:
    accuracy, recall, f1 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results['Method'].append(model.__class__.__name__)
    results['ACC'].append(accuracy)
    results['Recall'].append(recall)
    results['F1'].append(f1)

# Tworzenie ramki danych z wynikami
results_df = pd.DataFrame(results)

# Zapisanie wyników do pliku Excel (xlsx)
results_df.to_excel('comparison.xlsx', index=False)
