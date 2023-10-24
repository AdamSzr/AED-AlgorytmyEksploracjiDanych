import numpy as np
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
import os
import pandas as pd

# Wczytaj dane treningowe i testowe
X_train = np.loadtxt(os.getcwd()+'/X_train.txt')
y_train = np.loadtxt(os.getcwd()+'/y_train.txt')
X_test = np.loadtxt(os.getcwd()+'/X_test.txt')
y_test = np.loadtxt(os.getcwd()+'/y_test.txt')

# Stwórz klasyfikatory
bagging_classifier = BaggingClassifier(
    estimator=GaussianNB(), n_estimators=3)
random_forest_classifier = RandomForestClassifier(n_estimators=3)
gb_classifier = GradientBoostingClassifier(max_depth=1, n_estimators=5)

# Trenuj każdy klasyfikator na całym zbiorze treningowym
bagging_classifier.fit(X_train, y_train)
random_forest_classifier.fit(X_train, y_train)
gb_classifier.fit(X_train, y_train)

# Przeprowadź prognozowanie na zbiorze testowym
y_pred_bagging = bagging_classifier.predict(X_test)
y_pred_random_forest = random_forest_classifier.predict(X_test)
y_pred_bayes_optimal = gb_classifier.predict(X_test)

# --------------------------------------------------------------------------------------

# Możesz wykonać głosowanie większościowe (np. dla klasyfikacji binarnej)
y_ensemble = (y_pred_bagging + y_pred_random_forest +
              y_pred_bayes_optimal >= 2).astype(int)

# Oblicz dokładność
ensemble_accuracy = accuracy_score(y_test, y_ensemble)
print("Dokładność modelu zespołowego: {:.2f}".format(ensemble_accuracy))


# --------------------------------------------------------------------------------------

# Przeprowadź kroswalidację dla 5 podprób
y_pred_bagging_cv = cross_val_predict(
    bagging_classifier, X_train, y_train, cv=5)
y_pred_random_forest_cv = cross_val_predict(
    random_forest_classifier, X_train, y_train, cv=5)
y_pred_bayes_optimal_cv = cross_val_predict(
    gb_classifier, X_train, y_train, cv=5)

# Oblicz wypadkową ocenę stosując zasadę większości głosów na podstawie wyników kroswalidacji
ensemble_result_cv = (y_pred_bagging_cv + y_pred_random_forest_cv +
                      y_pred_bayes_optimal_cv >= 2).astype(int)

#########################

# Oblicz dokładność wypadkowej oceny w kroswalidacji
ensemble_accuracy_cv = accuracy_score(y_train, ensemble_result_cv)

# Oblicz czułość (Recall) w kroswalidacji
ensemble_recall_cv = recall_score(
    y_train, ensemble_result_cv, average='weighted')

# Oblicz F1 Score w kroswalidacji
ensemble_f1_score_cv = f1_score(
    y_train, ensemble_result_cv, average='weighted')

# # Oblicz AUC (Area Under the ROC Curve) w kroswalidacji
# ensemble_auc_cv = roc_auc_score(
#     y_train, ensemble_result_cv)

print("Dokładność modelu zespołowego w kroswalidacji (ACC): {:.2f}".format(
    ensemble_accuracy_cv))
print("Czułość modelu zespołowego w kroswalidacji (Recall): {:.2f}".format(
    ensemble_recall_cv))
print("F1 Score modelu zespołowego w kroswalidacji: {:.2f}".format(
    ensemble_f1_score_cv))



# print("AUC modelu zespołowego w kroswalidacji: {:.2f}".format(ensemble_auc_cv))


##########################

# Przeprowadź prognozowanie na zbiorze testowym
y_pred_bagging_test = bagging_classifier.predict(X_test)
y_pred_random_forest_test = random_forest_classifier.predict(X_test)
y_pred_bayes_optimal_test = gb_classifier.predict(X_test)

# Oblicz wypadkową ocenę stosując zasadę większości głosów na podstawie wyników na zbiorze testowym
ensemble_result_test = (y_pred_bagging_test + y_pred_random_forest_test +
                        y_pred_bayes_optimal_test >= 2).astype(int)


##########################

# Oblicz dokładność wypadkowej oceny na zbiorze testowym
ensemble_accuracy_test = accuracy_score(y_test, ensemble_result_test)

# Oblicz czułość (Recall) na zbiorze testowym
ensemble_recall_test = recall_score(
    y_test, ensemble_result_test, average='weighted')

# Oblicz F1 Score na zbiorze testowym
ensemble_f1_score_test = f1_score(
    y_test, ensemble_result_test, average='weighted')

# # Oblicz AUC (Area Under the ROC Curve) na zbiorze testowym
# ensemble_auc_test = roc_auc_score(
#     y_test, ensemble_result_test)

##########################

print("Dokładność modelu zespołowego na zbiorze testowym (ACC): {:.2f}".format(
    ensemble_accuracy_test))
print("Czułość modelu zespołowego na zbiorze testowym (Recall): {:.2f}".format(
    ensemble_recall_test))
print("F1 Score modelu zespołowego na zbiorze testowym: {:.2f}".format(
    ensemble_f1_score_test))
# print("AUC modelu zespołowego na zbiorze testowym: {:.2f}".format(
#     ensemble_auc_test))
# ---------------------------------------------------------------------------------------


# Tworzenie DataFrame
data = {
    'Metryka': ['Accuracy', 'Recall', 'F1 Score', 'AUC'],
    'Kroswalidacja': [ensemble_accuracy_cv, ensemble_recall_cv, ensemble_f1_score_cv, 'error'],
    'Testowanie': [ensemble_accuracy_test, ensemble_recall_test, ensemble_f1_score_test, 'error']
}

df = pd.DataFrame(data)

# Zapis do pliku CSV (CSV)
df.to_excel('ensemble_learning.xls', index=False)
