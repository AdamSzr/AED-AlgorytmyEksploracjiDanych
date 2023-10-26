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


random_forest_classifier = RandomForestClassifier(n_estimators=3)


random_forest_classifier.fit(X_train, y_train)


y_pred_random_forest = random_forest_classifier.predict(X_test)


y_pred_random_forest_cv = cross_val_predict(
    random_forest_classifier, X_train, y_train, cv=5)

# # Oblicz AUC (Area Under the ROC Curve) w kroswalidacji
# ensemble_auc_cv = roc_auc_score(
#     y_train, y_pred_random_forest_cv, multi_class='ovr')

ensemble_auc_cv = roc_auc_score(y_train, random_forest_classifier.predict_proba(X_train), multi_class='ovr'),


print("AUC modelu zespołowego w kroswalidacji: " + str(ensemble_auc_cv))

