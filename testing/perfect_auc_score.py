from sklearn.metrics import roc_auc_score

# Przykładowe rzeczywiste etykiety (ground truth) i wyniki klasyfikacji modelu
y_true = [0, 1, 1, 0, 1, 1, 0, 0]
y_scores = [0.2, 0.7, 0.8, 0.4, 0.6, 0.9, 0.3, 0.5]

# Oblicz ROC AUC
roc_auc = roc_auc_score(y_true, y_scores)

# Wyświetl wynik
print(f'ROC AUC Score: {roc_auc}')
