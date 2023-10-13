import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Wczytaj dane z pliku CSV do DataFrame
data = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')

# Przetwórz dane, usuń brakujące wartości
data = data.dropna()

values = []

# Utwórz pętlę od 0 do 51 (włącznie)
for i in range(52):
    value = f'Normalized {i}'
    # value = f'W{i}'
    values.append(value)

print(values)
item1 = values[random.randint(0, 51)]
item2 = values[5]

print(item1, item2)
features = data[values].copy()  # Stwórz kopię danych

# Wybierz parametry DBSCAN
eps = 0.2  # Promień sąsiedztwa
min_samples = 5  # Minimalna liczba próbek w sąsiedztwie

# Przeprowadź analizę klastrowania DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(features)

# Przygotuj etykiety klastrów w formie kategorii całkowitych
unique_labels = set(dbscan.labels_)
print(unique_labels)
cluster_labels = {}
for label in unique_labels:
    if label == -1:
        cluster_labels[label] = 'Noise'
    else:
        cluster_labels[label] = len(cluster_labels) + 1

# Dodaj kategorie do DataFrame
features['Cluster'] = [cluster_labels[label] for label in dbscan.labels_]

# Wykres z oznaczeniem klastrów
sns.scatterplot(x=features[item1], y=features[item2],
                hue=features['Cluster'], style=features['Cluster'], palette='viridis')
plt.title('DBSCAN Clustering')

plt.show()
