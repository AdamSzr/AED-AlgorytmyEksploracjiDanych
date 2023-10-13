import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import seaborn as sns

# Wygeneruj dane w kształcie "księżyca"
X, _ = make_moons(n_samples=200, noise=0.05, random_state=0)
data = pd.DataFrame(X, columns=['X', 'Y'])

# Wybierz parametry DBSCAN
eps = 0.2  # Promień sąsiedztwa
min_samples = 5  # Minimalna liczba próbek w sąsiedztwie

# Przeprowadź analizę klastrowania DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
print(max(data['Y']))
dbscan.fit(data)

# Przygotuj etykiety klastrów w formie kategorii całkowitych
unique_labels = set(dbscan.labels_)
cluster_labels = {}
for label in unique_labels:
    if label == -1:
        cluster_labels[label] = 'Noise'
    else:
        cluster_labels[label] = len(cluster_labels) + 1

# Dodaj kategorie do DataFrame
data['Cluster'] = [cluster_labels[label] for label in dbscan.labels_]

# Wykres z oznaczeniem klastrów
sns.scatterplot(x=data['X'], y=data['Y'], hue=data['Cluster'],
                style=data['Cluster'], palette='viridis')
plt.title('DBSCAN Clustering - Moon-Shaped Data')

plt.show()
