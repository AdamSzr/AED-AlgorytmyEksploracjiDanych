import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Wczytaj dane z pliku CSV do DataFrame
data = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')

# Przetwórz dane, usuń brakujące wartości
data = data.dropna()

values = []

# Utwórz pętlę od 0 do 51 (włącznie)
for i in range(52):
    # value = f'W{i}'
    value = f'Normalized {i}'
    values.append(value)

features = data[values].copy()  # Stwórz kopię danych

# Przygotuj dane (standaryzacja)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Przygotuj listę algorytmów i nazw
algorithms = [KMeans, KMedoids, AgglomerativeClustering, DBSCAN]
algorithm_names = ['K-Means', 'K-Medoids', 'Agglomerative', 'DBSCAN']

# Przygotuj zakres liczby klastrów
n_clusters_range = range(2, 16)

# Domyślna wartość n_init
default_n_init = 10

# Przygotuj wykres
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Średnia miara silhouette dla różnych algorytmów')

for i, (algorithm, name) in enumerate(zip(algorithms, algorithm_names)):
    ax = axes[i // 2, i % 2]
    silhouette_scores = []

    for n_clusters in n_clusters_range:
        if algorithm == DBSCAN and n_clusters == 1:
            continue  # DBSCAN requires at least 2 clusters
        if algorithm == AgglomerativeClustering and n_clusters == 1:
            continue  # AgglomerativeClustering requires at least 2 clusters

        if algorithm == KMedoids:
            model = algorithm(n_clusters=n_clusters, random_state=0)
        elif algorithm == DBSCAN:
            model = algorithm(eps=5, min_samples=5)
        elif algorithm == AgglomerativeClustering:
            model = algorithm(n_clusters=n_clusters,
                              linkage='ward', metric='euclidean')
        else:
            model = algorithm(n_clusters=n_clusters,
                              n_init=default_n_init, random_state=0)

        labels = model.fit_predict(scaled_features)
        if len(np.unique(labels)) > 1:
            silhouette_avg = silhouette_score(scaled_features, labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)

    ax.plot(n_clusters_range, silhouette_scores,
            marker='o', linestyle='-', label=name)
    ax.set_title(name)
    ax.set_xlabel('Liczba klastrów')
    ax.set_ylabel('Średnia miara silhouette')

plt.tight_layout()
plt.show()
