import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
import numpy as np

# Wczytaj dane z pliku CSV do DataFrame
data = pd.read_csv(
    '/home/adam-pc/Documents/mgr/AED-AlgorytmyEksploracjiDanych/lab3/Sales_Transactions_Dataset_Weekly.csv')

# Przetwórz dane, usuń brakujące wartości
data = data.iloc[:, 1:53].dropna()

values = []

# Utwórz pętlę od 0 do 51 (włącznie)
for i in range(52):
    value = f'W{i}'
    # value = f'Normalized {i}'
    values.append(value)

features = data[values].copy()  # Stwórz kopię danych

# Wybierz parametry AgglomerativeClustering
n_clusters = 3  # Określ liczbę klastrów, np. 3
linkage_type = 'ward'  # Typ wiązania: 'ward', 'complete', 'average', itp.
# Metryka odległości: 'euclidean', 'manhattan', itp.
distance_metric = 'euclidean'

# Przeprowadź analizę AgglomerativeClustering
agg_clustering = AgglomerativeClustering(
    n_clusters=n_clusters, linkage=linkage_type, affinity=distance_metric)
agg_clustering.fit(features)

# Dodaj wyniki klastry do kopii DataFrame
features['Cluster'] = agg_clustering.labels_

# Zakresy przedziałów wartości
n_bins = 10
value_ranges = np.linspace(0, data.max().max(), n_bins + 1)

# Inicjalizacja pustej tablicy wynikowej
heatmap_data = []

# Inicjalizacja słownika do zliczania klastrów
cluster_counts = {cluster: [] for cluster in range(n_clusters)}

# Iteruj po kolumnie 'Cluster'
for cluster in features['Cluster']:
    for i in range(n_clusters):
        # Dla każdego klastra zapisz wartość 1, jeśli jest to szukany klaster, w przeciwnym razie 0
        cluster_counts[i].append(1 if cluster == i else 0)

# Konwertuj słownik na DataFrame
cluster_counts_df = pd.DataFrame(cluster_counts)

# Wykreśl wykres słupkowy ilości wystąpień klastrów
cluster_counts_df.sum().plot(kind='bar')


# Inicjalizacja słownika do przechowywania wyników
results = {i: [] for i in range(n_clusters)}

# Podziel dane na 10 równych zakresów
n_ranges = 10
range_size = len(features) // n_ranges

for i in range(n_ranges):
    start = i * range_size
    end = (i + 1) * range_size

    # Dla każdego klastra, zlicz wystąpienia w danym zakresie
    for cluster in range(n_clusters):
        counts = (features.iloc[start:end]['Cluster'] == cluster).sum()
        results[cluster].append({'range': f'{start}-{end}', 'count': counts})

print(results)

# Konwersja danych na ramkę danych
df = pd.DataFrame(results).T


# Tworzenie tablicy z wartościami zakresów "range"
range_values = [item['range'] for item in results[0]]

# Wyświetlenie tablicy z wartościami 'range'
print(range_values)

# Tworzenie tablicy z wartościami pola "count"

v = []

for key, value in results.items():
    counts = [item['count'] for item in value]
    v.append(counts)

print(v)


# Tworzenie heatmapy
plt.figure(figsize=(10, 6))
sns.heatmap(v, cmap='hot', annot=True, fmt='d', xticklabels=range_values,
            vmin=0, vmax=max(map(max, v)))
plt.xlabel('Range')
plt.ylabel('Cluster')
plt.title('Heatmap')
plt.show()
