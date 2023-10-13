import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
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

# Przygotuj mapę cieplną
sns.heatmap(features.transpose(), cmap='viridis')

# Przygotuj dendrogram
linkage_matrix = linkage(features, method=linkage_type, metric=distance_metric)
dendrogram(linkage_matrix, orientation="top")

plt.title('Klasteryzacja aglomeracyjna - Dendrogram i mapa cieplna')
plt.show()
