import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Wczytaj dane z pliku CSV do DataFrame
data = pd.read_csv(
    '/home/adam-pc/Documents/mgr/AED-AlgorytmyEksploracjiDanych/lab3/Sales_Transactions_Dataset_Weekly.csv')

# Przetwórz dane, usuń brakujące wartości
data = data.iloc[:, 1:53].dropna()

values = []

min_clusters = 2
max_clusters = 15
# Pusty słownik do przechowywania wyników średnich miar silhouette
silhouette_scores = {}
# Konwertuj DataFrame do tablicy numpy
data = data.to_numpy()

# Iteruj po liczbie klastrów i oblicz miarę silhouette
for n_clusters in range(min_clusters, max_clusters + 1):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init='auto')
    cluster_labels = kmeans.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    silhouette_scores[n_clusters] = silhouette_avg

# Tworzenie wykresu z wynikami średnich miar silhouette
# Rozdziel dane na oś X i oś Y
x_values = list(silhouette_scores.keys())
y_values = list(silhouette_scores.values())
# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, marker='o', linestyle='-')
plt.xlabel('Liczba klastrów')
plt.ylabel('Silhouette')
plt.title('Optymalna liczba klastrów K-Means / K-Means++')
plt.grid(True)
plt.show()


# Pusty słownik do przechowywania wyników średnich miar silhouette
silhouette_scores = {}

# Iteruj po liczbie klastrów i oblicz miarę silhouette
for n_clusters in range(min_clusters, max_clusters + 1):
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmedoids.fit_predict(data)
    silhouette_avg = silhouette_score(data, cluster_labels)
    silhouette_scores[n_clusters] = silhouette_avg

# Tworzenie wykresu z wynikami średnich miar silhouette
plt.figure(figsize=(10, 6))
plt.plot(list(silhouette_scores.keys()), list(
    silhouette_scores.values()), marker='o', linestyle='-')
plt.xlabel('Liczba klastrów')
plt.ylabel('Średnia miara silhouette')
plt.title('Optymalna liczba klastrów (K-Medoids)')
plt.grid(True)
plt.show()

# TODO: DBSCAN
# POPRAW Aglomeracjna
