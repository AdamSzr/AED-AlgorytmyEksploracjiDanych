import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Wczytaj dane z pliku CSV do DataFrame
data = pd.read_csv(
    '/home/adam-pc/Documents/mgr/AED-AlgorytmyEksploracjiDanych/lab3/Sales_Transactions_Dataset_Weekly.csv')

# Przetwórz dane, usuń brakujące wartości
data = data.dropna()

values = []

# Utwórz pętlę od 0 do 51 (włącznie)
for i in range(52):
    value = f'W{i}'
    values.append(value)

features = data[values].copy()


# Przygotuj listę wartości inercji dla różnej liczby klastrów
inertia_values = []

# Przeprowadź analizę K-means dla różnych liczb klastrów
for n_clusters in range(1, 8):
    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(features)
    inertia_values.append(kmeans.inertia_)

# Wykres metody "łokcia"
plt.plot(range(1, 8), inertia_values, marker='o')
plt.xlabel('Liczba klastrów')
plt.ylabel('Wartość inercji')
plt.title('Metoda Łokciowa')
plt.show()

# Przeprowadź analizę K-means na danych
# Określ liczbę klastrów, np. 3
# Poproś użytkownika o wprowadzenie liczby klastrów
n_clusters = int(input("Podaj liczbę klastrów: "))

kmeans = KMeans(n_clusters=n_clusters, n_init=10, init='k-means++')
kmeans.fit(features)

# Dodaj wyniki klastry do DataFrame
features['Cluster'] = kmeans.labels_
print(features)

# Wykres
plt.scatter(features['W2'], features['W3'],
            c=features['Cluster'], cmap='viridis')
plt.xlabel('W2')
plt.ylabel('W3')
plt.title('Analiza K-means ++')

# Dodaj centra klastrów na wykres
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], c='red',
            marker='x', s=200, label='Centroids')

# Dodaj etykiety do centrów
for i, txt in enumerate(centroids):
    plt.annotate(f'Centroid {i}', (centroids[i, 0], centroids[i, 1]),
                 textcoords="offset points", xytext=(0, 10), ha='center')

plt.legend()
plt.show()

# Wybierz dane należące do konkretnego klastra (np. klastra o indeksie 0)
cluster_0_data = features.loc[features['Cluster'] == 0]

# Wyświetl dane należące do klastra 0
print("Dane dla klastra 0:")
print(cluster_0_data)
