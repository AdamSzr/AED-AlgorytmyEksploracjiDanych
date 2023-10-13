import pandas as pd
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt

# Wczytaj dane z pliku CSV do DataFrame
data = pd.read_csv('Sales_Transactions_Dataset_Weekly.csv')

# Przetwórz dane, usuń brakujące wartości
data = data.dropna()

values = []

# Utwórz pętlę od 0 do 51 (włącznie)
for i in range(52):
    value = f'W{i}'
    values.append(value)

features = data[values].copy()  # Stwórz kopię danych


# Poproś użytkownika o wprowadzenie liczby klastrów
n_clusters = int(input("Podaj liczbę klastrów: "))


# Przeprowadź analizę K-Medoids na kopii danych
kmedoids = KMedoids(n_clusters=n_clusters)  # Określ liczbę klastrów, np. 3
kmedoids.fit(features)

# Dodaj wyniki klastry do kopii DataFrame
features['Cluster'] = kmedoids.labels_

# Wykres klastrów dla całego zestawu danych (K-Medoids)
plt.scatter(features['W0'], features['W1'],
            c=features['Cluster'], cmap='viridis')
plt.xlabel('W0')
plt.ylabel('W1')
plt.title('Klastrowanie danych (K-Medoids)')

# Dodaj centra klastrów (medoidy) na wykres
medoids = features.iloc[kmedoids.medoid_indices_]
plt.scatter(medoids['W0'], medoids['W1'], c='red',
            marker='x', s=200, label='Medoids')

plt.legend()
plt.show()
