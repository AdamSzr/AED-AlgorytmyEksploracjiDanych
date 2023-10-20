import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Wczytaj dane z pliku CSV
rawData = pd.read_csv(
    '/home/adam-pc/Documents/mgr/AED-AlgorytmyEksploracjiDanych/lab3/Sales_Transactions_Dataset_Weekly.csv')
rawData = rawData.iloc[:, 54:].dropna()
# Wybierz odpowiednie kolumny do klasteryzacji (numer klienta lub produkty, oraz odpowiednie cechy)
X = rawData.copy()
# Standardyzacja danych

# Wczytaj dane z dataframe
# Zakładamy, że 'df' to Twój dataframe
# Upewnij się, że Twój dataframe zawiera tylko liczbowe cechy, ponieważ K-means działa na danych numerycznych.

# Wczytaj dane z dataframe
# Zakładamy, że 'df' to Twój dataframe
# Upewnij się, że Twój dataframe zawiera tylko liczbowe cechy, ponieważ DBSCAN działa na danych numerycznych.
X = X.values

# Standardyzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Inicjalizacja modelu DBSCAN z dostosowanymi parametrami
dbscan = DBSCAN(eps=0.3, min_samples=5)  # Dostosuj parametry eps i min_samples

# Dopasowanie modelu do danych
dbscan.fit(X_scaled)

# Otrzymane etykiety klastrów
labels = dbscan.labels_

# Wyniki klasteryzacji
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print(f"Liczba wykrytych klastrów: {n_clusters_}")
print(f"Liczba próbek odstających: {n_noise_}")

# Ewentualnie, jeśli potrzebujesz, możesz dodać etykiety klastrów z powrotem do oryginalnego dataframe
rawData['Cluster_Labels'] = labels

# Tworzenie wykresu
plt.figure(figsize=(10, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title('Wykres DBSCAN z dostosowanymi parametrami')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()
