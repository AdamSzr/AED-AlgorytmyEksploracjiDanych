import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from cluster_evaluation import cluster_evaluation
import numpy as np

data = pd.read_csv(
    '/home/adam-pc/Documents/mgr/AED-AlgorytmyEksploracjiDanych/lab3/Sales_Transactions_Dataset_Weekly.csv')
product_codes = data['Product_Code']
# Usuń kolumnę 'Product_Code' z danych
col_num = [0]
col_num.extend(list(np.arange(1, 53)))
X = data.iloc[:, col_num]  \
    .set_index(keys="Product_Code")
data = data.drop(columns=['Product_Code'])

wcss = []  # Within-Cluster-Sum-of-Squares
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('Metoda "łokcia"')
plt.xlabel('Liczba klastrów')
plt.ylabel('Within-Cluster-Sum-of-Squares')
plt.show()

# Wybierz optymalną liczbę klastrów
kmeans = KMeans(n_clusters=3, n_init='auto')
kmeans.fit(data)
clusterLabels = kmeans.fit_predict(data)
# Dodaj kolumnę z etykietami klastrów do danych
data['Cluster'] = kmeans.labels_

# -------------------------------------------------
# Usuń kolumnę "Product_Code"
data2 = data.copy()

# Grupowanie danych na podstawie kolumny 'Cluster'
grouped_data = data2.groupby('Cluster')

n_clusters = grouped_data.ngroups  # Pobierz liczbę klastrów

# Lista do przechowywania tablic płaskich dla każdej grupy
flat_arrays = []

# Iteracja po klastrach i tworzenie tablic płaskich
for cluster_id in range(n_clusters):
    group_values = grouped_data.get_group(
        cluster_id).drop(columns=['Cluster']).values
    flat_array = group_values.flatten()
    flat_arrays.append(flat_array)

# flat_arrays zawiera teraz tablice płaskie dla każdej grupy
for cluster_id, flat_array in enumerate(flat_arrays):
    print(
        f'Flat array [{flat_array.size}] for Cluster {cluster_id}: {flat_array} ')

# Tworzenie wykresu z oznaczeniem klastrów
fig, ax = plt.subplots()

label = f'Klaster{cluster_id}'
ax.scatter(flat_arrays[0], flat_arrays[1], label=label)

plt.xlabel('Indeks')
plt.ylabel('Wartości')
plt.legend()
plt.show()

# ------------------------------------------------------

cluster_centers = pd.DataFrame(
    kmeans.cluster_centers_, columns=data.columns[:-1])
print(cluster_centers)

# fig, ax = plt.subplots()

# for cluster in range(3):  # Teraz używamy 3 klastrów
#     cluster_data = data[data['Cluster'] == cluster]

#     ax.scatter(cluster_data['W0'], cluster_data['W4'],
#                label=f'Klaster {cluster}')

# # Dodaj centra klastrów na wykres
# centroids = kmeans.cluster_centers_
# plt.scatter(centroids[:, 0], centroids[:, 4], c='red',
#             marker='x', s=200, label='Centroids')

# # Dodaj etykiety do centrów
# for i, txt in enumerate(centroids):
#     plt.annotate(f'Centroid K-{i}', (centroids[i, 0], centroids[i, 1]),
#                  textcoords="offset points", xytext=(0, 10), ha='center')

# plt.xlabel('Wartości kolumny')
# plt.ylabel('Wartości kolumny')
# plt.legend()
# plt.show()

# X_clustered = pd.concat([X,
#                          pd.DataFrame(cluster_labels,
#                                       columns=['cluster_label'], index=X.index)],
#                         axis=1)


# cluster_evaluation(X=X, range_n_clusters=range(
#     2, 6, 1), xlowerlim=-0.3, store_values=True)
