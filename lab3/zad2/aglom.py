import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch

# Przykład: Wczytanie danych jako DataFrame
data = pd.read_csv(
    '/home/adam-pc/Documents/mgr/AED-AlgorytmyEksploracjiDanych/lab3/Sales_Transactions_Dataset_Weekly.csv')


# Przetwórz dane, usuń brakujące wartości
data = data.dropna()

# # Wybierz odpowiednie kolumny jako przykładowe cechy
# features = data[['W0', 'W1', 'W2', 'W3']]

# # Hierarchiczna klasteryzacja
# dendrogram = sch.dendrogram(sch.linkage(features, method='ward'))

# # Przygotuj 9 etykiet do osi Y
# yticklabels = [str(i) for i in range(1, 10)]  # Zakładam, że chcesz 9 etykiet


# # Mapa cieplna z hierarchiczną klasteryzacją
# sns.clustermap(features, cmap='viridis',
#                method='ward', yticklabels=yticklabels)
# plt.title('Mapa Cieplna z Hierarchiczną Klasteryzacją')
# plt.show()


# Przetwórz dane, usuń brakujące wartości
data = data.iloc[:, 1:53]

# Wybierz kolumnę 'W1'
column_to_plot = 'W1'
data_to_plot = data[column_to_plot]

# Znajdź największą wartość w ramce danych
max_value = data.max().max()

# Utwórz 10 równo rozmieszczonych elementów od 0 do największej wartości
elements = [i * (max_value / 9) for i in range(10)]

print(elements)
# Utwórz zakresy na podstawie otrzymanych elementów
ranges = [(elements[i], elements[i + 1]) for i in range(len(elements) - 1)]


# Wyświetl zakresy
for i, (start, end) in enumerate(ranges):
    print(f'Zakres {i + 1}: {start} - {end}')

counts = []

# Iteruj po każdej kolumnie w ramce danych
for column in data:  # Pomijamy pierwszą kolumnę 'Product_Code'
    column_counts = [0] * len(ranges)

    # Iteruj po zakresach
    for i, (start, end) in enumerate(ranges):
        # Zlicz, ile wartości w danej kolumnie mieści się w zakresie
        column_counts[i] = ((data[column] >= start) &
                            (data[column] < end)).sum()

    counts.append(column_counts)


# Przygotuj dane w formie tablicy dwuwymiarowej (DataFrame)
heatmap_data = data

# Utwórz etykiety dla kolumn i wierszy
column_labels = [f'{start:.1f}-{end:.1f}' for start, end in ranges]


# Utwórz mapę cieplną
plt.figure(figsize=(12, 16))  # Rozmiar wykresu
sns.heatmap(counts, cmap='YlGnBu', annot=True, fmt='d',
            xticklabels=column_labels,  yticklabels=data.columns)

# Dodaj etykiety
plt.xlabel('Zakresy')
plt.ylabel('Wiersze')
plt.title('Mapa cieplna ilości wystąpień w zakresach')

# Wyświetl mapę cieplną
plt.show()

# Agglomerative Clustering z różnymi parametrami
n_clusters = 3  # Liczba klastrów
linkage_type = 'ward'  # Typ wiązania
distance_metric = 'euclidean'  # Metryka odległości

agg_cluster = AgglomerativeClustering(
    n_clusters=n_clusters, linkage=linkage_type, affinity=distance_metric)
cluster_labels = agg_cluster.fit_predict(data)

# Wyświetl etykiety klastrów
print("Etykiety klastrów:", cluster_labels)

# # Utwórz histogram
# plt.hist(data_to_plot, bins=20, edgecolor='k')  # 20 koszyków histogramu

# # Dodaj etykiety i tytuł
# plt.xlabel('Wartość')
# plt.ylabel('Liczebność')
# plt.title(f'Histogram dla kolumny "{column_to_plot}"')

# # Wyświetl histogram
# plt.show()

# # Podziel dane na 8 zakresów, co 10
# min_value = data.min().iloc[1:53].min()  # Minimalna wartość w danych
# max_value = data.max().iloc[1:53].max()  # Maksymalna wartość w danych
# bin_size = 73  # Rozmiar każdego zakresu
# num_bins = 8

# bin_ranges = [(i, i + bin_size) for i in range(min_value, max_value, bin_size)]

# # Inicjalizuj słownik do zliczania liczb w każdym zakresie
# count_dict = {i: 0 for i in range(num_bins)}

# # Iteruj po danych i zlicz liczbę w każdym zakresie
# # Pomijamy pierwszą kolumnę "Product_Code" i ostatnie trzy kolumny "MIN", "MAX", "Normalized"
# for column in data.columns[1:53]:
#     for value in data[column]:
#         for bin_idx, (bin_start, bin_end) in enumerate(bin_ranges):
#             if bin_start <= value < bin_end:
#                 count_dict[bin_idx] += 1

# # Wyświetl liczbę w każdym zakresie
# for bin_idx, (bin_start, bin_end) in enumerate(bin_ranges):
#     print(f"Zakres {bin_start}-{bin_end}: {count_dict[bin_idx]} liczb")

# # Przygotuj dane do mapy cieplnej
# max_value = data.max().iloc[1:53].max()  # Maksymalna wartość w danych
# data_counts = data.apply(lambda x: np.histogram(
#     x, bins=np.arange(max_value+2))[0])

# # Utwórz mapę cieplną
# sns.set(font_scale=1.2)
# sns.heatmap(data_counts, cmap='viridis', annot=True, cbar=True, linewidths=0.5)

# # Dodaj tytuł i etykiety osi
# plt.title('Mapa Cieplna z Ilością Wystąpień Dla Każdej Wartości')
# plt.xlabel('Wartość')
# plt.ylabel('Cecha "Wx"')

# # Wyświetl mapę cieplną
# plt.show()


# # Podziel dane na 10 zakresów i oblicz średnią dla każdego zakresu
# num_ranges = 10
# data_range = len(data) // num_ranges

# result_data = []
# for i in range(num_ranges):
#     start_idx = i * data_range
#     end_idx = (i + 1) * data_range
#     # Wybieramy kolumny z W0 do W51
#     range_data = data.iloc[start_idx:end_idx, 1:-3]
#     avg_data = range_data.mean()
#     result_data.append(avg_data)

# # Utwórz nowy DataFrame na podstawie wynikowych danych
# result_df = pd.DataFrame(result_data)

# # Teraz masz DataFrame z 10 wierszami, z których każdy zawiera średnią z określonego zakresu
# print(result_df)

# # Wybierz cechy 'Wx' i utwórz DataFrame z 10 wartościami dla każdej cechy
# features = pd.DataFrame()
# for i in range(52):
#     feature_name = f'W{i}'
#     # Wybierz pierwsze 10 wartości
#     feature_values = data[feature_name].values[:10]
#     features[feature_name] = feature_values

# # Utwórz mapę cieplną
# sns.set(font_scale=1.2)
# sns.heatmap(result_df[:, 50], cmap='viridis',
#             annot=True, cbar=True, linewidths=0.5)

# # Dodaj tytuł i etykiety osi
# plt.title('Mapa Cieplna z 10 Wartościami dla Każdej Cechy "Wx"')
# plt.xlabel('Cecha "Wx"')
# plt.ylabel('Numer próbki')

# # Wyświetl mapę cieplną
# plt.show()
# Ten kod wczytuje dane, wybiera 10 pierwszych wartości dla każdej cechy 'Wx' i tworzy mapę cieplną na podstawie tych wartości. Mapa cieplna wyświetla wartości w formie kolorowych pól na krzyżu osi X (cechy) i osi Y (numery próbek). Możesz dostosować parametry mapy cieplnej, takie jak kolorystyka (cmap) i etykiety, zgodnie z własnymi preferencjami.
