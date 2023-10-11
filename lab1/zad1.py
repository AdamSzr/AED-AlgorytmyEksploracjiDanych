import pandas as pd
# Wczytanie danych z pliku CSV
df = pd.read_csv('netflix_titles.csv')

# Wyświetlenie pierwszych kilku wierszy ramki danych
[wiersze_count, kolumny_count] = df.shape
print('Ilość wczytanych wierszy: ' + str(wiersze_count))

size_str = str(kolumny_count) + ' x ' + str(wiersze_count)
print('Wymiar wczytanych danych (' + size_str + ")")

print( df.isnull().sum())
ilosc_null = df.isnull().sum().sum()
print('Ilość wystąpien wartości "null": ' + str(ilosc_null))