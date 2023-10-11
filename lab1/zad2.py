import seaborn as sns
import pandas as pd

titanic_data = sns.load_dataset("titanic")
df = pd.DataFrame(titanic_data)
ilosc_nan = df.isnull().sum().sum()
print("Ilość wartości pustych (NaN) w zbiorze danych:", ilosc_nan)

ilosc_nan_w_kolumnach = df.isnull().sum()
skumulowana_ilosc_nan = ilosc_nan_w_kolumnach.cumsum()
print("Ilość wartości pustych (null) w każdej kolumnie:")
print(ilosc_nan_w_kolumnach.to_frame().T)
print("\nSuma skumulowana ilości wartości pustych (null) w kolumnach:")
print(skumulowana_ilosc_nan)

prog = 0.3  # 30%
column_count = str(df.shape[1])
minimalna_liczba_wartosci = int((1 - prog) * len(df))
df = df.dropna(axis=1, thresh=minimalna_liczba_wartosci)
print('\nIlość kolumn przed usunięciem: '+ column_count)
print("Ilość kolumn po usunięciu: "+ str(df.shape[1]))

mapowanie = {"female": 0, "male": 1}
df["sex"] = df["sex"].map(mapowanie)
print("\nRamka danych po zamianie danych kategorycznych:")
print(df['sex'].head(10).to_frame(name='sex').T)