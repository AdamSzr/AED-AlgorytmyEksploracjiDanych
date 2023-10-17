import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Przykładowe dane - macierz 100x10
data = np.random.rand(10, 10)

# Redukcja wymiarowości przy użyciu PCA
pca = PCA(n_components=3)  # Redukcja do 2 wymiarów
data_reduced = pca.fit_transform(data)

# Tworzenie mapy cieplnej na zredukowanych danych
sns.set(font_scale=1.2)
sns.heatmap(data_reduced, cmap='viridis',
            annot=True, cbar=True, linewidths=0.5)

# Dodanie tytułu i osi
plt.title('Mapa Cieplna z Zredukowanymi Danymi (PCA)')
plt.xlabel('Oś X')
plt.ylabel('Oś Y')

# Wyświetlenie mapy cieplnej
plt.show()
