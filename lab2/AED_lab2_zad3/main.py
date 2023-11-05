# Import necessary libraries
from matplotlib import pyplot as plt
import pandas as pd  # to load the dataframe
import numpy as np  # to save in csv
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.decomposition import PCA  # to apply PCA
from sklearn.datasets import load_breast_cancer  # dane do pracy

# Load the Dataset
cancer = load_breast_cancer()
# convert the dataset into a pandas data frame
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

scalar = StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(df))  # scaling the data

pca = PCA()
pca.fit_transform(scaled_data)

# obliczamy skumulowaną wariancję, żeby sprawdzić ile kolumn potrzebujemy
cum_sum_eigenvalues = np.cumsum(pca.explained_variance_ratio_)

components = len(list(filter(lambda v: v < 0.9, cum_sum_eigenvalues))) + 1
pca2 = PCA(n_components=components)
data_pca = pca2.fit_transform(scaled_data)
outputData = np.hstack((data_pca, np.array([cancer.target]).T))
np.savetxt('dataset_pca_' + str(components) + '.csv', outputData, delimiter=';')

print(pca2.explained_variance_)  # wariancja zbioru
print(pca2.explained_variance_ratio_)  # wariancja wyjaśniona

# do celów wizualizacji
exp_var_pca = pca2.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)
#
# Create the visualization plot
#
plt.bar(range(0,len(exp_var_pca)), exp_var_pca, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(0,len(cum_sum_eigenvalues)), cum_sum_eigenvalues, where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal component index')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

