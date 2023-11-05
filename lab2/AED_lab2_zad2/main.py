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

pca = PCA(n_components=5)
data_pca = pca.fit_transform(scaled_data)
outputData = np.hstack((data_pca, np.array([cancer.target]).T))
np.savetxt('dataset_pca_5.csv', outputData, delimiter=';')

print(pca.explained_variance_)  # wariancja zbioru
print(pca.explained_variance_ratio_)  # wariancja wyjaśniona
variance = np.var(data_pca, 0)
print(variance)  # wariancja zbioru (nie wiem czemu delikatnie się różni)

# do celów wizualizacji
exp_var_pca = pca.explained_variance_ratio_
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
