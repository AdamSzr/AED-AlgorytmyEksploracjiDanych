# Import necessary libraries
import pandas as pd  # to load the dataframe
import numpy as np  # to save in csv
from sklearn.preprocessing import StandardScaler  # to standardize the features
from sklearn.datasets import load_breast_cancer  # dane do pracy
from sklearn.decomposition import FastICA  # to apply ICA

# Load the Dataset
cancer = load_breast_cancer()
# convert the dataset into a pandas data frame
df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

scalar = StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(df))  # scaling the data

transformer = FastICA(n_components=7)
data_ica = transformer.fit_transform(scaled_data)
print(data_ica.shape)
print(data_ica)

outputData = np.hstack((data_ica, np.array([cancer.target]).T))
np.savetxt('dataset_ica_7.csv', outputData, delimiter=';')
