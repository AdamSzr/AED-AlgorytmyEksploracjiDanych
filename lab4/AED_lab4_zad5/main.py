import pandas as pd  # to load the dataframe
from sklearn.decomposition import PCA  # to apply PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# load train data
X = np.loadtxt("Train/X_train.txt")
y = np.loadtxt("Train/y_train.txt")

# scale data
scalar = StandardScaler()
scaled_data = pd.DataFrame(scalar.fit_transform(pd.DataFrame(X)))  # scaling the data
X = PCA(n_components=2).fit_transform(scaled_data)

# train model
svm = SVC(probability=True).fit(X, y)

# display trained model
disp = DecisionBoundaryDisplay.from_estimator(
    svm, X, response_method="predict",
)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k")
plt.show()
