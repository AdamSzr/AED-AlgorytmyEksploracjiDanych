# import libraries
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

# load train data
X = np.loadtxt("Train/X_train.txt")
y = np.loadtxt("Train/y_train.txt")

# load test data
X_test = np.loadtxt("Test/X_test.txt")
y_test = np.loadtxt("Test/y_test.txt")

######################
# make SVM model
######################
svm = SVC(probability=True).fit(X, y)
svm_predict = svm.predict(X_test)

# calculate accurate predictions
correct = len(y_test) - np.count_nonzero(np.subtract(y_test, svm_predict))
print("Model SVM poprawnie zaklasyfikował " + str(correct) + " próbek na " + str(len(y_test)))

######################
# make kNN model
######################
knn = KNeighborsClassifier().fit(X, y)
knn_predict = knn.predict(X_test)

# calculate accurate predictions
correct = len(y_test) - np.count_nonzero(np.subtract(y_test, knn_predict))
print("Model kNN poprawnie zaklasyfikował " + str(correct) + " próbek na " + str(len(y_test)))

######################
# make Decision Tree model
######################
decisionTree = tree.DecisionTreeClassifier().fit(X, y)
decisionTree_predict = decisionTree.predict(X_test)

# calculate accurate predictions
correct = len(y_test) - np.count_nonzero(np.subtract(y_test, decisionTree_predict))
print("Model Decision Tree poprawnie zaklasyfikował " + str(correct) + " próbek na " + str(len(y_test)))

######################
# make Random Forest
######################
forest = RandomForestClassifier().fit(X, y)
forest_predict = forest.predict(X_test)

# calculate accurate predictions
correct = len(y_test) - np.count_nonzero(np.subtract(y_test, forest_predict))
print("Model Random Forest poprawnie zaklasyfikował " + str(correct) + " próbek na " + str(len(y_test)))
