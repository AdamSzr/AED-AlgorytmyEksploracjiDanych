# import libraries
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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
print("SVM")
scores = cross_val_score(svm, X, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

######################
# make kNN model
######################
knn = KNeighborsClassifier().fit(X, y)
knn_predict = knn.predict(X_test)

# calculate accurate predictions
print("kNN")
scores = cross_val_score(knn, X, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

######################
# make Decision Tree model
######################
decisionTree = DecisionTreeClassifier().fit(X, y)
decisionTree_predict = decisionTree.predict(X_test)

# calculate accurate predictions
print("Decision Tree")
scores = cross_val_score(decisionTree, X, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

######################
# make Random Forest
######################
forest = RandomForestClassifier().fit(X, y)
forest_predict = forest.predict(X_test)

# calculate accurate predictions
print("Random Forest")
scores = cross_val_score(forest, X, y, cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
