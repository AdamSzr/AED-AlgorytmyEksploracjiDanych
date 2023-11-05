# import libraries
import numpy as np

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

# load train data
X = np.loadtxt("Train/X_train.txt")
y = np.loadtxt("Train/y_train.txt")
X = X[0:100]
y = y[0:100]

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
print(confusion_matrix(y_test, svm_predict))
print(accuracy_score(y_test, svm_predict))
print(recall_score(y_test, svm_predict, average='micro'))
print(f1_score(y_test, svm_predict, average='micro'))
print(roc_auc_score(y, svm.predict_proba(X), multi_class='ovr'))


######################
# make kNN model
######################
knn = KNeighborsClassifier().fit(X, y)
knn_predict = knn.predict(X_test)

# calculate accurate predictions
print("kNN")
print(confusion_matrix(y_test, knn_predict))
print(accuracy_score(y_test, knn_predict))
print(recall_score(y_test, knn_predict, average='micro'))
print(f1_score(y_test, knn_predict, average='micro'))
print(roc_auc_score(y, knn.predict_proba(X), multi_class='ovr'))

######################
# make Decision Tree model
######################
decisionTree = tree.DecisionTreeClassifier().fit(X, y)
decisionTree_predict = decisionTree.predict(X_test)

# calculate accurate predictions
print("Decision Tree")
print(confusion_matrix(y_test, decisionTree_predict))
print(accuracy_score(y_test, decisionTree_predict))
print(recall_score(y_test, decisionTree_predict, average='micro'))
print(f1_score(y_test, decisionTree_predict, average='micro'))
print(roc_auc_score(y, decisionTree.predict_proba(X), multi_class='ovr'))

######################
# make Random Forest
######################
forest = RandomForestClassifier().fit(X, y)
forest_predict = forest.predict(X_test)

# calculate accurate predictions
print("Random Forest")
print(confusion_matrix(y_test, forest_predict))
print(accuracy_score(y_test, forest_predict))
print(recall_score(y_test, forest_predict, average='micro'))
print(f1_score(y_test, forest_predict, average='micro'))
print(roc_auc_score(y, forest.predict_proba(X), multi_class='ovr'))
