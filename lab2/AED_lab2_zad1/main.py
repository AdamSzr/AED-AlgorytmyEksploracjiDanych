from sklearn.datasets import load_breast_cancer
import numpy as np

data = load_breast_cancer().data
target = load_breast_cancer().target

print("Wymiar wczytanych danych: " + str(len(data.shape)))

print("Ilość wartości unikalnych w wektorze target: " + str(len(np.unique(target))))


def reduceData(data, columns):
    def getIndexOfMin(arr):
        m = min(arr)
        for i in range(len(arr)):
            if arr[i] == m:
                return i

    for i in range(columns):
        variance = np.var(data, 0)
        index = getIndexOfMin(variance)
        data = np.delete(data, index, axis=1)

    return data


reducedData = reduceData(data, 2)
outputData = np.hstack((reducedData, np.array([target]).T))
np.savetxt('dataset_cut.csv', outputData, delimiter=';')
