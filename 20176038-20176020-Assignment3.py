import numpy as np
import pandas as pd
from operator import itemgetter

dataset1 = pd.read_csv('TrainData.txt', header=None)
dataset2 = pd.read_csv('TestData.txt', header=None)

train_dataset = dataset1.values[:, 0:8]
test_dataset = dataset2.values[:, 0:8]

outputTrain = (dataset1.values[0:, [8]])
outputTest = (dataset2.values[0:, 8])

x1=train_dataset.min(axis=0)
x2=train_dataset.max(axis=0)
train_dataset = (train_dataset- x1)/ (x2-x1)
test_dataset = (test_dataset-x1)/(x2-x1)


def euclideanDistance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    result = np.linalg.norm(v1 - v2)
    return result


def NearestNeighbours(rowOfTest, k):
    allDistances = []
    for j in range(len(train_dataset)):
        result = euclideanDistance(rowOfTest, train_dataset[j])
        allDistances.append((result, outputTrain[j]))
    allDistances = sorted(allDistances, key=itemgetter(0))
    neighbours=getNeighbours(k,allDistances)
    return neighbours

def getNeighbours(k,allDistances):
    nearestNeighbours = []
    for i in range(k):
        nearestNeighbours.append(allDistances[i])
    return nearestNeighbours

def KNN_Classification(rowOfTest, k):
    outputs = []
    listNeighbours = NearestNeighbours(rowOfTest, k)
    listNeighbours = np.array(listNeighbours, dtype="object")
    for i in range(len(listNeighbours)):
        outputs.append(listNeighbours[i, 1])
    predicted = majority(outputs)
    return predicted

def majority(List):
    counter = 0
    string = List[0]
    for i in List:
        indexCount = List.count(i)
        if (indexCount > counter):
            string = i
            counter = indexCount

    return string

def sendTestRows(k):
    count = 0
    for i in range(len(test_dataset)):
        predicted = KNN_Classification(test_dataset[i], k)
        if predicted == outputTest[i]:
            count = count + 1
    return count

def getAccuracy(k):
    count=sendTestRows(k)
    print("Number of correctly classified instances:", count, ",Total number of instances:", len(outputTest),
          ", Accuracy:", (count / len(outputTest)))


k = 1
print("At K=1")
getAccuracy(k)

k = 3
print("At K=3")
getAccuracy(k)

k = 5
print("At K=5")
getAccuracy(k)

k = 7
print("At K=7")
getAccuracy(k)

k = 9
print("At K=9")
getAccuracy(k)
