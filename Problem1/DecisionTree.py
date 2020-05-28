"""
Created on Tue Nov 26 21:28:33 2019
@author: Om-mostafa El-Hariry
"""

import numpy as np
import pandas as pd
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

def DecisionTree(testSize, accuracyArr, nodeNum):
    X_train , X_test, Y_train, Y_test = train_test_split(X, Y, test_size = testSize, random_state = None)
    
    calc_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = None, max_depth = None, min_samples_leaf = 5)
    calc_entropy.fit(X_train, Y_train)  
    
    Y_predicted = calc_entropy.predict(X_test)
    
    nodes = calc_entropy.tree_.node_count
    
    accuracy = accuracy_score(Y_test, Y_predicted)*100
    
    accuracyArr.append(accuracy)
    #trainSize.append(len(X_train))
    nodeNum.append(nodes)
    return len(X_train)

def CalcMean(reqList):
    sum = 0
    for i in range(0, 5):
        sum += reqList[i]
    mean = sum/5
    return mean 

data = pd.read_csv('house-votes.csv', sep = ',', header = None)
#print ("data length = ", len(data))
#print ("data shape = ", data.shape)

data.replace(to_replace = 'y', value = 1, inplace = True)
data.replace(to_replace = 'n', value = 0, inplace = True)

for i in range(0, 435): 
    count_y = 0
    count_n = 0
    for j in range (1, 17):
        if (data[j][i] == 1):
            count_y += 1
        elif (data[j][i] == 0):
            count_n += 1
    if(count_y >= count_n):
        data.replace(to_replace = '?', value = 1, inplace = True)
    else:
         data.replace(to_replace = '?', value = 0, inplace = True)
    
X = data.values[:, 1:17]
Y = data.values[:, 0]

accuracyArr = []
trainSize = []
nodeNum = []

for i in range(0, 5):
    trainSize.append(DecisionTree(0.75, accuracyArr, nodeNum))
        
print('\n\n\nTraining set size 25% from all data, selected randomly:- \n')    
print('Acurracy', accuracyArr)
print('Training set size', trainSize)
print('Number of nodes ', nodeNum)


print('\n\n')
print('After Change Training set size to be from 30% to 70% :-\n')
trainSize = []
meanAccuracyArr = []
meanNodeNum = []

testNum = 0.7    
for j in range(0, 5):
    accuracyArr = []
    nodeNum = []
    trainSetSize = 0
    for i in range(0, 5) :
        trainSetSize = DecisionTree(testNum, accuracyArr, nodeNum)
    trainSize.append(trainSetSize)
    print('\n\nAt ', (1-testNum) * 100, '% of Training set size', trainSize[j], ':-')
    print('\n\tMax Accuracy ', max(accuracyArr))
    print('\n\tMin Accuracy ', min(accuracyArr))
    print('\n\tMax Number of Nodes ', max(nodeNum))
    print('\n\tMin Number of Nodes ', min(nodeNum))
    accuracyMean = CalcMean(accuracyArr)
    numOfNodeMean = CalcMean(nodeNum)
    print('\n\tAccuracy Mean =', accuracyMean)
    print('\n\tNumber of nodes Mean =', numOfNodeMean)
    meanAccuracyArr.append(accuracyMean)
    meanNodeNum.append(numOfNodeMean)
    testNum -= 0.1

print('\n\n') 
plt.subplot(2,1,1)
plt.plot(trainSize, meanAccuracyArr)
plt.xlabel('TrainSet size')
plt.ylabel('Accuracy')
plt.show()

plt.subplot(2,1,2)
plt.plot(trainSize, meanNodeNum)
plt.xlabel('TrainSet Size')
plt.ylabel('noOfNodes')
plt.show()
