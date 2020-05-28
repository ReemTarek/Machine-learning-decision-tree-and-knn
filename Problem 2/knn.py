import pandas
from math import sqrt
import csv
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset


#calculate ecludien distance
def calc_ecludien(row1,row2):
    totaldist=0
    for i in range(len(row1)-1):
        totaldist += (float(row1[i])-float(row2[i]))**2
   # print(row1[-1])
    return sqrt(totaldist)
#--------end function-------------
    
#-------calculate knn-------------
def knn(trainset,testset,k):
    distances = list()
    dist = list()
    for train in trainset:
        distance = calc_ecludien(train,testset)
        #print("train",train)
        dist.append(distance)
        distances.append((train,distance))
    #print("distances",distances)
    dist.sort()
    distances.sort(key = lambda tup:tup[1])
    nn = list()
    tie  = 0
    allarray = list()
    for i in range(k):
        if(dist.count(dist[i])==k):
            #print("tie")
            tie = 1
        nn.append(distances[i][0])
        allarray.append(distances)
    #print(nn)
    
    return nn , tie , allarray
#-----------end function----------
    
#---------classify----------------
def classify(train,test,k):
    nn , tie, allarray = knn(train,test,k)
    
    classes = list()
    #print(nn)
    for row in nn:
       # print("row",row)
        classes.append(row[-1])
    if(tie == 1):
        classes.sort()
        return classes[0]
    
    elements = list()
    counts = list()
    flag = 0
    for i in range(len(classes)):
       if(classes[i] in elements):
           continue
       else:
           elements.append(classes[i])
           counts.append(classes.count(classes[i]))
           
  #  for i in range(len(elements)):
   #    print("elements",elements[i])
    #if(len(counts) > 1 and len(counts) <= k):
     #   classes.sort()
      #  return classes[0]
    if(len(counts)==k):
        return nn[0][-1]
    maxcount = counts[0]
    maxelement = elements[0]
    for i in range(len(elements)):
        if(counts[i] > maxcount):
             maxcount = counts[i]
             maxelement = elements[i]
    return maxelement
#-------------end function--------
    
#---convert last element to int---
def turn_to_int(dataset,index):
    dictionary = list()
    classes = list()
    for i in range(len(dataset)):
        #print (dataset[i][index])
        classes.append(dataset[i][index])
    flag = 0
    for i in range(len(classes)):    
      if(classes[i] in dictionary and len(dictionary)>1):
         continue
      else:
         dictionary.append(classes[i])
   # for i in range(len(dictionary)):
    #    print(dictionary[i],i)
    for i in range(len(dataset)):
        for j in range(len(dictionary)):
            if (dataset[i][index] == dictionary[j]):
                dataset[i][index] = j
    return dictionary
#-----------end function----------
    
#---------convert to float--------
def turn_to_float(dataset):
    for i in range(len(dataset)):
        for j in range(len(dataset[i])-1):
            dataset[i][j] = float(dataset[i][j])
        #print (dataset[i])
        
#--------------MAIN---------------
traindata = load_csv("TrainData.csv")
testdata = load_csv("TestData.csv")
#del traindata[-1]
#del testdata[-1]
print("test")
turn_to_float(testdata)
print("train")
turn_to_float(traindata)
#turn_to_float(traindata[len(traindata)-1])
dictionarytrain = turn_to_int(traindata,len(traindata[0])-1)
dictionarytest = turn_to_int(testdata,len(testdata[0])-1)
#print("train data last ",traindata[len(traindata)-1])
#print("test data last ",testdata[len(testdata)-1])

#for i in range(len(testdata)-1):
#   print("train data",testdata[i][8])
sumofcor = 0
for j in range(1,9):
    print("at k = ",j)
    accuracy=list()
    newclass = list()
    for i in range(len(testdata)):
        newclass.append(classify(traindata,testdata[i],j))
    for i in range(len(testdata)):
        print("predicted class ",dictionarytrain[newclass[i]]," actual class ", dictionarytest[testdata[i][len(testdata[i])-1]])
        if(dictionarytest[testdata[i][-1]] == dictionarytrain[newclass[i]]):
            accuracy.append(1)
        else:
            accuracy.append(0)
    correct = accuracy.count(1)
    false = accuracy.count(0)
    print("correct results ",correct, " from total ",len(accuracy))
    print("accuracy ",correct/len(accuracy))
    print("------------------------------------------------------")