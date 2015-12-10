# This sciprt file contains a frame for learning handwritten digitals from the MNIST dataset
source("tools.R")
source("load_data.R")
library("pROC")
# load training data from files
data <- loadMNISTData("D:\\LearnMNIST\\LearnMNIST\\train-images.idx3-ubyte", "D:\\LearnMNIST\\LearnMNIST\\train-labels.idx1-ubyte")
trainLabels <- data$labels
trainData <- data$data

print(dim(trainData))
print(dim(trainLabels))

# train a model
classifier <- learnModel(data = trainData, labels = trainLabels)

predictedLabels <- testModel(classifier, trainData)

#calculate accuracy on training data
print("accuracy on training data:\t")
print(sum(predictedLabels == trainLabels)/length(trainLabels))

#calculate the following error metric for each class obtained on the train data:
#Recall, precision, specificity, F-measure, FDR and ROC for each class separately. Use a package for ROC. 
calcAndShowMetrics(predictedLabels,trainLabels)

# test the model
data <- loadMNISTData("D:\\LearnMNIST\\LearnMNIST\\t10k-images.idx3-ubyte", "D:\\LearnMNIST\\LearnMNIST\\t10k-labels.idx1-ubyte")
testLabels <- data$labels
testData <- data$data

print(dim(testData))
print(dim(testLabels))
#trainingData should be 10000x786,  10000 data and 784 features (28x28), tha matrix trainData has 10000 rows and 784 columns
#trainingLabels should have 10000x1, one class label \in {0,1,...9} for each data.

predictedLabels <- testModel(classifier, testData)

#calculate accuracy
print("accuracy on test data:\t")
print(sum(predictedLabels == testLabels)/length(testLabels))

#calculate the following error metric for each class obtained on the test data:
#Recall, precision, specificity, F-measure, FDR and ROC for each class separately. Use a package for ROC. 
calcAndShowMetrics(predictedLabels,testLabels)