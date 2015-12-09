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
TP <- 0 * 1:10  # True Positive
TN <- 0 * 1:10  # True Negative
FP <- 0 * 1:10  # Fasle Positive
FN <- 0 * 1:10  # False Negative

for (i in 1:10){
  TP[i] <- sum((testLabels == (i-1)) * (predictedLabels== (i-1)))
  TN[i] <- sum((testLabels != (i-1)) * (predictedLabels != (i-1)))
  FP[i] <- sum((testLabels != (i-1)) * (predictedLabels == (i-1)))
  FN[i] <- sum((testLabels == (i-1)) * (predictedLabels != (i-1)))
  plot(roc(as.numeric((testLabels==(i-1))), as.numeric(predictedLabels==(i-1))))
}

# Calculate error metrics

# recall
accuracy <- (TP+TN)/(TP+TN+FP+FN)
print(accuracy)

# precision
precision <- TP/(TP+FP)
print(precision)

# specificity
specificity <- TP/(TP+FN)
print(specificity)

# F-measure
fmeasure <- 2*TP/(2*TP+FP+FN)
print(fmeasure)

# FDR
FDR <- FP/(FP+TP)
print(FDR)