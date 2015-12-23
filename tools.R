learnModel <- function(data, labels){
  # Learn logit regression
  lambda <- 0.15
  result <- matrix(data = 10, nrow = 10, ncol = ncol(data) + 1, byrow = T)
  data <- cbind(1, data)
  initial_theta <- 0 * 1 : (ncol(data))
  for (i in 0:9){
    print(i)
    result[i+1,] <- GradDesc(initial_theta, data, (labels == i), lambda)
  }
  return(result)
}

testModel <- function(classifier, testData){
  # Returns prediction
  result <- max.col(sFunc(cbind(1, testData) %*% t(classifier)))
  result <- result - 1
  return(result)
}

logRegCostFunction <- function(theta, data, labels, lambda){
  # Cost function for logit regression
  result <- (sum((-labels) * log(sFunc(data %*% theta)) - (1 - labels) * log(1 - sFunc(data %*% theta)))) * (1 / length(labels)) + 
      (sum(sum(theta[2 : length(theta)]^2))) * (lambda / (2 * length(labels)))
  return(result)
}

gradLogRegCostFunction <- function(theta, data, labels, lambda){
  # Returns gradient for cost function
  result <- 1 / length(labels) * colSums((sFunc(data %*% theta) - labels) %*% rep(1, length(theta)) * data)
  result[2 : length(result)] <- result[2 : length(result)] + lambda / length(labels) * theta[2 : length(theta)]
  return(result)
}

sFunc <- function(arg){
  # Returns Sigma 
  sigma <- 1 / (1 + exp(-arg))
  return(sigma)
}

calcAndShowMetrics <- function(predictedLabels,testLabels){
#calculate the following error metrics:
#Recall, precision, specificity, F-measure, FDR and ROC
TP <- 0 * 1:10  # True Positive
TN <- 0 * 1:10  # True Negative
FP <- 0 * 1:10  # Fasle Positive
FN <- 0 * 1:10  # False Negative

for (i in 1:10){
  TP[i] <- sum((testLabels == (i - 1)) * (predictedLabels == (i - 1)))
  TN[i] <- sum((testLabels != (i - 1)) * (predictedLabels != (i - 1)))
  FP[i] <- sum((testLabels != (i - 1)) * (predictedLabels == (i - 1)))
  FN[i] <- sum((testLabels == (i - 1)) * (predictedLabels != (i - 1)))
  plot(roc(as.numeric((testLabels==(i - 1))), as.numeric(predictedLabels==(i - 1))))
}

# recall
accuracy <- (TP + TN) / (TP + TN + FP + FN)
print(accuracy)

# precision
precision <- TP / (TP + FP)
print(precision)

# specificity
specificity <- TP / (TP + FN)
print(specificity)

# F-measure
fmeasure <- 2 * TP / (2 * TP + FP + FN)
print(fmeasure)

# FDR
FDR <- FP / (FP + TP)
print(FDR)
}

GradDesc <- function(theta, data, labels, lambda){
  #Gradient descent
  repeat{
    initial_theta <- theta
    gradFunc <- gradLogRegCostFunction(initial_theta, data, labels, lambda)
    theta <- initial_theta - 0.0005 * gradFunc
    if(sum((initial_theta - theta)^2) < 0.0001){
      return(theta)
    }
  }
  return(theta)
}