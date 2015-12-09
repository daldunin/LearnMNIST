learnModel <- function(data, labels){
  # Learn logit regression
  lambda <- 0.15
  result <- matrix(data = 0, nrow = 10, ncol = ncol(data) + 1, byrow = T)
  initial_theta <- 0 * 1 : (ncol(data) + 1)
  for (i in 1:10){
    print(i)
    buff <- optim(initial_theta, logRegCostFunction, gr = gradLinRegCostFunction,
    	data=cbind(1, data), labels = (labels == (i - 1)), lambda = lambda, method = "CG")
    result[i,] <- buff$par
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
  result <- (sum(-labels * log(sFunc(data %*% theta)) - (1 - labels) * log(1 - sFunc(data %*% theta)))) * (1 / length(labels)) + 
      (sum(sum(theta[2 : length(theta)]^2))) * (lambda / (2 * length(labels)))
  return(result)
}

gradLinRegCostFunction <- function(theta, data, labels, lambda){
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