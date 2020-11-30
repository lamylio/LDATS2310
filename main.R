---
title: "Main"
author: "Lamy Lionel"
date: "10/11/2020"
output: pdf_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

# Load all the libraries in one-shot

```{r message=F, warning=F}
library(MASS)
library(boot)
library(mgcv)
library(pscl)
library(gbm)

library(ggplot2)
library(dplyr)
library(tidyr)
library(countreg)

library(rpart)
library(rpart.plot)

library(glmnet)
library(caret)
library(jtools)

library(rfCountData) # From https://github.com/fpechon/rfCountData
```

# Retrieve and format the two datasets

```{r}
base_test <- read.table("./data/DBtest.csv", sep=",", header=TRUE)
base_train = read.table("./data/DBtrain.csv", sep=",", header=TRUE)

base_train = within(base_train, {
  X = NULL
  Gender = factor(Gender, labels=c("M", "F"))
  Area = factor(Area, labels=c("Suburban", "Urban", "Countryside low", "Coutryside high"))
  Leasing = factor(Leasing, labels=c("Yes", "No"))
  Power = factor(Power, labels=c("Low", "Normal", "Intermediate", "High"))
  Contract = factor(Contract,  labels=c("Basic", "Intermediate", "Full"))
  Fract = factor(Fract, labels=c("Monthly", "Quarterly", "Yearly"))
})

base_test = within(base_test, {
  X = NULL
  Gender = factor(Gender, labels=c("M", "F"))
  Area = factor(Area, labels=c("Suburban", "Urban", "Countryside low", "Coutryside high"))
  Leasing = factor(Leasing, labels=c("Yes", "No"))
  Power = factor(Power, labels=c("Low", "Normal", "Intermediate", "High"))
  Contract = factor(Contract, labels=c("Basic", "Intermediate", "Full"))
  Fract = factor(Fract, labels=c("Monthly", "Quarterly", "Yearly"))
})

# As there isn't any Exposure equal to zero, we can compute the real frequencies
base_train_freq = base_train
base_train_freq$Frequency = base_train$Nbclaims / base_train$Exposure
```

# Transform continous variables to factor using kmeans 
```{r}

# Use this to find the number of clusters
# kmean_withinss = function(k) {
#    cluster = kmeans(base_train$CarAge, k) # or DriverAge
#    return (cluster$tot.withinss)
# }
# max_k = 15
# wss = sapply(2:15, kmean_withinss) # then plot


# Use the mean of 50 centers to "cancel" variations
bootKMeans = (function(k1=6, k2=8){
  carMeans = matrix(NA, 50, k1)
  driverMeans = matrix(NA, 50, k2)
  
  for (i in 1:50){
    driverMeans[i, ] = sort(kmeans(base_train$DriverAge, k2)$centers)
    carMeans[i, ] = sort(kmeans(base_train$CarAge, k1)$centers)
  }
  
  return (
    c(colMeans(driverMeans),colMeans(carMeans))
  )
})()

driver_centers = bootKMeans[1:8]
car_centers = bootKMeans[9:12]

glm_train = within(base_train, {
  DriverAge = factor(kmeans(base_train$DriverAge, driver_centers)$cluster)
  CarAge = factor(kmeans(base_train$CarAge, car_centers)$cluster)
})

glm_test = within(base_test, {
  DriverAge = factor(kmeans(base_test$DriverAge, driver_centers)$cluster)
  CarAge = factor(kmeans(base_test$CarAge, car_centers)$cluster)
})
```

# Declare formulas

```{r}
formule.covariates = c("Gender", "DriverAge", "CarAge", "Area", "Leasing", "Power", "Fract", "Contract")

formule.long = as.formula(paste("Nbclaims ~ ", paste(formule.covariates, collapse=" + ")))
formule.long.offset = as.formula(
  paste("Nbclaims ~ offset(log(Exposure)) + ", paste(formule.covariates, collapse=" + "))
)
```

# Define our Deviance function 

```{r}
cost_deviance = function(y, fitt){
  left = y * log(y) - y * log(fitt)
  right = -y + fitt
  # Replace NA's by 0
  left[y==0] = 0 
  return(2*sum(left+right))
}
```

# Custom function to compare models faster

```{r}
models.comparison = data.frame()

addComparison = function(model, model_name){
  to_combine = data.frame(
      LogLik = logLik(model)[1],
      AIC = AIC(model),
      BIC = BIC(model),
      #Deviance = deviance(model),
      Deviance.custom = cost_deviance(base_train_freq$Frequency, model$fitted),
      # Percent.of.zero = round(mean(exp(-exp(predict(model))))*100,2),
      row.names = as.character(model_name)
  )
  rbind(
    models.comparison,
    to_combine
  )
}
```

### Generalized Linear Models

```{r}
# Naive
model.glm.poisson.1 = glm(formule.long.offset, data = base_train, family="poisson")
model.glm.poisson.2 = glm(formule.long.offset, data = glm_train, family="poisson") 

# Now we we glm_train
# Quasipoisson
model.glm.quasi.1 = glm(formule.long.offset, data = glm_train, family="quasipoisson")

# Negative-binomial
model.glm.nb.1 = glm(formule.long.offset, data = glm_train, family=negative.binomial(1))
model.glm.nb.2 = glm(formule.long.offset, data = glm_train, family=negative.binomial(4791))
model.glm.nb.3 = glm.nb(formule.long, data = glm_train)

# Zero-Inflated
model.glm.zero.1 = zeroinfl(formule.long, data=glm_train, dist="poisson", offset=log(Exposure))
model.glm.zero.2 = zeroinfl(formule.long, data=glm_train, dist="negbin", offset=log(Exposure))

# Hurdle
model.glm.hurdle.1 = hurdle(formule.long.offset, data=glm_train, dist="negbin", zero.dist = "poisson")
model.glm.hurdle.2 = hurdle(formule.long.offset, data=glm_train, dist="negbin", zero.dist = "binomial")

# Bagging 
model.glm.bagging = (function(B = 50){
  
  nr = nrow(base_train)
  nrTest = nrow(base_test)
  
  allPredicted = matrix(NA, B, nrTest)
  
  for (i in 1:B){
    
    samplePick = sample(nr, nr, T)
    sampleTrain = glm_train[samplePick,]
    
    sampleModel = glm(formule.long.offset, data = sampleTrain, family="poisson")
    
    allPredicted[i, ] = predict(sampleModel, base_test, type="response")
  }
  return (colMeans(allPredicted))
})()

# Tweedie

model.glm.tweedie.1 = glm(formule.long, data=base_train, family=tweedie(var.power=1, link.power = 0), offset = log(Exposure))
model.glm.tweedie.2 = glm(formule.long, data=glm_train, family=tweedie(var.power=1, link.power = 0), offset = log(Exposure))

# Mixed poisson with random


# Add to comparison

models.comparison = addComparison(model.glm.poisson.1, "Poisson 1")
models.comparison = addComparison(model.glm.poisson.2, "Poisson 2")

models.comparison = addComparison(model.glm.quasi.1, "Quasipoisson")

models.comparison = addComparison(model.glm.nb.1, "NB 1")
models.comparison = addComparison(model.glm.nb.2, "NB 4791")

models.comparison = addComparison(model.glm.zero.1, "Zero poisson")
models.comparison = addComparison(model.glm.zero.2, "Zero negbin")

models.comparison = addComparison(model.glm.tweedie.1, "Tweedie 1")
models.comparison = addComparison(model.glm.tweedie.2, "Tweedie 2")

```

#### Cross validation (by deviance)

```{r}
# https://stackoverflow.com/questions/44706961/output-of-cv-glm-vs-cv-glmnet
cost_deviance_overflow <- function(y, eta) {
  deveta = y * log(eta) - eta
  devy = y * log(y) - y
  devy[y == 0] = 0
  sum(2 * (devy - deveta))
}

model.glm.1.cv = cv.glm(base_train, model.glm.1, cost_deviance, K=10)$delta
model.glm.2.cv = cv.glm(base_train, model.glm.2, cost_deviance, K=10)
model.glm.3.cv = cv.glm(glm_train, model.glm.3, cost_deviance, K=10)$delta
model.glm.4.cv = cv.glm(glm_train, model.glm.4, cost_deviance, K=10)$delta

models.glm.cv = data.frame(model.glm.1.cv, model.glm.2.cv, model.glm.3.cv, model.glm.4.cv)
```

### --------------
### GAM

```{r}

model.gam.1 = gam(
    Nbclaims ~ s(DriverAge, bs="cr", by = Gender) + Power+Gender+Area+Leasing+Fract,
    data = base_train, 
    family=poisson(),
    offset = log(Exposure),
    method = "REML"
)

model.gam.2 = gam(Nbclaims ~ s(DriverAge, by=Gender) + s(CarAge) +Power+Gender+Area+Leasing+Fract,
    data = base_train, 
    family=poisson(),
    offset=log(Exposure),
    method = "REML"
)

model.gam.3 = gam(
    Nbclaims ~ s(DriverAge, bs="cr", k=6) + Power+Gender+Area+Leasing+Fract,
    data = base_train, 
    family=poisson(),
    offset = log(Exposure),
    method = "REML"
)

model.gam.4 = gam(
    Nbclaims ~ s(DriverAge, bs="cr") + Power+Gender+Area+Leasing+Fract + s(CarAge, bs="cr"),
    data = base_train, 
    family=poisson(),
    offset = log(Exposure),
    method = "REML"
)

model.gam.1.predicted = predict(model.gam.1, type="response")
model.gam.2.predicted = predict(model.gam.2, type="response")
model.gam.3.predicted = predict(model.gam.3, type="response")
model.gam.4.predicted = predict(model.gam.4, type="response")


plot(model.gam.3, rug = T, se = T, shade = T, shade.col = 'gray90')

models.comparison = addComparison(model.gam.1, "GAM 1")
models.comparison = addComparison(model.gam.2, "GAM 2")
models.comparison = addComparison(model.gam.3, "GAM 3")
models.comparison = addComparison(model.gam.4, "GAM 3")
```

#### Regression tree (RT)

```{r}
# Factors are ordered
model.tree.1 = rpart(
  cbind(Exposure, Nbclaims) ~ Gender + DriverAge + CarAge + Power + Area + Fract + Leasing, 
  data=base_train_ordered,
  method = "poisson",
  parms = list(shrink=4), # Method is deviance
  control = rpart.control(cp = 0.001, xval = 10, maxdepth = 20)
)

rpart.plot(model.tree.1, digits=2, type=4)

model.tree.1.predicted = predict(model.tree.1)

# Bagging
model.tree.2.predicted = (function(M = 20){
  nr = nrow(base_train) # size of the dataset
  nrTest = nrow(base_test)
  
  allPredict = matrix(0, M, nrTest)
  
  for (i in 1:M){
    
    samplePick = sample(nr, nr, T)
    sampleTrain = base_train[samplePick,]
    
    sampleTree = rpart(
      cbind(Exposure, Nbclaims) ~ Gender + DriverAge + CarAge + Power, 
      data=sampleTrain,
      method = "poisson",
      control = rpart.control(cp = 0.001, xval = 10)
    )
    allPredict[i,] = predict(sampleTree, base_test)
  }
  
  return (colMeans(allPredict))
})()
```

# Random Forest

```{r}


model.forest.1 = (function(M=100, C=3){
  # C is number of covariates picked
  nr = nrow(base_train)
  nrTest = nrow(base_test)
  sampleSize = nr
  
  
  allResults = matrix(NA, M, nrTest)
  for (i in 1:M){
    
    trainSample = sample(nr, sampleSize, replace=T)
    trainSample = base_train[trainSample,]
    
    covSample = sample(formule.covariates, C)
    equation = paste("cbind(Exposure, Nbclaims) ~ ", paste(covSample, collapse = " + "))
    
    tree = rpart(equation, data=trainSample, method="poisson", control=rpart.control(cp=0.0001, xval = 10))
    allResults[i,] = predict(tree, base_test)
  }
  
  return (colMeans(allResults))
})()

# This one is parametric (poisson)
model.forest.2 = (function(M=100, C=3){
  # C is number of covariates picked
  
  baseTree = model.tree.1 # See above
  
  nr = nrow(base_train)
  nrTest = nrow(base_test)
  sampleSize = nr
  
  
  allResults = matrix(NA, M, nrTest)
  for (i in 1:M){
    
    baseFit = predict(baseTree, base_train)
    baseFitNbclaims = rpois(nr, baseFit*base_train$Exposure)
    
    trainSample = base_train
    trainSample$Nbclaims = baseFitNbclaims # Replace by predicted
    
    covSample = sample(formule.covariates, C)
    equation = paste("cbind(Exposure, Nbclaims) ~ ", paste(covSample, collapse = " + "))
    
    tree = rpart(equation, data=trainSample, method="poisson", control=rpart.control(cp=0.001, xval = 10))
    allResults[i,] = predict(tree, base_test_ordered)
  }
  
  return (colMeans(allResults))
})()
```

# Gradient boosting model

```{r}
model.gbm.1 = gbm(formule.long.offset, data=base_train_ordered, distribution = "poisson", shrinkage = 0.001, n.trees = 2000, n.cores = 4, cv.folds = 3, train.fraction = 0.7)

model.gbm.2 = gbm(formule.long.offset, data=base_train_ordered, distribution = "poisson", shrinkage = 0.001, n.trees = 2000, n.cores = 4, cv.folds = 3, interaction.depth = 2)
```

# MOB Random forest

```{r}
library(mobForest)


model.mob = mobforest.analysis(formule.long.offset, data=base_train, family = poisson(), partition_vars = formule.covariates)

```

```{r}
rfPoisson_train = within(base_train, {
    
  Nbclaims = NULL
  Exposure = NULL
  Frequency = NULL
  
})

model.rfCount = rfPoisson(x = rfPoisson_train, y=base_train$Nbclaims, offset = log(base_train$Exposure),
                       ntree=100, nodesize=1, maxnodes=30, do.trace=T, replace = T)
```


