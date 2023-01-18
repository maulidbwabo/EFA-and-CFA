##Machine learning approaches to predict the effect of gender on dynamic capability
## drivers on sustainable performance
##Ensemble methods
library(ggplot2)
library(cowplot)
library(data.table)
library(viridisLite)
library(viridis)
install.packages("RSNNS")
library(Rcpp)
library(RSNNS)
install.packages("kernlab")
library(kernlab)
library(rpart)
install.packages("rattle")
install.packages("tibble")
install.packages("bitops")
library(tibble)
library(bitops)
library(rattle)
install.packages("DALEX")
library(DALEX)
library(lattice)
library(ggplot2)
library(caret)
install.packages("sp")
install.packages("spData")
install.packages("sf")
library(sp)
library(spData)
library(sf)
install.packages("spdep")
library(spdep)
install.packages("ranger")
library(ranger)
library(e1071)
install.packages("gbm")
library(gbm)
library(plyr)
set.seed(1234)
library(data.table)
###
head(New_Data1)
tail(New_Data1)
##
##Principal components analysis(Exploratory Data Analysis)
New_Data1 = as.data.table(New_Data1)
dynamicP=subset(dynamic,select = -c(1:14))
str(New_Data1)
New_Data1$Education.Level=as.factor(New_Data1$Education.Level)
New_Data1$Size=as.factor(New_Data1$Size)
New_Data1$Sex=as.factor(New_Data1$Sex)
New_Data1$Occupancy=as.factor(New_Data1$Occupancy)
str(New_Data1)
sort(unique(d$Sex))
d = copy(New_Data1)
str(d)
##
#Hot coding or dummy coding to the character variable(sex)
ddum = dummyVars("~.", data = d)
d= data.table(predict(ddum, newdata = d))
str(d)
remove(ddum)
##
dscaled=scale(d)
str(d)
dscaled=as.data.table(dscaled)
d=cbind(d[,c(1:3)],dscaled)
##
#Transformation (Evaluating if the data have a bunches of outliers)
boxplot(d, las = 2)
##Shapiro Test
par(mfrow = c(1,2))
hist(d$KS3, 100)
qqnorm(d$KS3)
par(mfrow = c(1,1))
shapiro.test(d$KS3)
range(d$KS3)
range(d$KS3)
shapiro.test(log(d$KS3))
##
#Training and Validation of the data
set.seed(1234)
index = createDataPartition(New_Data1$Sex, p = 0.8, list = FALSE)
trainData = New_Data1[index, ]
validationData = New_Data1[-index, ]
##PCA
#confirm structure
str(trainData)
View(trainData)
#base R / traditional method
pc = prcomp(trainData[,c(1:40)], center = TRUE, scale. = TRUE)
summary(pc)
pcValidationData1= predict(pc, newdata = validationData[,c(1:40)])
#scalable method using PCA Methods
install.packages("FactoMineR")
library(FactoMineR)
install.packages("factoextra")
library(factoextra)
biocLite("pcaMethods")
##Bioc
if (!require("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
BiocManager::install()
BiocManager::available()
library(BiocManager)
BiocManager::install("pcaMethods")
library(pcaMethods)
##
pc=pca(trainData[,c(1:40)], method = "svd",nPcs = 4, scale = "uv", 
       center = TRUE)
pc
summary(pc)
##
##Support Vector Machines
svmDataTrain = trainData[,c(1:40)]
svmDataValidate = validationData[,c(1:40)]
##
##SVM
set.seed(12345)
svm1 = train(x = svmDataTrain,
            y = trainData$Sex,
            method = "svmLinear",
            preProcess = NULL,
            metric = "Accuracy",
            trControl = trainControl(method = "cv",
                                     number = 5,
                                     seeds = c(123, 234, 345, 456, 567, 678)
            )
)
svm1
##
#predict the country name on our training data using our new model
predictOnTrain = predict(svm1, newdata = svmDataTrain)
mean( predictOnTrain == trainData$Sex)

##
set.seed(12345)
svmLinear1 = train(x = svmDataTrain,
                  y = trainData$Sex,
                  method = "svmLinear",
                  preProcess = c("scale", "center", "pca"),
                  metric = "Accuracy",
                  trControl = trainControl(method = "cv",
                                           number = 4,
                                           seeds = c(123, 234, 345, 456, 567, 678)
                  )
)
svmLinear1
##Polynomial 
set.seed(12345)
svmPoly <- train(x = svmDataTrain,
                 y = trainData$Sex,
                 method = "svmPoly",
                 preProcess = c("scale", "center", "pca"),
                 metric = "Accuracy",
                 trControl = trainControl(method = "cv",
                                          number = 5
                 )
)

svmPoly
##Training 
predictOnTrainP <- predict(svmPoly, newdata = svmDataTrain)
mean( predictOnTrainP == trainData$Sex)
predictOnTestL <- predict(svmLinear1, newdata = svmDataValidate)
mean(predictOnTestL == trainData$Sex)
##CLASSIFICATION REGRESSION TREE
cartDataTrain = copy(trainData)
cartDataValidation = copy(svmDataValidate)
##
cartModel1 <- train(x = svmDataTrain,
                   y = trainData$Sex,
                   method = "rpart",
                   preProcess = c("scale", "center", "pca"),
                   metric = "Accuracy",
                   tuneLength = 10,
                   trControl = trainControl(method = "cv",
                                            number = 5
                   )
)
cartModel1
summary(cartModel1)
##
plot(cartModel1$finalModel)
text(cartModel1$finalModel, cex = 0.5)
##
fancyRpartPlot(cartModel1$finalModel, cex = 0.4, main = "")
##
predictOnTrainT = predict(cartModel1, newdata = trainData)
mean( predictOnTrainT == trainData$Sex)
##
predictOnTestT = predict(cartModel1, newdata = trainData)
mean(predictOnTestT == cartDataTrain$Sex)
##Confusion Matrix 
confusionMatrix(predictOnTestT, as.factor(trainData$Sex))
##Random Forest 
rfDataTrain = copy(trainData)
rfDataValidation = copy(svmDataValidate)
##
set.seed(12345)
rfModel1 = train(x = svmDataTrain,
                 y = trainData$Sex,
                 method = "ranger",
                 preProcess = c("scale", "center", "pca"),
                 metric = "Kappa",
                 num.trees = 50,
                 trControl = trainControl(method = "cv",
                                          number = 5
                 )
)


rfModel
##Random forest 
set.seed(222)
ind <- sample(2, nrow(trainData), replace = TRUE, prob = c(0.7, 0.3))
train <- trainData[ind==1,]
test <- trainData[ind==2,]
str(train)
##
library(caret)
library(randomForest)
library(rattle)
rf = randomForest(Sex~., data=train, proximity=TRUE)
rf
##
p1 <- predict(rf, train)
confusionMatrix(p1, train$Sex)
##
p2 <- predict(rf, test)
confusionMatrix(p2, test$Sex)
##
plot(rf)
#variable of importance 
hist(treesize(rf),
     main = "No. of Nodes for the Trees",
     col = "green")
"Variable Importance"
varImpPlot(rf,
           sort = T,
           n.var = 10,
           main = "Top 10 - Variable Importance")
importance(rf)
MeanDecreaseGini
##
partialPlot(rf, train, Sex, "Female")
##
MDSplot(rf, train$Sex)
## Library C50
BiocManager::install("C50")
library(C50)
C5tree = C5.0(Sex~., data=train)
C5tree
summary(C5tree)
##Stochastic Gradient Boosting (GBM)
BiocManager::install("gbm")
library(gbm)
##
sgbDataTrain = copy(trainData)
sgbDataTrain1 = copy(trainData)
sgbDataTrain1
ddum2 = dummyVars("~.", data = sgbDataTrain1)
ddum2
head(ddum2)
str(sgbDataTrain1)
sgbDataTrain = data.table(predict(ddum2, newdata = sgbDataTrain))
sgbDataValidation = data.table(predict(ddum2, newdata = sgbDataValidation))
sgbDataTrain[,Sex:=as.numeric(Sex)]
sgbDataTrain[,Occupancy:=as.numeric(Occupancy)]
sgbDataTrain[,Size:=as.numeric(Size)]
sgbDataTrain[,Education.Level:=as.numeric(Education.Level)]
str(sgbDataTrain)
view(sgbDataTrain)
sgbDataValidation = copy(validationData)
##Dummiffication 
sgbDataTrain = cbind(sgbDataTrain, dummy(sgbDataTrain,sep="_"))
ddum1 = dummyVars("~.", data = sgbDataTrain)
sgbDataTrain = data.table(predict(ddum1, newdata = sgbDataTrain))
sgbDataValidation = data.table(predict(ddum1, newdata = sgbDataValidation))
rm(ddum)
str(sgbDataTrain)
##SGM Model
sgbModel <- train(Sex ~.,
                  data = sgbDataTrain,
                  method = "gbm",
                  preProcess = c("scale", "center"),
                  metric = "RMSE",
                  trControl = trainControl(method = "cv",
                                           number = 5
                  ),
                  tuneGrid = expand.grid(interaction.depth = 1:3,
                                         shrinkage = 0.1,
                                         n.trees = c(50, 100, 150),
                                         n.minobsinnode = 10),
                  verbose = FALSE
)
sgbModel
summary(sgbModel)
##Model Accuracy 
mean(stats::residuals(sgbModel)^2)
##
mean((predict(sgbModel, sgbDataValidation) -
        sgbDataValidation$TC4)^2)
##
explainSGBt = explain(sgbModel, label = "sgbt",
                       data = sgbDataTrain,
                       y = sgbDataTrain$Sex)
##
explainSGBv = explain(sgbModel, label = "sgbv",
                       data = sgbDataValidation,
                       y = sgbDataTrain$Sex)

##Graphs 
performanceSGBt = model_performance(explainSGBt)
performanceSGBv = model_performance(explainSGBv)
plot_grid(
  plot(performanceSGBt, performanceSGBv),
  plot(performanceSGBt, performanceSGBv, geom = "boxplot"),
  ncol = 2)
##
forGBM = trainData
forGBM$Sex=ifelse(forGBM$Sex=="successful",1,0)
str(forGBM)
head(forGBM)
##
gbmModel1 = gbm(Sex~., data=forGBM,
                 distribution = "bernoulli",
                 interaction.depth = 9,
                 n.trees = 1400,
                 shrinkage = 0.01,
              ## The function produces copious amounts
              ## of output by default.
             verbose = FALSE)
gbmModel1
##
predict(gbmModel1, type = "response")
##Prediction
gbmPred1 = predict(gbmModel1,newdata = head(trainData),
                   type = "response",
                   ## The number of trees must be
                   ## explicitly set
                   n.trees = 1400)
gbmPred1
##xGBoost
library(caret)
BiocManager::install("xgboost")
library(xgboost)
#Xgboost model
modelxg <- train(Sex~., data=train, method ="xgbTree",  trControl =
    trainControl("cv", number =  10))

modelxg
##Training set 
Xg1= predict(modelxg, train)
confusionMatrix(Xg1, train$Sex)
##Testing set 
Xg1T= predict(modelxg, test)
confusionMatrix(p2, test$Sex)
# Best tuning parameter
modelxg$bestTune
##Model Prediction accuracy(Trainning set and Test set)
predictOnTrainXG = predict(modelxg, newdata = train)
mean( predictOnTrainXG == trainData$Sex)
##Testing set 
predictOnTestXG = predict(modelxg, newdata = test)
mean( predictOnTestXG == test$Sex)
##Evaluation
##ROC Curve
library(pROC)
# Load data
library(usethis)
install.packages("devtools")
library(devtools)
devtools::install_github("rstudio/keras")
library(keras)        
install_keras()  
fashion_mnist = keras::dataset_fashion_mnist()
##Example 
set.seed(123)
model <- train(trainData$Sex ~., data =svmDataTrain, method ="svmLinear"
  ,trControl =trainControl("cv", number =10),  tuneGrid =expand.grid
  (C = seq(0, 2, length =20)),  preProcess =c("center","scale" ))# Plot model accuracy vs different values of Cost

plot(model)
