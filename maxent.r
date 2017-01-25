library(NLP)
library(tm)
library(caret)
library(maxent)
pathData = 'D:/kaggle/titanic/logistic'
setwd(pathData)
Data = read.csv('data.csv')
data <- cbind(Data$Age,Data$Embarked,Data$Sex,Data$cla,Data$Farre,Data$Sibl)
data <- cbind(Data$Embarked,Data$Sex)
class<- as.factor(Data$Survived[1:889])
output_testing <- data.frame(Data[890:nrow(Data),1])
model <- maxent(data[1:889,],as.factor(Data$Survived)[1:889])
results <- predict(model,data[890:nrow(data),])
library(lattice)
library(ggplot2)
library(caret)
confusionMatrix(cek$labels,cek$Data.890.nrow.Data...1.)


#namun ketika dimasukkan faktor age, sex,Pclass, Fare, SibSl akurasi berkurang menjadi 71.05%