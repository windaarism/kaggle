library(NLP)
library(tm)
library(caret)
library(ggplot2)
library(nnet)
library(maxent)
pathData = 'D:/kaggle/titanic/logistic'
setwd(pathData)
Data = read.csv('data.csv')
#data tanpa parch, survived, passengerID dan cabin serta tiket,
dataa <- Data[,2:7]
#melakukan standarisasi
dataa$Age<- (dataa$Age- min(dataa$Age))/(max(dataa$Age)-min(dataa$Age))
dataa$Pclass<- (dataa$Pclass- min(dataa$Pclass))/(max(dataa$Pclass)-min(dataa$Pclass))
dataa$SibSp<- (dataa$SibSp- min(dataa$SibSp))/(max(dataa$SibSp)-min(dataa$SibSp))
dataa$Fare<- (dataa$Fare- min(dataa$Fare))/(max(dataa$Fare)-min(dataa$Fare))
#menambah variabel dummy pada data kategorik
dataa$Embarked<- class.ind(dataa$Embarked)
dataa$Sex<- class.ind(dataa$Sex)
training<- as.matrix(dataa)

output_testing <- data.frame(Data[890:nrow(Data),1])
#Maxent
model <- maxent(training[1:889,],as.factor(Data$Survived)[1:889])
results <- predict(model,training[890:nrow(data),])
cek<- cbind(results,output_testing)

#cek akurasi
confusionMatrix(cek$labels,cek$Data.890.nrow.Data...1.)
#hasil akurasi 77.51%