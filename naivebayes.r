library(NLP)
library(SparseM)
library(MASS)
library(mlbench)
library(ggplot2)
library(caret)
library(e1071)
pathData = 'D:/kaggle/titanic/logistic'
setwd(pathData)
Data = read.csv('data.csv')
#melakukan standarisasi
dataa$Age<- (dataa$Age- min(dataa$Age))/(max(dataa$Age)-min(dataa$Age))
dataa$Pclass<- (dataa$Pclass- min(dataa$Pclass))/(max(dataa$Pclass)-min(dataa$Pclass))
dataa$SibSp<- (dataa$SibSp- min(dataa$SibSp))/(max(dataa$SibSp)-min(dataa$SibSp))
dataa$Fare<- (dataa$Fare- min(dataa$Fare))/(max(dataa$Fare)-min(dataa$Fare))
#definisikan untuk training dan testing(tanpa parch, cabin ,tiket, passengerid)
training_data <- dataa[1:889,1:7]
testing_data <- dataa[890:nrow(dataa),2:7]
output_testing <- data.frame(dataa[890:nrow(dataa),1])
model <- naiveBayes(Survived ~ ., data = training_data)
predic<-predict(model, testing_data)
predict<-data.frame(predicted)
cek<- cbind(output_testing,predict)
#cek akurasi
confusionMatrix(cek$predicted,cek$dataa.890.nrow.dataa...1.)

#akurasi 77.51%