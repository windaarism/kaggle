library(NLP)
library(SparseM)
library(MASS)
library(mlbench)
library(e1071)
pathData = 'D:/kaggle/titanic/logistic'
setwd(pathData)
Data = read.csv('data.csv')
training_data <- Data[1:889,1:7]
output_training <- Data[1:889,7]
testing_data <- Data[890:nrow(Data),2:7]
output_testing <- data.frame(Data[890:nrow(Data),1])
model <- naiveBayes(Survived ~ ., data = training_data)
predic<-predict(model, testing_data)
predict<-data.frame(predicted)
cek<- cbind(output_testing,predict)


library(lattice)
library(ggplot2)
library(caret)
confusionMatrix(cek$predicted,cek$Data.890.nrow.Data...1.)
View(output_testing)

#akurasi 77.51%