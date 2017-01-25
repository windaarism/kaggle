library(e1071)
library(ggplot2)
library(caret)
pathData = 'D:/kaggle/titanic/logistic'
setwd(pathData)
Data = read.csv('data.csv')
dataa<- Data[,1:7]
#melakukan standarisasi
dataa$Age<- (dataa$Age- min(dataa$Age))/(max(dataa$Age)-min(dataa$Age))
dataa$Pclass<- (dataa$Pclass- min(dataa$Pclass))/(max(dataa$Pclass)-min(dataa$Pclass))
dataa$SibSp<- (dataa$SibSp- min(dataa$SibSp))/(max(dataa$SibSp)-min(dataa$SibSp))
dataa$Fare<- (dataa$Fare- min(dataa$Fare))/(max(dataa$Fare)-min(dataa$Fare))
#menambah variabel dummy pada data kategorik
dataa$Embarked<- class.ind(dataa$Embarked)
dataa$Sex<- class.ind(dataa$Sex)
dataclean<- as.matrix(dataa)
#definisikan data training dan testing
training_data <- dataclean[1:889,]
testing_data <- dataclean[890:nrow(dataclean),]
output_testing <- data.frame(dataclean[890:nrow(dataclean),1])
#tuning gamma and cost
svm_tune <- tune(svm, Survived~.,data=training_data,kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)
#model svm dengan cost dan gamma hasil tuning
model <- svm(Survived~.,data=training_data, kernel="radial", cost=1,gamma=1)
predicted <- predict(model,testing_data,type='raw')
predicted<- round(predicted)
predic<- data.frame(predicted)
cek<- cbind(predic,output_testing)
confusionMatrix(cek$predicted,cek$dataclean.890.nrow.dataclean...1.)

#akurasi 78.4 %