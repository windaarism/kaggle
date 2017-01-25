library(e1071)
library(caret)
pathData = 'D:/kaggle/titanic'
setwd(pathData)
Data = read.csv('trainclean.csv')
training_data <- Data[1:889,10:19]
testing_data <- Data[890:nrow(Data),11:19]
output_testing <- data.frame(Data[890:nrow(Data),10])
#tuning gamma and cost
svm_tune <- tune(svm, survive~.,data=training_data,kernel="radial", ranges=list(cost=10^(-1:2), gamma=c(.5,1,2)))
print(svm_tune)
#model svm dengan cost dan gamma hasil tuning
model <- svm(survive~.,data=training_data, kernel="radial", cost=1,gamma=1)
predicted <- predict(model,testing_data)
predic<- data.frame(predicted)
cek<- cbind(predic,output_testing)
confusionMatrix(cek$predicted,cek$Data.890.nrow.Data...10.)

#akurasi 78.4 %