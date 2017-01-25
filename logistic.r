pathData = 'D:/kaggle/titanic/logistic'
setwd(pathData)
Data = read.csv('data.csv')
training_data <- Data[1:889,1:7]
output_training <- Data[1:889,7]
testing_data <- Data[890:nrow(Data),2:7]
output_testing <- data.frame(Data[890:nrow(Data),1])
glm = glm(Survived ~ cla + Sibl + Farre + Age + Sex+Embarked, family=binomial(logit), data=training_data)
summary(glm)
anova(glm,test="Chisq")
#cek multikolinear : korelasi variabel bebas
cor(Data[,c("Pclass","SibSp","Fare","Age")])
predicted<-round(predict(glm,type="response",newdata=testing_data))
predict<-data.frame(predicted)
predic<- cbind(output_testing,predict)

#cek akurasi dengan confusionmatrix
library(lattice)
library(ggplot2)
library(caret)
confusionMatrix(predic$predicted,predic$Data.890.nrow.Data...1.)
write.csv(predict,'test2.csv')
View(predict)

#akurasi 77.51 %