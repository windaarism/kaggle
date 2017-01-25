library(ggplot2)
library(caret)
pathData = 'D:/kaggle/titanic/logistic'
setwd(pathData)
dataa = read.csv('data.csv')
#melakukan standarisasi
dataa$Age<- (dataa$Age- min(dataa$Age))/(max(dataa$Age)-min(dataa$Age))
dataa$Pclass<- (dataa$Pclass- min(dataa$Pclass))/(max(dataa$Pclass)-min(dataa$Pclass))
dataa$SibSp<- (dataa$SibSp- min(dataa$SibSp))/(max(dataa$SibSp)-min(dataa$SibSp))
dataa$Fare<- (dataa$Fare- min(dataa$Fare))/(max(dataa$Fare)-min(dataa$Fare))
#definisikan untuk training dan testing
training_data <- dataa[1:889,1:7]
testing_data <- dataa[890:nrow(dataa),2:7]
output_testing <- data.frame(dataa[890:nrow(dataa),1])
glm = glm(Survived ~ Pclass + SibSp + Fare + Age + Sex+Embarked, family=binomial(logit), data=training_data)
summary(glm)
anova(glm,test="Chisq")
#cek multikolinear : korelasi variabel bebas
cor(Data[,c("Pclass","SibSp","Fare","Age")])
#prediksi
predicted<-round(predict(glm,type="response",newdata=testing_data))
predict<-data.frame(predicted)
predic<- cbind(output_testing,predict)

#cek akurasi dengan confusionmatrix
confusionMatrix(predic$predicted,predic$dataa.890.nrow.dataa...1.)

#akurasi 77.51 %
#ketika cabin masuk malah jadi turun 76