pathData = 'D:/kaggle/mnist'
setwd(pathData)
Data = read.csv('full.csv')

#buang nol
clean<-Data[, colSums(abs(Data)) != 0]
library(Matrix)
sparse_matrix <- sparse.model.matrix(label ~ .-1, data = clean)

output_vector = clean[1:42000,1] 

library(xgboost)
library(readr)
library(stringr)
library(caret)
library(car)

# definisikan train dan test
df_train <- sparse_matrix[1:42000,]
df_test <- sparse_matrix[42001:nrow(clean),]



#bangun model
xgb <- xgboost(data = data.matrix(df_train), 
 label = output_vector, 
 eta = 0.1,
 max_depth = 15, 
 nround=25, 
 subsample = 0.5,
 colsample_bytree = 0.5,
 seed = 1,
 eval_metric = "merror",
 objective = "multi:softmax",
 num_class = 10,
 nthread = 3
)


# predict values in test set
y_pred <- predict(xgb, data.matrix(df_test))
write.csv(y_pred,'xgboost2.csv')