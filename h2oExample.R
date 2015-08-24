library(h2o); library(caret)
trainIndex <- createDataPartition(iris$Species, p=.75, list=FALSE)
iris.train <- iris[trainIndex, ]
iris.test <- iris[-trainIndex, ]

localH2O = h2o.init(nthreads = 8)
iris.train.h2o <- as.h2o(iris.train, localH2O, destination_frame='iris.h2o')
iris.test.h2o <- as.h2o(iris.test, localH2O)
model = h2o.deeplearning(x = colnames(iris)[-ncol(iris)],
                         y = "Species",
                         training_frame = iris.h2o,
                         activation = "Tanh",
                         hidden = c(10, 10, 10),
                         epochs = 10000)
predictions = predict(object = model, newdata = iris.test.h2o)

# Export predictions from H2O Cluster as R dataframe
predictions.R = as.data.frame(predictions)
head(predictions.R)
tail(predictions.R)

# Check performance of the classification model
performance = h2o.performance(model = model, data=iris.test.h2o)
print(performance)
h2o.shutdown(localH2O)