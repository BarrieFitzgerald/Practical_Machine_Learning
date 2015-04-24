## Working directory
setwd("C:/Users/bdfitzgerald/Desktop/data science competitions")

## libraries used
library(caret); library(rpart)
library(rattle); library(rpart.plot)
library(randomForest); library(RColorBrewer)
library(corrplot)

## importing in the training data file
train <- read.csv("./pml-training.csv")
dim(train)

## finding the number of complete cases for analysis
sum(complete.cases(train))

## column clean up function
cleanup <- function (x) {
        ## saves training data as an object
        data <- x
        ## pulls out the grades or "classe"
        grades <- as.data.frame(data[, max(length(data))])
        colnames(grades) <- "classe"
        ## filtering down to complete objects
        data <- data[, colSums(is.na(data)) == FALSE]
        ## function to remove unneeded columns
        remove <- function (x) {
                remove <- grepl("^X|timestamp|window", names(x))
        }
        ## applying the funciton
        data <- data[,!remove(data)]
        ## converting columns to numeric values
        data <- data[, sapply(data, is.numeric)]
        ## adding the grades or "classe" values back in
        data <- cbind(data, grades)
}

## applying the cleanup function
train <- cleanup(train)

## setting the seed for reproducible purposes
set.seed(2015)
## creating training and cross validation data sets
inTrain <- createDataPartition(train$classe, 
                               p = 0.60,
                               list = FALSE)
        ## 60% of data goes into train data
train <- train[inTrain, ]
        ## remaining 40% goes into cross validation data
crossvalid <- train[-inTrain, ]

## random forest model to predict the outcomes
model <- train(classe ~ ., data = train, method = "rf", 
               trControl = trainControl(method = "cv", 5), 
               ntree = 250)
model
prp(model)

## predictions on the cross validation data
predictions <- predict(model, crossvalid)
confusionMatrix(crossvalid$classe, predictions)

## checking the accuracy of the model
model.accuracy <- postResample(predictions, crossvalid$classe)
model.accuracy

## out of sample error
samp.error <- 1 - 
        as.numeric(confusionMatrix(
                crossvalid$classe, 
                predictions)$overall[1])
samp.error

## importing the test data
test <- read.csv("./pml-testing.csv")
dim(test)

## clean up function is modify to remove the empty column
        ## where the classe was in the training data
cleanup <- function (x) {
        ## saves training data as an object
        data <- x
        ## pulls out the grades or "classe"
        data <- data[, -max(length(data))]
        ## filtering down to complete objects
        data <- data[, colSums(is.na(data)) == 0]
        ## function to remove unneeded columns
        remove <- function (x) {
                remove <- grepl("^X|timestamp|window", names(x))
        }
        ## applying the funciton
        data <- data[,!remove(data)]
        ## converting columns to numeric values
        data <- data[, sapply(data, is.numeric)]
}
 
## applying the revised clean up function to the test data
test <- cleanup(test)
test$prediction <- predict(model, test)
test$prediction

## Correlation matrix of the data
## plot <- featurePlot(x = train, y = train$classe, plot = "pairs")
corrPlot <- cor(train[,-max(length(train))])
corrplot(corrPlot, method="color")

## Decision Tree Plot
tree <- rpart(classe ~ ., data = train, method = "class") 
fancytree <- fancyRpartPlot(tree)
basictree <- prp(tree)

## prediction out puts
pml_write_files = function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
        }
}

pml_write_files(test$prediction)

## Knitting function
library(knitr)

build.report <- function(x) {
        knit2html(x, "project.html")
}

