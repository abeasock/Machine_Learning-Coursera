# Machine Learning: Prediction Assignment
Amber Beasock  
31 January 2016  

----------------------------------------------------------------------------------

### Project Overview

#### Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website [here](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset).

#### Data

The training data for this project are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:
<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>

The data for this project come from this source: <http://groupware.les.inf.puc-rio.br/har>. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.


The `classe` variable contains 5 different ways barbell lifts were performed correctly and incorrectly:

- Class A: exactly according to the specification
- Class B: throwing the elbows to the front
- Class C: lifting the dumbbell only halfway
- Class D: lowering the dumbbell only halfway
- Class E: throwing the hips to the front

#### Objective

The goal of your project is to predict the manner in which they did the exercise. This is the `classe` variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

----------------------------------------------------------------------------------

### Loading the data

Packages used for analysis. This assumes the packages are already installed. Use the `install.packages("")` command if a package not installed yet.

```r
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(rattle)
```

Load the data into R

```r
# The location where the training data is to be downloaded from
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
# The location where the testing data is to be downloaded from
test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

#Before downloading the data, you can change your working directory by setting the path in setwd()
# Download the training data in your working directory, if it hasn't been already
if (!file.exists("train_data.csv")){
  download.file(train_url, destfile="train_data.csv", method="curl")
}
# Download the testing data in your working directory
if (!file.exists("test_data.csv")){
download.file(test_url, destfile="test_data.csv", method="curl")
}

# Read the Training CSV file into R & replace missing values & excel division error strings #DIV/0! with 'NA'
train_data <- read.csv("train_data.csv", na.strings=c("NA","#DIV/0!",""), header=TRUE)

# Read the Testing CSV file into R & replace missing values & excel division error strings #DIV/0! with 'NA'
test_data <- read.csv("test_data.csv", na.strings=c("NA","#DIV/0!",""), header=TRUE)

# Take a look at the Training data classe variable
summary(train_data$classe)
```

```
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

### Partitioning the data for Cross-validation

The training data is split into two data sets, one for training the model and one for testing the performance of our model. The data is partitioned by the `classe` variable, which is the varible we will be predicting. The data is split into 60% for training and 40% for testing.


```r
inTrain <- createDataPartition(y=train_data$classe, p = 0.60, list=FALSE)
training <- train_data[inTrain,]
testing <- train_data[-inTrain,]

dim(training); dim(testing)
```

```
## [1] 11776   160
```

```
## [1] 7846  160
```

### Data Processing
Drop the first 7 variables because these are made up of metadata that would cause the model to perform poorly.

```r
training <- training[,-c(1:7)]
```

Remove NearZeroVariance variables

```r
nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[, nzv$nzv==FALSE]
```

There are a lot of variables where most of the values are 'NA'. Drop variables that have 60% or more of the values as 'NA'.

```r
training_clean <- training
for(i in 1:length(training)) {
  if( sum( is.na( training[, i] ) ) /nrow(training) >= .6) {
    for(j in 1:length(training_clean)) {
      if( length( grep(names(training[i]), names(training_clean)[j]) ) == 1)  {
        training_clean <- training_clean[ , -j]
      }   
    } 
  }
}

# Set the new cleaned up dataset back to the old dataset name
training <- training_clean
```

Transform the test_data dataset

```r
# Get the column names in the training dataset
columns <- colnames(training)
# Drop the class variable
columns2 <- colnames(training[, -53])
# Subset the test data on the variables that are in the training data set
test_data <- test_data[columns2]
dim(test_data)
```

```
## [1] 20 52
```

### Cross-Validation: Prediction with Random Forest

A Random Forest model is built on the training set. Then the results are evaluated on the test set

```r
set.seed(54321)
modFit <- randomForest(classe ~ ., data=training)
prediction <- predict(modFit, testing)
cm <- confusionMatrix(prediction, testing$classe)
print(cm)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2230   18    0    0    0
##          B    1 1491   11    0    0
##          C    0    9 1349   15    0
##          D    0    0    8 1270    4
##          E    1    0    0    1 1438
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9913         
##                  95% CI : (0.989, 0.9933)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.989          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9822   0.9861   0.9876   0.9972
## Specificity            0.9968   0.9981   0.9963   0.9982   0.9997
## Pos Pred Value         0.9920   0.9920   0.9825   0.9906   0.9986
## Neg Pred Value         0.9996   0.9957   0.9971   0.9976   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2842   0.1900   0.1719   0.1619   0.1833
## Detection Prevalence   0.2865   0.1916   0.1750   0.1634   0.1835
## Balanced Accuracy      0.9979   0.9902   0.9912   0.9929   0.9985
```


```r
overall.accuracy <- round(cm$overall['Accuracy'] * 100, 2)
sam.err <- round(1 - cm$overall['Accuracy'],2)
```

The model is 99.13% accurate on the testing data partitioned from the training data. The expected out of sample error is roughly 0.01%. 


```r
plot(modFit)
```

![](./prediction_assignment_files/figure-html/unnamed-chunk-10-1.png) 

In the above figure, error rates of the model are plotted over 500 trees. The error rate is less than 0.04 for all 5 classe. 

### Cross-Validation: Prediction with a Decision Tree


```r
set.seed(54321)
modFit2 <- rpart(classe ~ ., data=training, method="class")
prediction2 <- predict(modFit2, testing, type="class")
cm2 <- confusionMatrix(prediction2, testing$classe)
print(cm2)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2026  252   85  138   37
##          B   77  952   73  108  114
##          C   55  115 1032  200  171
##          D   42  105   96  690   68
##          E   32   94   82  150 1052
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7331          
##                  95% CI : (0.7232, 0.7429)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6606          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9077   0.6271   0.7544  0.53655   0.7295
## Specificity            0.9088   0.9412   0.9165  0.95259   0.9441
## Pos Pred Value         0.7983   0.7190   0.6561  0.68931   0.7461
## Neg Pred Value         0.9612   0.9132   0.9464  0.91293   0.9394
## Prevalence             0.2845   0.1935   0.1744  0.16391   0.1838
## Detection Rate         0.2582   0.1213   0.1315  0.08794   0.1341
## Detection Prevalence   0.3235   0.1687   0.2005  0.12758   0.1797
## Balanced Accuracy      0.9083   0.7842   0.8354  0.74457   0.8368
```


```r
overall.accuracy2 <- round(cm2$overall['Accuracy'] * 100, 2)
sam.err2 <- round(1 - cm2$overall['Accuracy'],2)
```

The model is 73.31% accurate on the testing data partitioned from the training data. The expected out of sample error is roughly 0.27%. 

Plot the decision tree model

```r
fancyRpartPlot(modFit2)
```

![](./prediction_assignment_files/figure-html/unnamed-chunk-13-1.png) 

### Prediction on the Test Data

The Random Forest model gave an accuracy of 99.13%, which is much higher than the 73.31% accuracy from the Decision Tree. So we will use the Random Forest model to make the predictions on the test data to predict the way 20 participates performed the exercise.

```r
final_prediction <- predict(modFit, test_data, type="class")
print(final_prediction)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

### Conclusion

There are many different machine learning algorithms. I chose to compare a Random Forest and Decision Tree model. For this data, the Random Forest proved to be a more accurate way to predict the manner in which the exercise was done.
