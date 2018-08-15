## Classification

Classification algorithms, as the name suggests, are used for classifying values into groups. We usually deal with binary values when it comes to classification.<br>
It's more like a Yes or No answer that you get from these algorithms.


## Linear Regression

Linear Regression is a basic and one of the most commonly used type of predictive analysis. The overall idea here is to basically determine (or more precise predict) a something. In this algorithm, you have 1 dependent variable and 1 or more independent variables.<br>
If you have more than one independent variables, we call it multiple linear regression.<br>
The simplest form of the regression equation with one dependent and one independent variable is `y = m*x + c ` where y is the estimated or predicted variable score (dependent variable), x is the score on independent variable , c = constant and m is the regression constant.<br>


## Decision Tree

Belongs to the family of supervised learning. It can be used for solving both, classification and regression problems.<br>
The basic concept of using decision trees is that it creates a training model which can be used to predict a class or value of dependent variables by learning decision rules that it gets from the training dataset.<br>
It solves a particular problem by using the tree representations wherein each internal node of the tree is an attribute and each leaf node corresponds to a class label.<br>
Works with both categorical and continuous data.<br>



## Bagging

Bagging (a.k.a Bootstrap Aggregation) is a simple yet powerful ensemble method. A technique that combines the predictions from multiple machine learning algorithms together to make more accurate predictions than any individual models is called an Ensemble Method.<br>
It basically splits the sample 25 times and does a regression everytime and adds that to the list of predictions.<br>
Bagging can be used in problems where the algorithms have high variance(decision trees, classification, regression) in order to reduce the variance.<br>


## Boosting

Boosting is an ensembling technique, which means that prediction is done by an ensemble of simpler estimators.<br>
The aim of gradient boosting is to create (or "train") an ensemble of trees, given that we know how to train a single decision.<br>
Types of boosting :
1) AdaBoosting<br>
2) Gradient Tree Boosting<br>
3) XGBoosting<br>
Boosting gives tells us what are the important independent variables for a model.<br>


## Random Forest Classifier

It's a supervised learning algorithm. As the name suggests, it creates a forest(involves deceision trees. For more info on that click <a href="">here</a>) and the createdd forest is basically random. <br>
Random Forest builds multiple decision trees and merges them together to get a more accurate and stable prediction. After, the decision trees are created, the Bagging method is used to train them to increase the accuracy of the dependent(or predicted) variable.<br>
Advantage of RF is that it can be used for both regression and classification and also tells us about the what independent variables are more useful to determine the predictions.

## KMeans

k-means clustering aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean, serving as a prototype of the cluster.<br>
It's a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups).<br> The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K.<br>
The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. 



## Logistic Regression

Part of generalised linear model. On using this, residuals must follow normal distribution curve.<br> People often get confused between classification and logistic regression.<br>
In classification, the final values are either 0 or 1.<br>
In Logistic Regression, we look at getting values between 2 continuous variables.<br>


## K Nearest Neighbours

KNN is a non-parametric, lazy learning algorithm.<br>
Its purpose is to use a database in which the data points are separated into several classes to predict the classification of a new sample point.<br>
When we say a technique is non-parametric , it means that it does not make any assumptions on the underlying data distribution.<br>
To locate nearest neighbours we need to know a distance function (usually we use euclidean distance). In KNN, K is very important.<br>
If K=1, we compare with a single thing.<br>
If K=n, we compare with n things.<br>
An important thing to note about KNN is that we should remove unwanted columns from dataset, and the variable to be predicted is factorized. Also, it works only in dataframes.<br>













