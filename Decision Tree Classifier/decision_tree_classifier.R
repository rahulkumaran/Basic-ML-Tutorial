library(tree)
library(caTools)

student <- read.csv("student.csv")

set.seed(1)	#similar to random state in python to select data for test and train

split <- sample.split(student$Grade, SplitRatio = 0.70)	#Helps in splitting data

studentTrain <- subset(student, split == TRUE)
studentTest <- subset(student, split == FALSE)

print(head(student))

table(student$Grade)

table(studentTrain$Grade)

table(studentTest$Grade)

prop.table(table(studentTest$Grade))	#Helps in understanding prop of something in train and test data set

prop.table(table(studentTrain$Grade))	#The test and train proportions should be similar only then you can perform analysis on these

# If proportion is not same, then we must do resampling again

#modelClassTree <- tree(Grade ~ Motivation + Age + Gender, data = studentTrain)

model <- tree(Grade~., data=studentTrain)	#. means takes all vars and chooses only ones needed
plot(model)		#plots the DCT
text(model)		#PLOTS WITH TEXT

pred <- predict(model, newdata = studentTest, type = "class")	#Predicting the value

pred 
conf <- table(studentTest$Grade, pred)	#Finding the matrix just like confusion matrix in python

conf

OAA <- (conf[1,1] + conf[2,2] + conf[3,3] + conf[4,4] + conf[5,5] + conf[6,6])/sum(conf)	#Sum of diagonals/ total sum of matrix

OAA

summary(model)	#gives summary of model
model		#Gives the way the DCT is divided with node values
