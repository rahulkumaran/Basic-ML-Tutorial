library(MASS)		#Has many datasets
library(descr)		#Has lot of important functions like freq
library(moments)	#Has a lot of statistical data

data <- read.csv("mtcars.csv")		#Helps in reading csv files

#dim(data)

test <- c(4,65,213,5432)
mean(test)		#Gives array mean



skewness(test)		#Gives skewness of matrix

kurtosis(test)		#Gives kurtosus of matrix
mode(test)		#Gives the data type on not statistical mode

summary(test)
which(test == max(test))	#Gives index of mode value

char = c("r","c",1,2,3)
char[3]

summary(char)

x = c(9,10,11,12)	#Array

l = lm(test~x)		#How to call linear regression model

l

summary(l)



#Rabbit		#Dataset in MASS

head(Rabbit)	#Getting 6 rows in dataset

summary(Rabbit)		#Lot of important info there here

freq(Rabbit$Animal)

sd(Rabbit$BPchange)	#Gives standard dev


var(Rabbit$BPchange)	#Gives variance

str(Rabbit)		#Gives info about the data fields

cor(Rabbit$BPchange, Rabbit$Dose)	#Gives the correlation between BPChange and Dose

#mtcars

head(mtcars)		#Scatter Plots of mtcars. Basically gives correlation graph

cor(mtcars)		#Gives correlation values

sapply(mtcars, median)	#Gives median of all vars. change with mode also!

ds <- mtcars

nrow(ds)		#numbers of rows
ncol(ds)		#number of cols

range(ds$hp)		#gives highest and lowest values

table(ds["hp"])		#frequency of element in row

#barplot(mtcars$cyl)

#barplot(table(mtcars["cyl"]))

#mtcars[order(-mtcars$mpg),]		#arranges in descending order. Remove - for inc.

#by(mtcars, mtcars$cyl, summary)	#Gives summary detail for each var for each different occurance of cyl here, (4,6,8)

pie(table(mtcars$cyl))			#Gives pie chart

by(mtcars, mtcars$cyl, decreasing = TRUE, order)	#Similar to the previous thing about dec order

