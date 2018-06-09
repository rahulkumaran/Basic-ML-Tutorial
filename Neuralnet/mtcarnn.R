library(MASS)
library(neuralnet)

train <- mtcars[1:20,]

test <- mtcars[21:32,]

net <- neuralnet(mpg~hp+wt+disp, train, hidden = 10, lifesign="minimal", linear.output = FALSE, threshold = 0.1)
net <- neuralnet(mpg~hp+wt+disp, train, hidden = 10, lifesign="minimal", linear.output = FALSE, threshold = 0.1)
net <- neuralnet(mpg~hp+wt+disp, train, hidden = 10, lifesign="minimal", linear.output = FALSE, threshold = 0.1)
net <- neuralnet(mpg~hp+wt+disp, train, hidden = 10, lifesign="minimal", linear.output = FALSE, threshold = 0.1)
net <- neuralnet(mpg~hp+wt+disp, train, hidden = 10, lifesign="minimal", linear.output = FALSE, threshold = 0.1)
temp_test <- subset(test, select=c("hp","wt","disp"))

net.results <- compute(net, temp_test)

results <- data.frame(actual = test$mpg, prediction = net.results$net.result)

results[1:10,]


