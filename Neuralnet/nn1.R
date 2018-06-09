library(neuralnet)

set.seed(1234567890)

cs <- read.csv("click_type_what.csv")

train <- cs[1:800,]

test <- cs[801:2000,]

net <- neuralnet(default10yr~LTI+age, train, hidden = 10, lifesign="minimal", linear.output = FALSE, threshold = 0.1)
net <- neuralnet(default10yr~LTI+age, train, hidden = 10, lifesign="minimal", linear.output = FALSE, threshold = 0.1)
net <- neuralnet(default10yr~LTI+age, train, hidden = 10, lifesign="minimal", linear.output = FALSE, threshold = 0.1)

net

#summary(net)

#plot(net, rep = "best")

temp_test <- subset(test, select = c("LTI","age"))

net.results <- compute(net, temp_test)

results <- data.frame(actual = test$default10yr, prediction = net.results$net.result)

results[100:115,]

results$prediction <- round(results$prediction)

results[100:115,]

table(results$actual, results$prediction)
