library(survival)
library(randomForest)
library(neuralnet)

train <- lung[1:150,]
test <- lung[151:228,]

rf <- randomForest(status~age+ph.karno, train)

pred <- predict(rf, test)

table(test$status,round(pred))


net <- neuralnet(status~age+time+ph.karno+sex, train, hidden = 10, lifesign="minimal", threshold = 0.1)
net <- neuralnet(status~age+time+ph.karno+sex, train, hidden = 10, lifesign="minimal", threshold = 0.1)
net <- neuralnet(status~age+time+ph.karno+sex, train, hidden = 10, lifesign="minimal", threshold = 0.1)
net <- neuralnet(status~age+time+ph.karno+sex, train, hidden = 10, lifesign="minimal", threshold = 0.1)

temp_test <- subset(test, select = c("age","time","ph.karno","sex"))

net.results <- compute(net, temp_test)

net.results

result <- data.frame(actual = test$status, prediction = net.results$net.result)

result[20:80,]


