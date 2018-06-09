library(neuralnet)

set.seed(1234567890)

cs <- read.csv("creditset.csv")

train <- cs[1:800,]

test <- cs[801:2000,]

net <- neuralnet(default10yr~LTI+age, train, hidden = 1, lifesign="minimal", linear.output = FALSE, threshold = 0.1)

summary(net)
