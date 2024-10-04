library(readr)
# Load necessary libraries
library(dplyr)

Pizza <- read_csv("D:/x_MSC/3rd_sem/class/Multivariate_stat/MVS_2024/code/Pizza.csv")
View(Pizza)

pizza<-as.matrix(Pizza)
print(pizza)

pizza <- pizza[, -c(1,2)]
pizza<-as.matrix(pizza)
View(pizza)

n<-nrow(pizza)
p<-ncol(pizza)
#-------------------------------------------------
pizza_df <- as.data.frame(pizza)

pizza <- pizza_df %>% mutate_all(~ as.numeric(.))

head(pizza)
str(pizza)

#--------------------------------------------------


StdpizzaPCA <- prcomp(pizza, center = TRUE, scale. = TRUE)
print(StdpizzaPCA)

print(StdpizzaPCA)

StdpizzaFactLoad <- StdpizzaPCA$rotation * t(matrix(rep(StdpizzaPCA$sdev, p), p, p))
print(StdpizzaFactLoad)  # Loadings on factors

StdpizzaSpecVar <- 1 - t(apply(StdpizzaFactLoad^2, 1, cumsum))
print(StdpizzaSpecVar)

CumProp <- (cumsum(StdpizzaPCA$sdev^2)) / sum(StdpizzaPCA$sdev^2)
print(CumProp)

Res1 <- cor(pizza) - StdpizzaFactLoad[,1] %*% t(StdpizzaFactLoad[,1]) - diag(StdpizzaSpecVar[,1])
print(Res1)
sum(Res1 * Res1)

Res2 <- cor(pizza) - StdpizzaFactLoad[,c(1,2)] %*% t(StdpizzaFactLoad[,c(1,2)]) - diag(StdpizzaSpecVar[,2])
print(Res2)
sum(Res2 * Res2)

Res3 <- cor(pizza) - StdpizzaFactLoad[,c(1,2,3)] %*% t(StdpizzaFactLoad[,c(1,2,3)]) - diag(StdpizzaSpecVar[,3])
print(Res3)
sum(Res3 * Res3)



##------------------------------------------------

pizzaFA <- factanal(pizza, factors = 2, scores = "none")

print(pizzaFA)

pizzaSpVr <- 1 - t(apply(pizzaFA$loadings^2, 1, cumsum))
print(diag(pizzaSpVr[,2]))


Res2M <- cor(pizza) - pizzaFA$loadings[,c(1,2)] %*% t(pizzaFA$loadings[,c(1,2)]) - diag(pizzaSpVr[,2])
print(Res2M)
sum(Res2M * Res2M)

m = 2
den = det(cor(pizza))
num = det(pizzaFA$loadings[,c(1,2)] %*% t(pizzaFA$loadings[,c(1,2)]) + diag(pizzaSpVr[,2]))
ts = (n - 1 - (2 * p + 4 * m + 5) / 6) * log(num / den)
df = ((p - m)^2 - p - m) / 2
pval = pchisq(ts, df, lower.tail = FALSE)
print(pval)


#------------------------------------------------------------
PizzaMean <- colMeans(pizza)
DHalf <- sqrt(diag(var(pizza)))
Stdpizza <- (pizza - t(matrix(rep(PizzaMean, n), p, n))) / t(matrix(rep(DHalf, n), p, n))

pizzaFA_Score_Reg <- t(pizzaFA$loadings) %*% solve(cor(pizza)) %*% t(Stdpizza)
plot(pizzaFA_Score_Reg[1,], pizzaFA_Score_Reg[2,])
pizzaFA_reg <- factanal(pizza, factors = 2, scores = "regression")

pizzaFA_Score_WLS <- solve(t(pizzaFA$loadings) %*% solve(cor(pizza)) %*% pizzaFA$loadings) %*% t(pizzaFA$loadings) %*% solve(cor(pizza)) %*% t(Stdpizza)
plot(pizzaFA_Score_WLS[1,], pizzaFA_Score_WLS[2,])
pizzaFA_wls <- factanal(pizza, factors = 2, scores = "Bartlett")

