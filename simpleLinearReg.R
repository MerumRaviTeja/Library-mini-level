
# library(alr3)
data(faithful)
head(faithful)
### Understand the data 
help("faithful")
str(faithful)
summary(faithful)

###Find the correlation between dependent and independent variables
cor(faithful$eruptions,faithful$waiting)
plot(faithful$eruptions,faithful$waiting)
abline(lm(eruptions ~ waiting, data=faithful))

###Build a linear regression model
eruption.lm = lm(eruptions ~ waiting, data=faithful)

###Find the summary of the model
summary(eruption.lm)

####Regression Coefficients
coeffs = coefficients(eruption.lm); coeffs

####Predictions
waiting = 80           # the waiting time
eruption = coeffs[1] + coeffs[2]*waiting
eruption

pred <- predict(eruption.lm,data=faithful)

head(faithful)

nrow(faithful)

###Residuals

res <- faithful$eruptions - predict(eruption.lm,data=faithful)
head(res)

pred1 <- cbind(faithful,pred)
pred2 <- cbind(pred1,res)
head(pred2)

#############R-Squared###################

eruption.lm = lm(eruptions ~ waiting, data=faithful)
summary(eruption.lm)$r.squared


################Model Summary################
eruption.lm = lm(eruptions ~ waiting, data=faithful)
summary(eruption.lm)


################Predictions################
attach(faithful)     # attach the data frame
eruption.lm = lm(eruptions ~ waiting)
newdata = data.frame(waiting=c(80,95,85))

###Prediction intervals
predict(eruption.lm, newdata, interval="confidence")
detach(faithful)     # clean up


################################
# Check Residuals Distribution

eruption.lm = lm(eruptions ~ waiting, data=faithful)
eruption.res = resid(eruption.lm)
sum(eruption.res)
plot(faithful$waiting, eruption.res,
     ylab="Residuals", xlab="Waiting Time", 
     main="Old Faithful Eruptions")
abline(0, 0)                  # the horizon


# Check Standard Residuals Distribution
eruption.lm = lm(eruptions ~ waiting, data=faithful)
eruption.stdres = rstandard(eruption.lm)
plot(faithful$waiting, eruption.stdres,
     ylab="Standardized Residuals", 
     xlab="Waiting Time", 
     main="Old Faithful Eruptions")
abline(0, 0)                  # the horizon

# Check Standard Residuals is nOrmally Distribution using QQ Plot
eruption.lm = lm(eruptions ~ waiting, data=faithful)
eruption.stdres = rstandard(eruption.lm)
qqnorm(eruption.stdres, 
       ylab="Standardized Residuals", 
       xlab="Normal Scores", 
       main="Old Faithful Eruptions")
qqline(eruption.stdres)

# Check Diagnostic Plots
par(mfrow = c(2,2))
plot(eruption.lm)
dev.off()



###Breusch-Pagan (BP) test for Homoscedasticity
library(lmtest)
bptest(eruption.lm)

data(cars)
head(cars)
