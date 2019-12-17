# Load required packages
library(ggplot2)
library(car)
library(caret)
library(corrplot)

#Loading data
data(mtcars)  

# Looking at variables
str(mtcars)
head(mtcars)

# Summarize Data
summary(mtcars)

# Data Preparation

mtcars$am   = as.factor(mtcars$am)
mtcars$cyl  = as.factor(mtcars$cyl)
mtcars$vs   = as.factor(mtcars$vs)
mtcars$gear = as.factor(mtcars$gear)

# Identifying and Correcting Collinearity

#Dropping dependent variable for calculating Multicollinearity
mtcars_a = subset(mtcars, select = -c(am, cyl, vs, gear))
mtcars_b = subset(mtcars, select = -c(mpg, cyl, vs, gear))

#Identifying numeric variables
numericData <- mtcars_a[sapply(mtcars_a, is.numeric)]
numericData1 <- mtcars_b[sapply(mtcars_b, is.numeric)]

#Calculating Correlation
descrCor <- cor(numericData1)

# Print correlation matrix and look at max correlation
print(descrCor)
# Visualize Correlation Matrix
corrplot(descrCor, order = "FPC", method = "color", type = "lower", tl.cex = 0.7, tl.col = rgb(0, 0, 0))
# Visualize Correlation Matrix
corrplot(descrCor)

# Checking Variables that are highly correlated
highlyCorrelated = findCorrelation(descrCor, cutoff=0.7)

#Identifying Variable Names of Highly Correlated Variables
highlyCorCol = colnames(numericData)[highlyCorrelated]

#Print highly correlated attributes
highlyCorCol

#Remove highly correlated variables and create a new dataset
dat3 = mtcars[, -which(colnames(mtcars) %in% highlyCorCol)]
dim(dat3)
str(dat3)
# Developing Regression Model

#Build Linear Regression Model
fit = lm(mpg ~ ., data=dat3)

#Check Model Performance
summary(fit)

#Extracting Coefficients
summary(fit)$coeff

#Diagnostic Plot
par(mfrow=c(2,2))
plot(fit)

# See the coefficients of Linear Regression Model and ANOVA table

summary(fit)

# Calculating Model Performance Metrics
#Extracting R-squared value
summary(fit)$r.squared
#Extracting Adjusted R-squared value
summary(fit)$adj.r.squared
AIC(fit)
BIC(fit)


# Variable Selection Methods

#Stepwise Selection based on AIC
library(MASS)
step <- stepAIC(fit, direction="both")
summary(step)

#Backward Selection based on AIC
step1 <- stepAIC(fit, direction="backward")
summary(step1)


#Forward Selection based on AIC
step2 <- stepAIC(fit, direction="forward")
summary(step2)


#Stepwise Selection with BIC
n = dim(dat3)[1]
stepBIC = stepAIC(fit,k=log(n))
summary(stepBIC)

AIC(stepBIC)
BIC(stepBIC)

# Calculating Variance Inflation Factor (VIF)

vif(step)

#Normality Of Residuals (Should be > 0.05)
res=residuals(step,type="pearson")
shapiro.test(res)

#Testing for homoscedasticity (Should be > 0.05)
bptest(step)

#See Residuals
resid = residuals(step)

# See Actual vs. Prediction

#See Predicted Value
pred = predict(step,dat3)
#See Actual vs. Predicted Value
finaldata = cbind(mtcars,pred)
print(head(subset(finaldata, select = c(mpg,pred))))

#Calculating RMSE
rmse = sqrt(mean((dat3$mpg - pred)^2))
print(rmse)