#Chapter 2 
library(ggplot2)
library(magrittr)

data <- read.csv('heck2011.csv')
summary(data)

#Simple Linear Regression 

model1<- lm(math ~ ses, data = data)
summary(model1)

#Visualise
ggplot(data = data, mapping = aes(x = ses, y = math)) + geom_point()

#Multiple Regression 
model2 <- lm(math ~ ses + female, data = data)
summary(model2)

#Interaction Terms 
model3 <- lm(math ~ ses + female + ses:female, data = data)
#Or lm(math~ses*female, data=data)
summary(model3)

library(sjPlot)
sjPlot::plot_model(model3, type = 'pred', terms = c('ses', 'female'))

#Chapter 3: Approaches to Multilevel Data. 

library(dplyr)
library(lmtest)
library(sandwich)

#Dealing with dependence: our data is clustered! How do we deal with this?

#Cluster-Robust Standard Errors
model <- lm(math ~ ses + female, data = data)
summary(model)
#In checking the coefficients, we can see they are the same between mmodels but significance levels differ - this means the differences are trivial and analysis can be proceeded with. 

#Chapter 4: Our First Multilevel Models 

#load in data and dependencies 
library(dplyr) # for data manipulation
library(ggplot2) # for visualizations
library(lme4) # for multilevel models
library(lmerTest) # for p-values
library(performance) # for intraclass correlation

data <- read.csv('heck2011.csv')

#Subset of 10 schools 
data_sub <- data%>%
  filter(schcode <=10)

#Similar ggplot to last, ignoring clustering 
data_sub %>% 
  ggplot(mapping = aes(x = ses, y = math)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE, fullrange = TRUE)

#Recreate plot, colouring the points by school 
data_sub %>% 
  ggplot(mapping = aes(x = ses, y = math, colour = factor(schcode))) +
  geom_point() +
  geom_smooth(mapping = aes(group = schcode), method = "lm", se = FALSE, fullrange = TRUE) +
  labs(colour = "schcode")

#Fixed vs Random Effects
#Fixed effect = average effect accross all clusters; random effect = how an effect for a given cluster differs from the average 

#The Null Model 
null_model <- lmer(math ~ 1 + (1|schcode), data = data)
summary(null_model)
#lmer(DV ~ 1 + IV1 + IV2 + ... + IVp + (random_effect1 + random_effect2 + ... + random_effect3 | grouping_variable), data = dataset)

#Key output to interpret: 
#No. of parameters
#Estimates of fixed effects
#Estimates of random effects variances 

#Understanding Variance
# Intraclass Correlation Coefficent: the more impactful the clustering, the more variance between clusters, the higher the ICC. 
performance::icc(null_model)

#Plausible Values Range: calculating 95% plausible value range for a given effect 

Tau0 <- VarCorr(null_model)$schcode[1]

lower_bound <- null_model@beta - 1.96*sqrt(Tau0)
upper_bound <- null_model@beta + 1.96*sqrt(Tau0)

lower_bound # =51.28024
upper_bound # =64.06822

#Empirical Bayes Estimates: visualising individual residual variance as a way to understand the variance overall 

#mean
data %>% 
  filter(schcode == 1) %>% # select only school code 1
  summarize(
    mean(math)
  )
#Mean = 58.99677
#Intercept from above: 57.6742 

#Group sample size: 
data %>% 
  filter(schcode == 1) %>% 
  count()
#N = 12 

empirical_bayes_data <- as_tibble(ranef(null_model))
head(empirical_bayes_data, 1)

#plot residuals to visualise distribution 
ggplot(data = empirical_bayes_data, mapping = aes(x = condval)) + # "condval" is the name of the EB estimates returned by the ranef function above 
  geom_histogram() +
  labs(x = "EB estimate of U0j")

#Chapter 5: Adding Fixed Predictors to MLMs 

#MLM with Level-1 Predictor 
ses_l1 <- lmer(math ~ ses + (1|schcode), data = data, REML = TRUE)
summary(ses_l1)
#Per the intercept, the average math achievement across all schools at mean ses is 57.596. A one-standard-deviation increase in ses is associated with a 3.87-point increase in math achievement. The variance term describing how schools vary around the intercept is 3.469, whereas the variance term describing how the students vary within schools, about their schools’ mean, is 62.807. These variance terms are different from our null model that had no predictors; we can quantify that difference in at least two ways.

#One option is to calculate how much level-1 variance was reduced by adding ses as a level-1 predictor

null <- sigma(null_model)^2
l1 <- sigma(ses_l1)^2

(null - l1) / null

#0.05624991 - we reduced about 5.6% of level-1 variance by adding SES as a level1 predictor 

#Another option is to calculate the conditional ICC
performance::icc(ses_l1)

#Adjusted ICC: 0.052; Unadjusted ICC: 0.046: After accounting for the effect of socioeconomic status, 4.6% of the variance in math achievement is accounted for by school membership

#MLM With Level-2 Predictor 
ses_l1_public_l2 <- lmer(math ~ 1 + ses + public + (1|schcode), data = data, REML = TRUE)
summary(ses_l1_public_l2)

#Per the intercept, the average math achievement across all private schools (public = 0) at mean SES (ses = 0) is 57.70. A one-standard-deviation increase in ses across all private schools is associated with a 3.87-point increase in math achievement. Public schools at mean ses have a -0.14-point decrease on average in math achievement relative to private schools.From our random effect variances, the variance term describing how schools vary around the intercept (at mean SES at private schools) is 3.48, and the variance term describing how students vary around their school means is 62.81.

#Let’s calculate variance reduced at level 1 and level 2 by adding school type as a predictor.

# level-1 variance reduced
sigma2_null <- sigma(null_model)^2
sigma2_public <- sigma(ses_l1_public_l2)^2
(sigma2_null - sigma2_public) / sigma2_null

##[1] 0.05624525

# level-2 variance reduced
tau2_null <- VarCorr(null_model)$schcode[1]
tau2_public <- VarCorr(ses_l1_public_l2)$schcode[1]
(tau2_null - tau2_public) / tau2_null

##[1] 0.6724414

#We reduced around 5.6% of variance in math achievement at level-1 and 67.2% of variance at level-2 by adding public as a level-2 predictor. It makes sense that the variance at level-2 was reduced by so much more, because we added a level-2 predictor that varies at level-2. 

#We have two sources of information to consider so far: the regression coefficient and the variance reduced. While the regression coefficient is relatively small, the intercept variance reduced at level-2 is quite large (67%!), so it seems like school type is a valuable predictor in our model.

#Random Effects and Cross-Level Interactions 

#MLM with Random Slope Effect 
ses_l1_random <- lmer(math ~ ses + (1 + ses|schcode), data = data, REML = TRUE)
summary(ses_l1_random)

#Let’s look at our fixed effects. Per the intercept, the average math achievement at mean SES (ses = 0) is 57.70. A one-standard-deviation increase in ses across all private schools is associated with a 3.96-point increase in math achievement. From our random effect variances, the variance term describing how schools vary around the intercept (at mean ses) is 3.20 (τ20), the variance term describing how school SES slopes vary around the grand mean slope is 0.78 (τ21), and the variance term describing how students vary around their school’s mean math achievement is 62.59 (σ2). We can find our random effect covariance by examining our Tau matrix. The Tau matrix is called a Tau matrix because it contains the estimates for our random effect variances, or Taus:  τ20,  τ21, etc. We have always been estimating a Tau matrix, but when we only had a random intercept it was just a 1-by-1 matrix of the random intercept term  τ20.

Matrix::bdiag(VarCorr(ses_l1_random))
#The code looks a little busy, but there are two steps. First, we extract our random effects variance-covariance matrix (Tau matrix) with VarCorr(ses_l1_random). Then, we use the bdiag() function from the Matrix package to construct a matrix that’s easy for us to read at a glance.

#We have our covariance from our Tau matrix: -1.58. We can see the standard deviations in the lme4 output: the standard deviation of the intercept variance is 1.79, the standard deviation of the slope variance 0.88. We can then compute the correlation:
-1.58/(1.79*0.88)
##[1] -1.003047

#Let’s visualize the relationship using Empirical Bayes estimates (see Chapter 4 for more on EB estimates) of the intercepts and slopes for each school; we expect to see a negative relationship between them.#
empirical_bayes_data <- ranef(ses_l1_random) # extract random effects for each school

empirical_bayes_intercepts <- empirical_bayes_data$schcode["(Intercept)"]

empirical_bayes_slopes <- empirical_bayes_data$schcode["ses"] # extracts the SES/slope EB estimates from the list

bind_cols(empirical_bayes_intercepts, empirical_bayes_slopes) %>%  # combine EB slopes and intercepts into a useable dataframe for graphing
  ggplot(mapping = aes(x = ses, y = `(Intercept)`)) +
  geom_point()

#MLM with Crosslevel Effect 
# is there a difference in the effect of SES on math achievement in a private or public school? We can answer this question by adding school type as a predictor of SES slopes and create a cross-level interaction. 

crosslevel_model <- lmer(math ~ 1 + ses + public + ses:public + (1 + ses|schcode), data = data, REML = TRUE)

summary(crosslevel_model)

#Let’s look at our fixed effects. Per the intercept, the average math achievement across all private schools (public = 0) at mean SES (ses = 0) is 57.72. A one-standard-deviation increase in ses across all private schools is associated with a 4.42-point increase in math achievement. Public schools (public = 1) at mean ses have a -0.02-point decrease on average in math achievement relative to private schools. The effect of ses on math achievement is lower in public schools by -0.63 points on average, which quantifies the interaction. We can calculate the expected slope for SES in public schools by using these coefficients: 4.42 - 0.63 = 3.79, so a one-unit increase in SES in public schools is associated with a 3.79-unit increase in math achievement, less of an affect than at private schools.
#From our random effect variances, the variance term describing how schools vary around the intercept (at mean SES at public schools) is 3.21, the variance of school slopes around the grand mean is 0.80, and the variance term describing how students vary around their school means is 62.56. We can see our random effect covariance of -1.6 with our Tau matrix, indicating that schools with higher values of mean math achievement at the intercept of ses = 0 have lower slopes of ses.

Matrix::bdiag(VarCorr(crosslevel_model))
