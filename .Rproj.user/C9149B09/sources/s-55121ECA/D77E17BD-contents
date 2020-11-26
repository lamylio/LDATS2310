rm(list=ls())

library(tidyr)
library(ggplot2)

# ---
# Chargement des données

base_test <- read.table("./data/DBtest.csv", sep=",", header=TRUE)
base_train = read.table("./data/DBtrain.csv", sep=",", header=TRUE)

# ---
# 
base_train$Gender = as.factor(base_train$Gender)
base_train$Area = as.factor(base_train$Area)
base_train$Leasing = as.factor(base_train$Leasing)
base_train$Power = as.factor(base_train$Power)
base_train$Contract = as.factor(base_train$Contract)
base_train$Fract = as.factor(base_train$Fract)

seuilAGE     = c(20,25,30,35,40,45,50,80)
base_train$DriverAge <- cut(base_train$DriverAge, breaks = seuilAGE )

# Area = ["Suburban", "Urban", "Countryside low altitude", "Coutryside high altitude"]
# Power = ["Low", "Normal", "Intermediate", "High"]
# Frac = Split = ["Monthly", "Quarterly", "Yearly"]
# Contract = ["Basic", "Intermediate", "Full"]
# Exposure = Duration of contract in year

# ---
# Claim frequency is Duration/Nbclaimsnumber : nb of claims divided by the duration for some group of policies in force during a specic period of time.
# Claim severity is total claim amount divided by the number of claims i.e. the average cost per claim.

base_train$ClaimFrequency = base_train$Exposure / base_train$Nbclaims

# ---
# GLM
    
# if w > 0 and Y = X/w : E(x) = wµ, E(y) = µ, var(x) = wsigma², var(y) = sigma²/w where µ and sigma are when w = 1

age_table = table(base_train$DriverAge)
barplot(age_table) # Enormément ont 18 ans, à part ça c'est ~ N(43.2642, 14.3147²)

# Assume nbClaim is a poisson distribution

summary(base_train)
age_table = table(base_train$DriverAge)
claim_table = table(base_train$Nbclaims)

# Faire une analyse de chacunes des variables
barplot(table(base_train$DriverAge))
# X% of them are in the xth group.. etc
# round(prop.table(table(base_train$DriverAge, base_train$Nbclaims)),4)*100
# 
# AtLeastOneClaim = base_train %>% group_by(DriverAge) %>% mutate(AtleastOne = Nbclaims > 0, Noclaims = Nbclaims < 1) %>% summarise(AtLeastOne = sum(AtleastOne), NoClaims=length(Noclaims)-sum(AtleastOne), PercentOfClaim=AtLeastOne/(AtLeastOne+NoClaims)*100, PercentOfNoClaim=100-PercentOfClaim)
# 
# PercentOfClaimPerDriverAgeComparedToOthers = AtLeastOneClaim %>% ungroup() %>% summarize(PercentOfClaim=AtLeastOne/sum(AtLeastOne)*100)
# barplot(PercentOfClaimPerYear$PercentOfClaim)
# 
# # Do it per Gender per Age
# ftable(base_train$Gender, base_train$DriverAge, base_train$Nbclaims)


# Modal classes with GLM breaks
train.modals = glm_train %>% count(Gender, DriverAge, CarAge, Area, Leasing, Power, Fract, Contract, Nbclaims) %>% top_n(1)
train.modals[] = lapply(train.modals, as.character)

# ---
# Regression Trees

# ---
# RF

# ---
# GBM

# ---
# Neural