Airbnb_data <- read.csv(file="C:/Users/simon/Documents/3142_ass1/airbnb-Sydney.csv", header=TRUE, na.strings = "N/A")

library(dplyr)
library(ggplot2)
library(MASS)
library(moments)
library(car)
library(stringr)


#Q1
#numeric predictors
numeric_data <- Airbnb_data[, sapply(Airbnb_data, is.numeric)]
numeric_data <- na.omit(numeric_data)

#calculate price correlations
price_cor <- cor(numeric_data$price_per_room, numeric_data)

#remove cor with x (counting column) and price_per_room
price_cor <- price_cor[, -c(1,33)]

#order correlations in ascending order
sorted_price_cor <- sort(price_cor, decreasing = F)

#increase plot limits
par(mfrow = c(1,1))
par(mar = c(13, 4, 4, 2))

#plot Correlations with price per room
barplot(sorted_price_cor,
        las = 2,
        col = "tomato",
        main = "Correlations with price_per_room",
        ylab = "Correlation")


#categorical predictors
#room_type

#create box plots of price per room based on room type
#Calculate medians of price per room based on room type
tapply(Airbnb_data$price_per_room,
       Airbnb_data$room_type,
       median)
#plot
ggplot(Airbnb_data, aes(x = reorder(room_type, price_per_room, median),
                          y = price_per_room, fill = room_type)) +
  #remove outliers from graph (still used in calculations)
  geom_boxplot(outlier.shape = NA) +
  coord_flip() + 
  labs(title = "Price per room by Room Type", 
       x = "Room Type", 
       y = "Price per Room") +
  theme_classic()


#Create lollipop chart of median price per rooms by neighbourhood
#Group data by neighbourhoods and arrange in order by median price

Airbnb_data %>%
  group_by(neighbourhood_cleansed) %>%
  summarise(median_price = median(price_per_room)) %>%
  arrange(median_price) %>%
  mutate(neighbourhood_cleansed = factor(
    neighbourhood_cleansed,
    levels = neighbourhood_cleansed
  )) %>%
  
  #Create lollipop chart
  ggplot(aes(x = neighbourhood_cleansed, y = median_price)) + 
  geom_segment(aes(xend = neighbourhood_cleansed, y = 0, yend = median_price)) +
  geom_point() +
  coord_flip() + 
  labs(title = "Median Price per Room by Neighbourhood", 
       x = "Neighbourhood", 
       y = "Median Price per Room") + 
  theme_classic()

#Calculate Correlations between Review Score ratings
cor(Airbnb_data[, c("review_scores_rating", "review_scores_location",
                    "review_scores_cleanliness", "review_scores_accuracy",
                    "review_scores_checkin", "review_scores_communication",
                    "review_scores_value")], use = "complete.obs")

#Additional Data manipulation for models (cleaning up data)

#Host is superhost was not a binary variable
table(Airbnb_data$host_is_superhost, useNA = "always")

#Blank returns are not NAs but indicate hosts who have not been evaluated to superhosts
#Thus T --> High quality host, F --> Lower quality hosts, NA --> new/inactive host
#Reclassify data to represent this fact 

Airbnb_data$host_is_superhost <- as.character(Airbnb_data$host_is_superhost)
Airbnb_data$host_is_superhost[Airbnb_data$host_is_superhost == ""] <- "not_evaluated"
Airbnb_data$host_is_superhost <- as.factor(Airbnb_data$host_is_superhost)

#plot host_is_superhost vs price_per_room

ggplot(Airbnb_data, aes(x = reorder(host_is_superhost, price_per_room, median),
                        y = price_per_room, fill = host_is_superhost)) +
  geom_boxplot(outlier.shape = NA) +
  coord_flip(ylim = c(0,400)) + 
  labs(title = "Price per room by Host Status", 
       x = "Host Status", 
       y = "Price per Room") +
  theme_classic()

#host_response_time
table(Airbnb_data$host_response_time, useNA = "always")
unique(Airbnb_data$host_response_time)

#Na responses in host_response_time likely represents when listings don't respond
#rather than lack of data

#Make NA responses into "never_responds"
Airbnb_data$host_response_time <- as.character(Airbnb_data$host_response_time)
Airbnb_data$host_response_time[is.na(Airbnb_data$host_response_time)] <- "never_responds"
Airbnb_data$host_response_time <- as.factor(Airbnb_data$host_response_time)

table(Airbnb_data$host_response_time, useNA = "always")


#Q2
#initial model using highest correlation predictors

#starting with all 4 
#drop longitude -- likely has overlap with neighrbouhood cleansed
#drop longitude bc R^2 is smaller when longitude is dropped compared to when neighbourhood is dropped

lrv1 <- lm(formula = price_per_room ~ longitude + reviews_per_month + 
             room_type + neighbourhood_cleansed, data = Airbnb_data)
summary(lrv1)
AIC(lrv1)

#Model without longitude
lrv2 <- lm(formula = price_per_room ~ reviews_per_month + 
             room_type + neighbourhood_cleansed, data = Airbnb_data)
summary(lrv2)
AIC(lrv2)

#Longitude and neighbourhood had overlapping explanatory power:
#Longitude removed due to lower R^2 

#Currated list of predictors
scope_formula <- list(
  lower = ~ 1,
  upper = ~ reviews_per_month + 
    room_type + 
    neighbourhood_cleansed + 
    property_type + 
    host_is_superhost + 
    review_scores_rating + 
    accommodates + bathrooms + 
    longitude + 
    latitude
)

#Produce Step model based on predictors
step.model <- stepAIC(lrv1, direction = "both", scope = scope_formula, trace = F)
summary(step.model)

step.model <- lm(formula = price_per_room ~ longitude + reviews_per_month + 
     room_type + neighbourhood_cleansed + property_type + review_scores_rating + 
     accommodates + bathrooms + latitude + host_is_superhost, 
   data = Airbnb_data)
plot(step.model)

#residual vs fitted -- spread of residuals fan out from 0-300 then narrow again -- models errors are not constant
#evidence of heteroscedasticity (violates assumption) -- should take the log and replot

#Change 1: Removed property type -- Introduces many coefficients for limited improvement
#Adj R^2: 0.2665
modelv1 <- lm(price_per_room ~ longitude + reviews_per_month + 
                room_type + neighbourhood_cleansed + review_scores_rating + 
                accommodates + bathrooms + latitude + host_is_superhost, 
              data = Airbnb_data)
summary(modelv1)

#Change 2: Removed longitude and latitude - Neighbourhood_cleansed better represents this information
#Added bedrooms
#Adj R^2: 0.2815
modelv2 <- lm(price_per_room ~  reviews_per_month + 
                room_type + neighbourhood_cleansed + review_scores_rating + 
                accommodates + bathrooms + bedrooms + host_is_superhost, 
              data = Airbnb_data)
summary(modelv2)

#Change 3: Large Skew (2.31) and Kurtosis (13.92)
skewness(Airbnb_data$price_per_room)
kurtosis(Airbnb_data$price_per_room)

#residual vs fitted -- spread of residuals fan out from 0-300 then narrow again -- models errors are not constant
#evidence of heteroscedasticity (violates assumption)
#Model Log price per room rather than price per room

#Create log price per room variable
Airbnb_data$log_price <- log(Airbnb_data$price_per_room)

#Initial log model to compare with step model
#Adj R^2: 0.4363
log_step_model <- lm(log_price ~ longitude + reviews_per_month + 
                       room_type + neighbourhood_cleansed + property_type + review_scores_rating + 
                       accommodates + bathrooms + latitude + host_is_superhost, 
                     data = Airbnb_data)
summary(log_step_model)
plot(log_step_model)

#Change 4: Apply changes made to step model to the initial log model
#Change 5: Remove property type
#Adj R^2: 0.4225
logv2 <- lm(log_price ~ reviews_per_month + 
              room_type + neighbourhood_cleansed + review_scores_rating + 
              accommodates + bathrooms + host_is_superhost + bedrooms, 
            data = Airbnb_data)

summary(logv2)

#Property Type analysis

#only a few were significant 
#property_typeHouseboat 0.00699 ** 
#property_typePrivate room in chalet 0.01033 * 
#property_typePrivate room in villa 0.03816 * 

#Majority insignificant -- 1.7% increase in Rsquared -- 55 Extra coefficients

#Create a binary predictor for houseboat, the most significant property type
Airbnb_data$is_houseboat <- ifelse(Airbnb_data$property_type == "Houseboat", 1, 0)

#Change 6: is_houseboat used as replacement for property type
#logv3 and logv4 used to compare the influence of host_is_superhost
#Without host_is_superhost

#Adj R^2: 0.423
logv3 <- lm(log_price ~ reviews_per_month + room_type + neighbourhood_cleansed +
              review_scores_rating + accommodates + bathrooms +
              bedrooms + is_houseboat,
            data = Airbnb_data)
summary(logv3)

#With host_is_superhost
#Adj R^2: 0.4233
logv4 <- lm(log_price ~ reviews_per_month + room_type + neighbourhood_cleansed +
              review_scores_rating + accommodates + bathrooms +
              bedrooms + is_houseboat + host_is_superhost,
            data = Airbnb_data)
summary(logv4)

#comparison of models
AIC(logv3)  # without host_is_superhost
AIC(logv4)  # with host_is_superhost
BIC(logv3)
BIC(logv4)

#Change 7: host_is_superhost removed -- Found to be insignificant
#Change 8: add host_response_time
#Adj R^2: 0.4258
logv5 <- lm(log_price ~ reviews_per_month + room_type + neighbourhood_cleansed +
              review_scores_rating + accommodates + bathrooms +
              bedrooms + is_houseboat + host_response_time,
            data = Airbnb_data)
summary(logv5)

cor(Airbnb_data$accommodates, Airbnb_data$bedrooms)
#accommodates and bedrooms have high correlation -- 0.87
#create guests per room -- accomodates/ bedrooms to replace

Airbnb_data$guests_per_bedroom <- ifelse(
  Airbnb_data$bedrooms > 0, Airbnb_data$accommodates/Airbnb_data$bedrooms,
  Airbnb_data$accommodates / 1
)

#Change 9: guests_per_bedroom introduced to replace accommodates and bedrooms

logv6 <- lm(log_price ~ reviews_per_month + room_type + neighbourhood_cleansed +
            review_scores_rating + bathrooms +is_houseboat + host_response_time
            + guests_per_bedroom,
            data = Airbnb_data)
summary(logv6)

#Problems introduced by model logv6: 
#Listings with 0 bedrooms create NaNs
  #Problem resolved by setting to equal accommodates
#host_response_time includes large amounts of blank responses
  #Problems resolved in additional data manipulations section

#Model with problems resolved

#Adj R^2: 0.4136
logv6fixed <- lm(log_price ~ reviews_per_month + room_type + neighbourhood_cleansed +
              review_scores_rating + bathrooms +is_houseboat + host_response_time
            + guests_per_bedroom,
            data = Airbnb_data)
summary(logv6fixed)

#Change 10: Reintroduced bedrooms, accommodates
#Adj R^2: 0.4496
logv7 <- lm(log_price ~ reviews_per_month + room_type + neighbourhood_cleansed +
              review_scores_rating + accommodates + bathrooms + bedrooms +
              is_houseboat + host_response_time + guests_per_bedroom,
            data = Airbnb_data)

summary(logv7)
vif(logv7)
#Both accommodates and bedrooms contribute to highly variability 
#Still remain significant to model -- retain all 3 predictors


#Create amenity count predictor
Airbnb_data$amenity_count <- str_count(Airbnb_data$amenities, ",") + 1

#Change 11: Add amenity Count
#Adj R^2: 0.4513
logv8 <- lm(log_price ~ reviews_per_month + room_type + neighbourhood_cleansed +
              review_scores_rating + accommodates + bathrooms + bedrooms +
              is_houseboat + host_response_time + guests_per_bedroom + amenity_count,
            data = Airbnb_data)
summary(logv8)

#Final model Comparisons
AIC(logv3); AIC(logv5); AIC(logv6fixed); AIC(logv7); AIC(logv8)
BIC(logv3); BIC(logv5); BIC(logv6fixed); BIC(logv7); BIC(logv8)

#Final model
lm_final <- logv8

summary(lm_final)
AIC(lm_final)
BIC(lm_final)
vif(lm_final)

#Q3
#take final formula from linear regression model for initial analysis
glm_formula <- price_per_room ~ reviews_per_month + room_type + neighbourhood_cleansed +
  review_scores_rating + accommodates + bathrooms + bedrooms +
  is_houseboat + host_response_time + guests_per_bedroom + amenity_count

glmv1 <- glm(glm_formula, family = Gamma(link = "log"), data = Airbnb_data)
summary(glmv1)

#Choosing Stochastic function: Gaussian vs Gamma

#Create Gaussian model
glm_gaussian <- glm(glm_formula, family = gaussian(link = "log"), data = Airbnb_data)
glm_gamma <- glm(glm_formula, family = Gamma(link = "log"), data = Airbnb_data)

AIC(glm_gaussian) - AIC(glm_gamma)
#4584.529 > Gamma structure fits better

plot(glm_gaussian)
plot(glm_gamma) 

#Choosing Link Function for Gamma: Log vs Inverse vs Identity

glmv1_log <- glm(glm_formula, family = Gamma(link = "log"), data = Airbnb_data)
glmv1_inverse <- glm(glm_formula, family = Gamma(link = "inverse"), data = Airbnb_data)

#Identity link fails to run - No valid set of coefficients
#Due to negatives being passed 
#Supply starting values using linear model
lm_start <- lm(glm_formula, data = Airbnb_data)

glmv1_identity <- glm(glm_formula, family = Gamma(link = "identity"),
                      start = coef(lm_start), data = Airbnb_data)
#still fails to run -- Evidence that the identity link is insufficient

#comparison will continue between log and inverse
AIC(glmv1_log); AIC(glmv1_inverse)

glm_log_family_links <- list(
  "log" = glmv1_log,
  "inverse" = glmv1_inverse
)

data.frame(
  #AIC
  AIC = sapply(glm_log_family_links, AIC),
  
  Pseudo_R_squared = sapply(glm_log_family_links, function(m) 
    round(1 - m$deviance / m$null.deviance, 4)),
  
  Residual_deviance = sapply(glm_log_family_links, function(m) 
    round(m$deviance / m$df.residual, 4))
)

#Procede with Log link function

#Perform stepwise selection to ensure predictors are significant

#Null and full scope
glm_null <- glm(price_per_room ~ 1,
                family = Gamma(link = "log"),
                data = Airbnb_data)

glm_full <- glm(glm_formula,
                family = Gamma(link = "log"),
                data = Airbnb_data)

#Stepwise on GLM
glm_step <- stepAIC(glm_full,
                    scope = list(lower = glm_null, upper = glm_full),
                    direction = "both",
                    trace = FALSE)

summary(glm_step)
#retained all variables 

#FINAL GLM model

glm_final <- glmv1_log
summary(glm_final)

#Q4
#display summary datas together
par(mfrow = c(2,2))
plot(logv8,     main = "Linear Model (logv8)")

par(mfrow = c(2,2))
plot(glm_final, main = "GLM (Gamma log)")


#Compare overall fit by generating predictions using original data

#linear model requires transformation back from log
pred_lm  <- exp(fitted(lm_final))

pred_glm <- fitted(glm_final)     

#Compute RMSE
actual <- Airbnb_data$price_per_room

rmse_lm  <- sqrt(mean((actual - pred_lm)^2,  na.rm = TRUE))
rmse_glm <- sqrt(mean((actual - pred_glm)^2, na.rm = TRUE))

data.frame(
  RMSE = c(rmse_lm, rmse_glm),
  row.names = c("logv8 (LM)", "glm_final (GLM)")
)
