Airbnb_data <- read.csv(file="C:/Users/simon/Documents/3142_ass2/airbnb-Sydney.csv", header=TRUE, na.strings = "N/A")

#NOTE: apologies in advance for messiness and large amounts of comments

library(dplyr)
library(ggplot2)
library(glmnet)
library(pROC)
library(car)
library(stringr)
library(moments)
library(mgcv)
library(xgboost)

#Classification
#DATACLEANING

Cleansed_data <- Airbnb_data

#Handle NAs in categorial predictors 
#host_is_superhost -- NA - not evaluated
Cleansed_data$host_is_superhost <- as.character(Cleansed_data$host_is_superhost)
Cleansed_data$host_is_superhost[Cleansed_data$host_is_superhost == ""] <- "not_evaluated"
Cleansed_data$host_is_superhost <- as.factor(Cleansed_data$host_is_superhost)

#host_response_time -- NA - never
Cleansed_data$host_response_time <- as.character(Cleansed_data$host_response_time)
Cleansed_data$host_response_time[is.na(Cleansed_data$host_response_time)] <- "never_responds"
Cleansed_data$host_response_time <- as.factor(Cleansed_data$host_response_time)

#host_response_rate and host_acceptance_rate -- remove ~900 nas introduced
table(Airbnb_data$host_acceptance_rate)
table(Airbnb_data$host_response_rate)

table(Airbnb_data$host_neighbourhood)
#DROP host_neighbourhood (72% of data missing)

#host_since --> create host_tenure_days
Cleansed_data$host_since <- as.Date(Cleansed_data$host_since)

#proxy when data was collected using most recent host creation date
snapshot_date <- max(Cleansed_data$host_since) 
Cleansed_data$host_tenure_days <- as.numeric(snapshot_date - Cleansed_data$host_since)

#Host_verification count --> sum number of verification methods rather than string list
Cleansed_data$verification_count <- str_count(Cleansed_data$host_verifications, ",") + 1

#Amenity count --> sum number of unique amenities
Cleansed_data$amenity_count <- str_count(Cleansed_data$amenities, ",") + 1

#Property_type: Combine rare levels (observations < 10) into an "other" category
#may resolve issues with train/test data splits
property_type_counts <- table(Cleansed_data$property_type)
rare_property_types <- names(property_type_counts[property_type_counts < 10])

Cleansed_data$property_type <- as.character(Cleansed_data$property_type)
Cleansed_data$property_type[Cleansed_data$property_type %in% rare_property_types] <- "Other"
Cleansed_data$property_type <- as.factor(Cleansed_data$property_type)
table(Cleansed_data$property_type)


#ASSEMBLING TRAINING/TEST data -- remove certain predictors and separate data
review_score_vars <- c("review_scores_rating", "review_scores_accuracy",
                       "review_scores_cleanliness", "review_scores_checkin",
                       "review_scores_communication", "review_scores_location",
                       "review_scores_value")

drop_vars <- c("X", "id", review_score_vars, "bathrooms_text", "host_neighbourhood",
               "host_since", "host_verifications", "amenities",
               "first_review", "last_review", "host_acceptance_rate", "host_response_rate")


model_data <- Cleansed_data %>%
  select(-any_of(drop_vars)) %>%
  mutate(rating_q1 = as.factor(rating_q1))

model_data <- na.omit(model_data)
nrow(model_data); nrow(Airbnb_data) 
#Preserved all obs

#Build entire model matrix first to avoid issues with test and training data not sharing identical columns

x_all <- model.matrix(rating_q1 ~ . , data = model_data)[, -1]
y_all <- model_data$rating_q1

#Split data into training and test
set.seed(3142)

train_index <- sample(seq_len(nrow(model_data)), size = 0.7 * nrow(model_data))
train_data <- model_data[train_index, ]
test_data <- model_data[-train_index,]

#Matrix Model for glmnet
x_train <- x_all[train_index, ]
y_train <- y_all[train_index]
x_test  <- x_all[-train_index, ]
y_test  <- y_all[-train_index]

#LASSO regression model
set.seed(3142)
cv_lasso <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1, type.measure = "auc")
plot(cv_lasso)
cv_lasso$lambda.min
cv_lasso$lambda.1se

#RIDGE regression model
set.seed(3142)
cv_ridge <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 0, type.measure = "auc")

#LASSO vs RIDGE regression
pred_lasso <- predict(cv_lasso, newx = x_test, s = "lambda.1se", type = "response")
pred_ridge <- predict(cv_ridge, newx = x_test, s = "lambda.min", type = "response")

roc_lasso <- roc(y_test, as.numeric(pred_lasso))
roc_ridge <- roc(y_test, as.numeric(pred_ridge))
table(Airbnb_data$property_type)
auc(roc_lasso)
auc(roc_ridge)

#TAKE THE COEFFS FROM THE LASSO REGRESSION 
lasso_coefs <- coef(cv_lasso, s = "lambda.1se")
selected_vars <- rownames(lasso_coefs)[which(lasso_coefs != 0)]
selected_vars <- setdiff(selected_vars, "(Intercept)")
selected_vars

#Selected variables remove some property_types and neighbourhood_cleansed
#Majority of neighbourhoods are removed (37/38) -- remove neighbourhood_cleansed from model
#Many property_types retained (9/16) -- reintroduce all property_types

glm_formula <- rating_q1 ~ host_response_time +
  host_total_listings_count + 
  property_type +  
  maximum_nights +  
  availability_90 + 
  number_of_reviews_l30d + 
  instant_bookable +
  amenity_count +  
  host_is_superhost +  
  host_identity_verified +  
  longitude + 
  bedrooms +  
  minimum_minimum_nights +  
  number_of_reviews + 
  estimated_occupancy_l365d +  
  price_per_room

glm_with_neighbourhood <- update(glm_formula, .~. + neighbourhood_cleansed)

#glm with neighrbourhood cleansed vs without

glm_v1 <- glm(glm_formula, data = train_data, family = "binomial")
glm_v2 <- glm(glm_with_neighbourhood, data = train_data, family = "binomial")
anova(glm_v1, glm_v2, test = "Chisq")

#Neighbourhood cleansed appears to be somewhat significant p - 0.02274 but increases DF by 37

glm_test_no_neigh <- predict(glm_v1, newdata = test_data, type = "response")
glm_test_with_neigh <- predict(glm_v2, newdata = test_data, type = "response")

auc(roc(test_data$rating_q1, glm_test_no_neigh))
auc(roc(test_data$rating_q1, glm_test_with_neigh))

#AUC decreases -- evidence of overfitting -- preserve glm_v1 without neighbourhood cleansed

summary(glm_v1)
#Take the exponent to convert log-odds into odds ratio
exp(coef(glm_v1))
#Highest OR is host_is_superhostt -- don't know what factors affect becoming a superhost
#Could be data leakage

#Intercept term is ^74 -- this is due to the longitude of sydney being across 150 - 151 degs
#Change: center both training and test data using training longitude mean
longitude_mean <- mean(train_data$longitude)

--------------------------------#COMMENTED TO NOT ACCIDENTALLY RUN AGAIN
#train_data$longitude <- train_data$longitude - longitude_mean
#test_data$longitude  <- test_data$longitude - longitude_mean
--------------------------------#COMMENTED TO NOT ACCIDENTALLY RUN AGAIN
mean(train_data$longitude)
mean(test_data$longitude)

#Some property_types had high standard error and are insignificant
#-- Entire vacation home has only 13 obs -- 
#increase minimum obs for to be classified as "other" to 20
property_type_counts <- table(Cleansed_data$property_type)
rare_property_types <- names(property_type_counts[property_type_counts < 20])

Cleansed_data$property_type <- as.character(Cleansed_data$property_type)
Cleansed_data$property_type[Cleansed_data$property_type %in% rare_property_types] <- "Other"
Cleansed_data$property_type <- as.factor(Cleansed_data$property_type)

#(3/4) host_response_time were found to be insignificant -- like with neighbourhood_cleansed - remove
glm_no_response_time <- update(glm_v1, . ~ . - host_response_time)
anova(glm_no_response_time, glm_v1, test = "Chisq")
#0.2107 likelihood -- likely insignificant

pred_no_response_time <- predict(glm_no_response_time, newdata = test_data, type = "response")
auc(roc(test_data$rating_q1, pred_no_response_time))
auc(roc(test_data$rating_q1, glm_test_no_neigh))
#AUC decreases by 0.0018 -- remove host_response_times

vif(glm_v1)
#All vif numbers below 3.2 and gvif below 1.5 -- below 2 rule of thumb concern threshold 
#multicolinearity is not a major issue in the model

#MODEL V3 - reperform lasso and ridge 
model_data_v3 <- Cleansed_data %>%
  select(-any_of(drop_vars)) %>%
  mutate(rating_q1 = as.factor(rating_q1))

model_data_v3 <- na.omit(model_data_v3)
nrow(model_data_v3); nrow(Airbnb_data) 

x_all_v3 <- model.matrix(rating_q1 ~ . , data = model_data_v3)[, -1]
y_all_v3 <- model_data_v3$rating_q1

#Split data into training and test
set.seed(3142)

train_index_v3 <- sample(seq_len(nrow(model_data_v3)), size = 0.7 * nrow(model_data_v3))
train_data_v3 <- model_data_v3[train_index_v3, ]
test_data_v3 <- model_data_v3[-train_index_v3,]

x_train_v3 <- x_all_v3[train_index_v3, ]
y_train_v3 <- y_all_v3[train_index_v3]
x_test_v3  <- x_all_v3[-train_index_v3, ]
y_test_v3  <- y_all_v3[-train_index_v3]

#train_data_v3$longitude <- train_data_v3$longitude - longitude_mean
#test_data_v3$longitude  <- test_data_v3$longitude - longitude_mean
#commented to avoid rerunning

#LASSO regression model v3
set.seed(3142)
cv_lasso_v3 <- cv.glmnet(x_train_v3, y_train_v3, family = "binomial", alpha = 1, type.measure = "auc")
plot(cv_lasso_v3)
cv_lasso_v3$lambda.min
cv_lasso_v3$lambda.1se

#RIDGE regression model v3
set.seed(3142)
cv_ridge_v3 <- cv.glmnet(x_train_v3, y_train_v3, family = "binomial", alpha = 0, type.measure = "auc")

#LASSO vs RIDGE regression
pred_lasso_v3 <- predict(cv_lasso_v3, newx = x_test_v3, s = "lambda.1se", type = "response")
pred_ridge_v3 <- predict(cv_ridge_v3, newx = x_test_v3, s = "lambda.min", type = "response")

roc_lasso_v3 <- roc(y_test_v3, as.numeric(pred_lasso_v3))
roc_ridge_v3 <- roc(y_test_v3, as.numeric(pred_ridge_v3))

auc(roc_lasso_v3)
auc(roc_ridge_v3)
#results insignificantly different from before - likely from noise fluctuation
#Prior data manip. steps do not affect predictive accuracy and resolve instability  

set.seed(3142)
lasso_coefs_v3 <- coef(cv_lasso_v3, s = "lambda.1se")
selected_vars_v3 <- rownames(lasso_coefs_v3)[which(lasso_coefs_v3 != 0)]
selected_vars_v3 <- setdiff(selected_vars_v3, "(Intercept)")
selected_vars_v3

#differences with prev lasso -- 2 Property types dropped (entireplace and private room in bed and breakfast)
#(small categories folded into other)
#Everything else is identical 

glm_formula_v3 <- rating_q1 ~ host_total_listings_count + 
  property_type +  
  maximum_nights +  
  availability_90 + 
  number_of_reviews_l30d + 
  instant_bookable +
  amenity_count +  
  host_is_superhost +  
  host_identity_verified +  
  longitude + 
  bedrooms +  
  minimum_minimum_nights +  
  number_of_reviews + 
  estimated_occupancy_l365d +  
  price_per_room

glm_v3 <- glm(glm_formula_v3, data = train_data_v3, family = "binomial")
summary(glm_v3)
exp(coef(glm_v3))
vif(glm_v3)

#glm_v3 analysis
pred_glm_v3 <- predict(glm_v3, newdata = test_data_v3, type = "response")

roc_glm_v3 <- roc(test_data_v3$rating_q1, pred_glm_v3)
auc(roc_glm_v3)
plot(roc_glm_v3, main = "ROC curve - logistic regression (v3)")

#convert model to classification outputs using 0.5 threshold
pred_class <- ifelse(pred_glm_v3 > 0.5, 1, 0)
conf_matrix_model <- table(Predicted = pred_class, Actual = test_data_v3$rating_q1)
conf_matrix_model

accuracy_model  <- sum(diag(conf_matrix_model)) / sum(conf_matrix_model)
precision_model <- conf_matrix_model["1","1"] / sum(conf_matrix_model["1",])
recall_model    <- conf_matrix_model["1","1"] / sum(conf_matrix_model[,"1"])
accuracy_model
precision_model
recall_model

#Upon analysis of accuracy and precision data -- try to optimise code by looking at different thresholds
#0.25, 0.5, 0.75

#Function to compute at different thresholds
evaluate_threshold <- function(pred_probs, actual, threshold) {
  pred_class <- ifelse(pred_probs > threshold, 1, 0)
  conf_matrix <- table(Predicted = factor(pred_class, levels = c(0,1)),
                       Actual = actual)
  
  accuracy  <- sum(diag(conf_matrix)) / sum(conf_matrix)
  precision <- conf_matrix["1","1"] / sum(conf_matrix["1",])
  recall    <- conf_matrix["1","1"] / sum(conf_matrix[,"1"])
  
  list(threshold = threshold, confusion_matrix = conf_matrix,
       accuracy = accuracy, precision = precision, recall = recall)
}

#Test thresholds
thresholds_to_test <- c(0.25, 0.5, 0.75)

threshold_results <- lapply(thresholds_to_test, function(t) {
  evaluate_threshold(pred_glm_v3, test_data_v3$rating_q1, t)
})
names(threshold_results) <- paste0("threshold_", thresholds_to_test)

#Build summary for thresholds
threshold_summary <- do.call(rbind, lapply(threshold_results, function(r) {
  data.frame(threshold = r$threshold, accuracy = r$accuracy,
             precision = r$precision, recall = r$recall)
}))
rownames(threshold_summary) <- NULL
threshold_summary

threshold_results$threshold_0.25$confusion_matrix
threshold_results$threshold_0.5$confusion_matrix
threshold_results$threshold_0.75$confusion_matrix


#Create naive model for comparison: always predicts majority class 0
pred_class_naive <- rep(0, nrow(test_data_v3))
conf_matrix_naive <- table(Predicted = factor(pred_class_naive, levels = c(0,1)),
                           Actual = test_data_v3$rating_q1)
conf_matrix_naive

accuracy_naive <- sum(diag(conf_matrix_naive)) / sum(conf_matrix_naive)
accuracy_naive

#Plot of drivers of rating drivers
#manually encode values

q2_drivers <- data.frame(
  predictor = c("Superhost status", "Instant bookable", "Number of reviews",
                "Host total listings count", "Bedrooms", "Amenity count",
                "Price per room", "Host identity verified"),
  odds_ratio = c(2.35, 0.73, 0.99, 0.997, 1.10, 1.008, 1.004, 0.58)
)
#Order by effect strength
q2_drivers$direction <- ifelse(q2_drivers$odds_ratio > 1, "Increases odds", "Decreases odds")
q2_drivers$predictor <- factor(q2_drivers$predictor,
                               levels = q2_drivers$predictor[order(abs(log(q2_drivers$odds_ratio)))])

ggplot(q2_drivers, aes(x = predictor, y = odds_ratio, fill = direction)) +
  geom_col(width = 0.65) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey") +
  coord_flip() +
  scale_fill_manual(values = c("Increases odds" = "#2166AC", "Decreases odds" = "#B2182B")) +
  labs(
    title = "Key Predictors of Achieving a Top-Quartile Rating",
    x = NULL,
    y = "Odds Ratio (reference line at 1 = no effect)",
    fill = NULL
  ) +
  theme_minimal(base_size = 12) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 13)
  )

#Regression -----


#Data Cleaning -- Q1 CODE NEEDS TO BE RUN FIRST --
#Q3 specific has _Q3
#Reuses Cleansed_data from Q1 section above (host_is_superhost,
#host_response_time, host_tenure_days, verification_count, amenity_count,
#and collapsed property_type are already built

#New excluded variables:
#Host_response_rate/Host_acceptance_rate -- Large no. of NAs (p1)
# Can introduce later

#host_neighbourhood -- Very large number of NAs

#bathrooms_text -- duplicate of number numeric bathrooms

#first_review, last_review

#host_since, host_verifications, ammenities -- replaced by numeric versions

drop_vars_Q3 <- c("X", "id", "bathrooms_text", "host_neighbourhood",
                  "host_since", "host_verifications", "amenities",
                  "host_response_rate", "host_acceptance_rate",
                  "first_review", "last_review",
                  "response_rate_missing_Q3", "acceptance_rate_missing_Q3")

model_data_Q3 <- Cleansed_data %>%
  select(-any_of(drop_vars_Q3))

nrow(model_data_Q3)
ncol(model_data_Q3)

#Split data into training and Test
#NOTE SEED 31423 - THIS IS INTENTIONAL
set.seed(31423)
train_index_Q3 <- sample(seq_len(nrow(model_data_Q3)), size = 0.7 * nrow(model_data_Q3))
train_data_Q3 <- model_data_Q3[train_index_Q3, ]
test_data_Q3 <- model_data_Q3[-train_index_Q3, ]

#START WITH GLM MODEL - LOG link (From assignment part 1)
#Also evidence from large skew in price per room (2.31)
#Skewness of log(price_per_room) becomes -0.02
skewness(Airbnb_data$price_per_room)
skewness(log(Airbnb_data$price_per_room))

#Create Baseline Lasso/ridge regression GLMs
#Entire Matrix Model

x_all_Q3 <- model.matrix(price_per_room ~ . , data = model_data_Q3)[, -1]
y_all_Q3 <- model_data_Q3$price_per_room

x_train_Q3 <- x_all_Q3[train_index_Q3, ]
y_train_Q3 <- y_all_Q3[train_index_Q3]
x_test_Q3 <- x_all_Q3[-train_index_Q3, ]
y_test_Q3 <- y_all_Q3[-train_index_Q3]

set.seed(31423)

cv_lasso_Q3 <- cv.glmnet(x_train_Q3, y_train_Q3, family = Gamma(link = "log"), alpha = 1, nfolds = 10)
plot(cv_lasso_Q3)
cv_lasso_Q3$lambda.min
cv_lasso_Q3$lambda.1se

set.seed(31423)

cv_ridge_Q3 <- cv.glmnet(x_train_Q3, y_train_Q3, family = Gamma(link = "log"), alpha = 0, nfolds = 10)

#Evaluate Ridge and Lasso using Test Data

pred_lasso_Q3 <- predict(cv_lasso_Q3, newx = x_test_Q3, s = "lambda.min", type = "response")
pred_ridge_Q3 <- predict(cv_ridge_Q3, newx = x_test_Q3, s = "lambda.min", type = "response")

#MSE FUNc
MSE <- function(data, prediction) {
  mean((data - prediction)^2)
}

(MSE_lasso_Q3 <- MSE(y_test_Q3, pred_lasso_Q3))
(MSE_ridge_Q3 <- MSE(y_test_Q3, pred_ridge_Q3))

#Naive Baseline -- predict mean(price per room) for everyone
naive_pred_Q3 <- mean(y_train_Q3)
(mse_naive_Q3 <- MSE(y_test_Q3, naive_pred_Q3))



#---- Introduction of host acceptance rate and host response rate: Clean the data to preserve 8000 obs
#convert string to numeric and remove percentage sign

Cleansed_data$host_response_rate <- str_remove(Cleansed_data$host_response_rate, "%")
Cleansed_data$host_response_rate <- as.numeric(Cleansed_data$host_response_rate)

Cleansed_data$host_acceptance_rate <- str_remove(Cleansed_data$host_acceptance_rate, "%")
Cleansed_data$host_acceptance_rate <- as.numeric(Cleansed_data$host_acceptance_rate)

#Create temp values for nas:
Cleansed_data$host_response_rate[is.na(Cleansed_data$host_response_rate)] <- -1
Cleansed_data$host_acceptance_rate[is.na(Cleansed_data$host_acceptance_rate)] <- -1

#Mark where missing data values are: 1 - missing, 0 - present -- #Creates new variable to test as well
Cleansed_data$response_rate_missing_Q3 <- ifelse(Cleansed_data$host_response_rate == -1, 1, 0)
Cleansed_data$acceptance_rate_missing_Q3 <- ifelse(Cleansed_data$host_acceptance_rate == -1, 1, 0)

drop_vars_Q3_v2 <- setdiff(drop_vars_Q3, c("host_response_rate", "host_acceptance_rate"))

model_data_Q3_v2 <- Cleansed_data %>%
  select(-any_of(drop_vars_Q3_v2))
model_data_Q3_v2 <- na.omit(model_data_Q3_v2)
nrow(model_data_Q3_v2)
#preserves all 8000 obs

#Create matrixes and train, test data split
x_all_Q3_v2 <- model.matrix(price_per_room ~ . , data = model_data_Q3_v2)[, -1]
y_all_Q3_v2 <- model_data_Q3_v2$price_per_room

set.seed(31423)
train_index_Q3_v2 <- sample(seq_len(nrow(model_data_Q3_v2)), size = 0.7 * nrow(model_data_Q3_v2))
train_data_Q3_v2 <- model_data_Q3_v2[train_index_Q3_v2, ]
test_data_Q3_v2  <- model_data_Q3_v2[-train_index_Q3_v2, ]

x_train_Q3_v2 <- x_all_Q3_v2[train_index_Q3_v2, ]
y_train_Q3_v2 <- y_all_Q3_v2[train_index_Q3_v2]
x_test_Q3_v2  <- x_all_Q3_v2[-train_index_Q3_v2, ]
y_test_Q3_v2  <- y_all_Q3_v2[-train_index_Q3_v2]

#Replace -1 placeholders with median values
response_rate_median <- median(train_data_Q3_v2$host_response_rate[train_data_Q3_v2$response_rate_missing_Q3 == 0])
acceptance_rate_median <- median(train_data_Q3_v2$host_acceptance_rate[train_data_Q3_v2$acceptance_rate_missing_Q3 == 0])

#apply to the model matrices (column names match variable names -- numeric)
x_train_Q3_v2[train_data_Q3_v2$response_rate_missing_Q3 == 1, "host_response_rate"] <- response_rate_median
x_test_Q3_v2[test_data_Q3_v2$response_rate_missing_Q3 == 1, "host_response_rate"]   <- response_rate_median
x_train_Q3_v2[train_data_Q3_v2$acceptance_rate_missing_Q3 == 1, "host_acceptance_rate"] <- acceptance_rate_median
x_test_Q3_v2[test_data_Q3_v2$acceptance_rate_missing_Q3 == 1, "host_acceptance_rate"]   <- acceptance_rate_median

#Refit LASSO/ridge Gamma GLM with the reintroduced variables
set.seed(31423)
cv_lasso_Q3_v2 <- cv.glmnet(x_train_Q3_v2, y_train_Q3_v2, family = Gamma(link = "log"), alpha = 1, nfolds = 10)
set.seed(31423)
cv_ridge_Q3_v2 <- cv.glmnet(x_train_Q3_v2, y_train_Q3_v2, family = Gamma(link = "log"), alpha = 0, nfolds = 10)

pred_lasso_Q3_v2 <- predict(cv_lasso_Q3_v2, newx = x_test_Q3_v2, s = "lambda.min", type = "response")
pred_ridge_Q3_v2 <- predict(cv_ridge_Q3_v2, newx = x_test_Q3_v2, s = "lambda.min", type = "response")

(MSE_lasso_Q3_v2 <- MSE(y_test_Q3_v2, pred_lasso_Q3_v2))
(MSE_ridge_Q3_v2 <- MSE(y_test_Q3_v2, pred_ridge_Q3_v2))

#compare directly against your existing baseline
MSE_lasso_Q3; MSE_lasso_Q3_v2
MSE_ridge_Q3; MSE_ridge_Q3_v2

#Proceed to fit GAM using train_data_Q3 and test_data_Q3

#Lasso slightly worse model than ridge but lasso is simpler
#Take variables from Q3 lasso coefficients -- reduces concurvity and removes redundant predictors (6 different min/max nights)
#has availability -- remove: 2 non-t options out of 8000 -- virtually 
#Availability_days -- only keep Availability_90 - only one that mattered in q1 model

lasso_coefs_Q3 <- coef(cv_lasso_Q3, s = "lambda.min")
selected_vars_Q3 <- rownames(lasso_coefs_Q3)[which(lasso_coefs_Q3 != 0)]
selected_vars_Q3 <- setdiff(selected_vars_Q3, "(Intercept)")

predictor_names_Q3 <- setdiff(names(model_data_Q3), c("price_per_room", "has_availability", "rating_q1"))
matched_vars_Q3 <- sapply(selected_vars_Q3, function(v) {
  matches <- predictor_names_Q3[sapply(predictor_names_Q3, function(p) startsWith(v, p))]
  matches[which.max(nchar(matches))]
})
final_vars_Q3 <- unique(unlist(matched_vars_Q3))

redundant_nights <- c("minimum_minimum_nights", "maximum_minimum_nights",
                      "minimum_maximum_nights", "maximum_maximum_nights",
                      "minimum_nights_avg_ntm")
redundant_availability <- c("availability_30", "availability_60", "availability_365")

final_vars_Q3 <- setdiff(final_vars_Q3, c(redundant_nights, redundant_availability))
final_vars_Q3

#Build GAM - to smooth continuous predictors -- need > 10 unique values
#Linear/ parametric terms for factors

build_gam_formula <- function(vars, data, response = "price_per_room", k = 10) {
  terms <- sapply(vars, function(v) {
    
    if (is.numeric(data[[v]]) && length(unique(data[[v]])) > 10) {
      paste0("s(", v, ", k = ", k, ")")
    } else {
      v
    }
  })
  
  as.formula(paste(response, "~", paste(terms, collapse = " + ")))
}

gam_formula_Q3 <- build_gam_formula(final_vars_Q3, train_data_Q3)
gam_formula_Q3

gam_Q3 <- gam(gam_formula_Q3, data = train_data_Q3, family = Gamma(link = "log"))
summary(gam_Q3)
gam.check(gam_Q3)          #checks basis dimension (k) adequacy -- watch for low p-values suggesting k too small
concurvity(gam_Q3, full = TRUE)  #GAM's analogue of VIF -- watch for values close to 1

#Evaluate on test set
pred_gam_Q3 <- predict(gam_Q3, newdata = test_data_Q3, type = "response")
(MSE_gam_Q3 <- MSE(y_test_Q3, pred_gam_Q3))

#compare against your existing baselines
MSE_lasso_Q3; MSE_ridge_Q3; MSE_gam_Q3

#MSE_gam_Q3 clearly the best model -- 
#Latitude and host_tenure days need fixing -- pvalues <<< 2e-16
#edfs close to 9 --> increase k to 20 

gam_formula_Q3_v2 <- update(gam_formula_Q3, . ~ . - s(latitude, k=10) - s(host_tenure_days, k=10)
                            + s(latitude, k=20) + s(host_tenure_days, k=20))
gam_Q3_v2 <- gam(gam_formula_Q3_v2, data = train_data_Q3, family = Gamma(link = "log"))
gam.check(gam_Q3_v2)

#increasing k seems to not have solved pvalues -- both latitude and host_tenure still <<< 2e-16
#Latitude and longitude model geographic location together -- model them bivariately
#Further increase k for host_tenure_days to 30

gam_formula_Q3_v3 <- update(gam_formula_Q3_v2, . ~ . - s(latitude, k=20) - s(longitude, k=10) - s(host_tenure_days, k=20)
                            + s(latitude, longitude, k = 10) + s(host_tenure_days, k=30))

gam_Q3_v3 <- gam(gam_formula_Q3_v3, data = train_data_Q3, family = Gamma(link = "log"))
gam.check(gam_Q3_v3)
pred_gam_Q3_v3 <- predict(gam_Q3_v3, newdata = test_data_Q3, type = "response")
(MSE_gam_Q3_v3 <- MSE(y_test_Q3, pred_gam_Q3_v3))


#host_tenure_days resolved -- increase k for bivariate longitude latitude
gam_formula_Q3_v4 <- update(gam_formula_Q3_v3, . ~ . - s(latitude, longitude, k=10)
                            + s(latitude, longitude, k = 30))

gam_Q3_v4 <- gam(gam_formula_Q3_v4, data = train_data_Q3, family = Gamma(link = "log"))
gam.check(gam_Q3_v4)
pred_gam_Q3_v4 <- predict(gam_Q3_v4, newdata = test_data_Q3, type = "response")
(MSE_gam_Q3_v4 <- MSE(y_test_Q3, pred_gam_Q3_v4))

#Proceed to XGboosting tree: 

#Reuse traininx   g data from glms -- x_train_Q3, y_train_Q3, x_test_Q3, y_test_Q3
d_train_Q3 <- xgb.DMatrix(data = x_train_Q3, label = y_train_Q3)
d_test_Q3 <- xgb.DMatrix(data = x_test_Q3, label = y_test_Q3)

#Tune with cross-validation -- assume that price per room follows gamma distribution
set.seed(31423)

xgb_params <- list(
  objective = "reg:gamma",
  max_depth = 6,
  eta = 0.05,
  subsample = 0.8,
  colsample_bytree = 0.8
)

xgb_cv_Q3 <- xgb.cv(
  data = d_train_Q3,
  params = xgb_params,
  nrounds = 1000,
  nfold = 10, 
  early_stopping_rounds = 20,
  verbose = 0
)

str(xgb_cv_Q3, max.level = 1)
names(xgb_cv_Q3)

head(xgb_cv_Q3$evaluation_log)
#Stops after 357 rounds - metric being tracked is test_gamma_deviance_mean

best_nrounds_Q3 <- which.min(xgb_cv_Q3$evaluation_log$test_gamma_deviance_mean)
best_nrounds_Q3     #337

set.seed(31423)
xgb_Q3 <- xgb.train(
  params = xgb_params,
  data = d_train_Q3,
  nrounds = best_nrounds_Q3
)

pred_xgb_Q3 <- predict(xgb_Q3, newdata = d_test_Q3)
(MSE_xgb_Q3 <- MSE(y_test_Q3, pred_xgb_Q3))

MSE_lasso_Q3; MSE_ridge_Q3; MSE_gam_Q3; MSE_xgb_Q3
#4244.57; 4221.918; 3469.924; 3264.435

#xgb improves MSE with untuned combination 
#Search for better combinations

grid_Q3 <- expand.grid(
  max_depth = c(3, 6, 9),
  eta = c(0.01, 0.05, 0.1)
)

grid_results_Q3 <- data.frame()

for (i in 1:nrow(grid_Q3)) {
  cat("Running combination", i, "of", nrow(grid_Q3),
      "-- max_depth =", grid_Q3$max_depth[i],
      ", eta =", grid_Q3$eta[i], "\n")
  
  set.seed(31423)
  cv_fit <- xgb.cv(
    data = d_train_Q3,
    params = list(objective = "reg:gamma",
                  max_depth = grid_Q3$max_depth[i],
                  eta = grid_Q3$eta[i],
                  subsample = 0.8, colsample_bytree = 0.8),
    nrounds = 1000, nfold = 10, early_stopping_rounds = 20, verbose = 0
  )
  
  best_iter <- which.min(cv_fit$evaluation_log$test_gamma_deviance_mean)
  best_dev  <- min(cv_fit$evaluation_log$test_gamma_deviance_mean)
  
  grid_results_Q3 <- rbind(grid_results_Q3,
                           data.frame(max_depth = grid_Q3$max_depth[i], eta = grid_Q3$eta[i],
                                      best_nrounds = best_iter, cv_deviance = best_dev))
}

grid_results_Q3 <- grid_results_Q3[order(grid_results_Q3$cv_deviance), ]
grid_results_Q3
#Best grid results -- max_depth 6, eta 0.05, nround 337 - original configuration

#Variable interpretation/ importance of best model
importance_Q3 <- xgb.importance(model = xgb_Q3)
head(importance_Q3, 15)

xgb.plot.importance(importance_Q3, top_n = 15)

#Large price outlier in train and test data could be causing rmse differences -- max 999 vs 787 from train test split
summary(y_train_Q3)
summary(y_test_Q3)
sd(y_train_Q3); sd(y_test_Q3)
max(y_train_Q3); max(y_test_Q3)


#Final models summary/ data representation
#Ridge

Q3_final_ridge <- cv_ridge_Q3
pred_ridge_train <- predict(Q3_final_ridge, newx = x_train_Q3, s = "lambda.min", type = "response")
pred_ridge_test <- predict(Q3_final_ridge, newx = x_test_Q3, s = "lambda.min", type = "response")

(train_rmse_ridge <- sqrt(MSE(y_train_Q3, pred_ridge_train)))
(test_rmse_ridge  <- sqrt(MSE(y_test_Q3, pred_ridge_test)))

#Lasso
Q3_final_lasso <- cv_lasso_Q3
pred_lasso_train <- predict(Q3_final_lasso, newx = x_train_Q3, s = "lambda.min", type = "response")
pred_lasso_test <- predict(Q3_final_lasso, newx = x_test_Q3, s = "lambda.min", type = "response")

(train_rmse_lasso <- sqrt(MSE(y_train_Q3, pred_lasso_train)))
(test_rmse_lasso  <- sqrt(MSE(y_test_Q3, pred_lasso_test)))

#GAM
Q3_final_gam <- gam_Q3
pred_gam_train <- predict(gam_Q3, newdata = train_data_Q3, type = "response")
pred_gam_test <- predict(gam_Q3, newdata = test_data_Q3, type = "response")

(train_rmse_gam <- sqrt(MSE(y_train_Q3, pred_gam_train)))
(test_rmse_gam  <- sqrt(MSE(y_test_Q3, pred_gam_test)))

#XGBM
Q3_final_xgbm <- xgb_Q3

#Rerun bc quirk of xgb (external pointers)
d_train_Q3 <- xgb.DMatrix(data = x_train_Q3, label = y_train_Q3)
d_test_Q3  <- xgb.DMatrix(data = x_test_Q3, label = y_test_Q3)

pred_xgb_train <- predict(Q3_final_xgbm, newdata = d_train_Q3)
pred_xgb_test <- predict(Q3_final_xgbm, newdata = d_test_Q3)

(train_rmse_xgb <- sqrt(MSE(y_train_Q3, pred_xgb_train)))
(test_rmse_xgb  <- sqrt(MSE(y_test_Q3, pred_xgb_test)))

model_comparison_Q3 <- data.frame(
  Model = c("LASSO (Gamma GLM)", "Ridge (Gamma GLM)", "GAM (Gamma, smooth terms)", "XGBoost"),
  Train_RMSE = c(train_rmse_lasso, train_rmse_ridge, train_rmse_gam, train_rmse_xgb),
  Test_RMSE  = c(test_rmse_lasso, test_rmse_ridge, test_rmse_gam, test_rmse_xgb)
)
model_comparison_Q3$Overfit_Gap <- model_comparison_Q3$Test_RMSE - model_comparison_Q3$Train_RMSE
model_comparison_Q3$Test_MSE <- model_comparison_Q3$Test_RMSE^2

#naive baseline for % reduction column
naive_rmse_Q3 <- sqrt(mse_naive_Q3)
model_comparison_Q3$Pct_Reduction_vs_Naive <- round(
  (1 - model_comparison_Q3$Test_MSE / mse_naive_Q3) * 100, 1
)

model_comparison_Q3

#Importance plot:
#Data preparation
importance_plot_data <- head(importance_Q3, 15)
importance_plot_data$Feature <- factor(importance_plot_data$Feature,
                                       levels = rev(importance_plot_data$Feature))

#Plot
ggplot(importance_plot_data, aes(x = Feature, y = Gain)) +
  geom_col(fill = "#2166AC", width = 0.65) +
  coord_flip() +
  labs(
    title = "Top 15 Predictors by Importance (XGBoost)",
    x = NULL,
    y = "Gain"
  ) +
  scale_y_continuous(labels = scales::percent_format(accuracy = 1)) +
  theme_minimal(base_size = 12) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 13),
  )

#Model Summaries (Q3)
#RIDGE
ridge_coefs_final <- as.matrix(coef(Q3_final_ridge, s = "lambda.min"))
ridge_coefs_final

#LASSO
lasso_coefs_final <- as.matrix(coef(Q3_final_lasso, s = "lambda.min"))
lasso_coefs_final

#GAM
summary(Q3_final_gam)

#XGBM

importance_Q3_full <- xgb.importance(model = Q3_final_xgbm)
importance_Q3_full

cat("XGBoost final model specification:\n")
cat("Objective:", xgb_params$objective, "\n")
cat("max_depth:", xgb_params$max_depth, "\n")
cat("eta:", xgb_params$eta, "\n")
cat("subsample:", xgb_params$subsample, "\n")
cat("colsample_bytree:", xgb_params$colsample_bytree, "\n")
cat("nrounds:", best_nrounds_Q3, "\n")

#EXCEL CODE SUBMISSION FOR Q5 ----------------------------------

library(openxlsx)
library(dplyr)
library(stringr)
library(xgboost)

test_raw <- read.xlsx("C:/Users/simon/Documents/3142_ass2/AirbnbTest.xlsx", sheet = 1, detectDates = TRUE)
wb <- loadWorkbook("C:/Users/simon/Documents/3142_ass2/AirbnbTest.xlsx")
nrow(test_raw)  #confirm 3865 -- must not change

test_clean <- test_raw

#host_is_superhost
test_clean$host_is_superhost <- as.character(test_clean$host_is_superhost)
test_clean$host_is_superhost[
  is.na(test_clean$host_is_superhost) |
    test_clean$host_is_superhost %in% c("", "N/A", "NA")
] <- "not_evaluated"

#host_response_time
#NOTE: read.xlsx() has no na.strings argument (unlike the read.csv() call used for
#training), so literal "N/A" text in the spreadsheet comes in as the STRING "N/A",
#not a true NA. Catch that explicitly here, mirroring the training-side logic.
test_clean$host_response_time <- as.character(test_clean$host_response_time)
test_clean$host_response_time[
  is.na(test_clean$host_response_time) |
    test_clean$host_response_time %in% c("", "N/A", "NA")
] <- "never_responds"

#host_since -> host_tenure_days -- REUSE training's snapshot_date, do not recompute
test_clean$host_since <- as.Date(test_clean$host_since)
test_clean$host_tenure_days <- as.numeric(snapshot_date - test_clean$host_since)

#host_verifications -> verification_count
test_clean$verification_count <- str_count(test_clean$host_verifications, ",") + 1

#amenities -> amenity_count
test_clean$amenity_count <- str_count(test_clean$amenities, ",") + 1

#property_type -- apply the SAME rare-level list computed from TRAINING data
test_clean$property_type <- as.character(test_clean$property_type)
test_clean$property_type[
  is.na(test_clean$property_type) |
    test_clean$property_type %in% c("", "N/A", "NA")
] <- "Other"
test_clean$property_type[test_clean$property_type %in% rare_property_types] <- "Other"
#also catch any property_type value never seen during training at all
test_clean$property_type[!(test_clean$property_type %in% levels(Cleansed_data$property_type))] <- "Other"

#Sweep check: scan every other factor variable for literal N/A / NA / "" strings ----
#read.xlsx() will not have converted these to true NA the way read.csv(na.strings="N/A")
#did on the training side, so anything missed here would silently turn into NAs once
#we force factor levels to match training in step 3.
factor_vars <- names(Cleansed_data)[sapply(Cleansed_data, is.factor)]
for (v in intersect(factor_vars, names(test_clean))) {
  bad <- test_clean[[v]] %in% c("N/A", "NA", "") | is.na(test_clean[[v]])
  if (any(bad)) {
    cat(v, ":", sum(bad), "literal N/A/blank/NA values found -- confirm these are handled above\n")
  }
}

#Force EVERY factor variable to match training's exact levels to prevent model.matrix from creating 
#mismatched dummy columns
for (v in intersect(factor_vars, names(test_clean))) {
  test_clean[[v]] <- factor(as.character(test_clean[[v]]), levels = levels(Cleansed_data[[v]]))
}

#check for NAs introduced by any unseen category (a value in test with no matching training level)
na_check <- sapply(test_clean[intersect(factor_vars, names(test_clean))], function(x) sum(is.na(x)))
na_check[na_check > 0]  #INSPECT THIS -- any nonzero entries need a decision before proceeding
#> named integer(0)

#Select the same predictor columns as model_data_Q3 (minus price_per_room, which doesn't exist here) ----
test_predictors <- test_clean %>% select(-any_of(c(drop_vars_Q3, "price_per_room")))
test_predictors <- test_predictors %>% select(all_of(setdiff(names(model_data_Q3), "price_per_room")))

#check for any remaining NAs before building the matrix
sum(is.na(test_predictors))
#> 0

#FIX FOR has_availability
#Force character/logical predictor columns to carry TRAINING's levels ----
#model.matrix() infers factor levels from whatever values are actually present in the
#data passed to it. Step 3 only re-levelled columns that were already factors in
#Cleansed_data -- but some predictors (e.g. has_availability) are character/logical and
#near-constant in training (has_availability is "t" for 7998/8000 training rows, "f" for
#only 2). In a smaller test batch it's entirely possible every row is "t", leaving only
#1 level and causing model.matrix() to error ("contrasts can be applied only to factors
#with 2 or more levels"). Explicitly set levels from model_data_Q3 for every such column
#so model.matrix() always builds the same dummy-column structure as x_train_Q3.
char_log_vars <- names(model_data_Q3)[sapply(model_data_Q3, function(x) is.character(x) || is.logical(x))]
char_log_vars <- setdiff(char_log_vars, "price_per_room")

for (v in intersect(char_log_vars, names(test_predictors))) {
  train_levels <- sort(unique(as.character(model_data_Q3[[v]])))
  test_predictors[[v]] <- factor(as.character(test_predictors[[v]]), levels = train_levels)
}

sapply(test_predictors[intersect(char_log_vars, names(test_predictors))], function(x) sum(is.na(x)))

#Build model matrix and align columns EXACTLY to x_train_Q3 ----
x_test_official <- model.matrix(~ . , data = test_predictors)[, -1]

missing_cols <- setdiff(colnames(x_train_Q3), colnames(x_test_official))
extra_cols   <- setdiff(colnames(x_test_official), colnames(x_train_Q3))
missing_cols  #should ideally be empty or explainable
extra_cols    #should be empty -- if not, factor levels didn't align correctly, go back to step 3

#pad any missing columns with 0 (a level present in training but absent from this test batch)
for (col in missing_cols) {
  x_test_official <- cbind(x_test_official, 0)
  colnames(x_test_official)[ncol(x_test_official)] <- col
}

#reorder to match training exactly
x_test_official <- x_test_official[, colnames(x_train_Q3)]
dim(x_test_official)  #should be 3865 x 103 > 3865 103

#Predict using your FINAL, already-fitted model (no retraining) ----
d_test_official <- xgb.DMatrix(data = x_test_official)
final_predictions <- predict(Q3_final_xgbm, newdata = d_test_official)

length(final_predictions)  #must equal 3865 - TRUE
sum(is.na(final_predictions))  #must be 0 - TRUE
summary(final_predictions)  #sanity check: values should look like plausible dollar prices, not negative/absurd

#Write predictions into column AY of the ORIGINAL workbook, preserving everything else ----
writeData(wb, sheet = 1, x = "z5690607", startCol = "AY", startRow = 1)
writeData(wb, sheet = 1, x = final_predictions, startCol = "AY", startRow = 2)

saveWorkbook(wb, "AirbnbTest_z5690607.xlsx", overwrite = TRUE)

