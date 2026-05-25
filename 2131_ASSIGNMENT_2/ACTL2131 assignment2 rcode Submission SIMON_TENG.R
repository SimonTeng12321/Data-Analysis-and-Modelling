#TESLA 2020-2025

SMP500_DATA <- read.table(file="C:/Users/simon/Downloads/sp500_adj_close.csv", header=TRUE, sep=',')
install.packages("dplyr")
install.packages("ggplot2")
install.packages("moments")
library(dplyr)
library(ggplot2)
library(moments)

DATES_DATA <- as.Date(SMP500_DATA$Date[756:2014])
TESLA_DATA <- SMP500_DATA$TSLA[756:2014]
SMP_ETF_DATA <- SMP500_DATA$SPY[756:2014]

#PART 1 LOG RETURNS
log_returns <- function(returns_vector) {
  return(diff(log(returns_vector)))
}

TESLA_RETURNS <- log_returns(TESLA_DATA)
SMP_ETF_RETURNS <- log_returns(SMP_ETF_DATA)

#plotting
tesla_plot_data <- data.frame(
  Date = DATES_DATA[2:1259],
  Returns = TESLA_RETURNS
)

ggplot(tesla_plot_data, aes(x = Date, y = Returns)) + 
  geom_point() +
  labs(
    title = "Returns of Tesla Stock",
    x = "Calendar Dates (Years)", 
    y = "Log Returns",
  ) + 
  scale_x_date(date_breaks = "year", date_labels = "%Y - %m") +
  theme_bw()

smp_plot_data <- data.frame(
  Date = DATES_DATA[2:1259],
  Returns = SMP_ETF_RETURNS
)

ggplot(smp_plot_data, aes(x = Date, y = Returns)) + 
  geom_point() +
  labs(
    title = "Returns of S&P",
    x = "Calendar Dates (Years)", 
    y = "Log Returns",
  ) + 
  scale_x_date(date_breaks = "year", date_labels = "%Y - %m") +
  theme_bw()

#PART 2 S&P 500 ETF vs Tesla: Summary statistics

summary_stats <- data.frame(
  stats = c("Mean", "Variance", "Skewness", "Kurtosis"),
  Tesla = c(
    mean(TESLA_RETURNS),
    var(TESLA_RETURNS),
    skewness(TESLA_RETURNS),
    kurtosis(TESLA_RETURNS)
  ),
  SMP_ETF = c(
    mean(SMP_ETF_RETURNS),
    var(SMP_ETF_RETURNS),
    skewness(SMP_ETF_RETURNS),
    kurtosis(SMP_ETF_RETURNS)
  )
)
print(summary_stats)

#PART 3: Histograms
#Tesla
hist(TESLA_RETURNS, freq = F, xlab = "Log Return", col = "red", breaks = 200, 
     main = "Tesla Log returns with Normal Distribution fitted")
curve(dnorm(x, mean = mean(TESLA_RETURNS), sd = sqrt(var(TESLA_RETURNS))), 
      add = T, col = "blue", lwd = 2)
grid()

#S&P
hist(SMP_ETF_RETURNS, freq = F, xlab = "Log Return", col = "green", 
     breaks = 200, main = "S&P Log returns with Normal Distribution fitted")
curve(dnorm(x, mean = mean(SMP_ETF_RETURNS), sd = sqrt(var(SMP_ETF_RETURNS))), 
      add = T, col = "blue", lwd = 2)
grid()


#PART 4: 
#Q4 CODE SUMMARY:
#1. Creates QQ plots for Tesla and S&P returns with the normal distribution
#2. Create initial t-distribution parameters for both datasets
#3. Use initial t-distribution paramters to create a log likelihood function
#4. Use MLE to optimise the t-distribution parameters to find the best fitting
#   t distribution for both datasets
#5. Create QQ-plots between the returns data and the optimised t-distributions
#6. Perform Chi-squared Goodness of fit test 

#Create QQ-plots for both datasets
qqnorm(TESLA_RETURNS,
       main = "QQ Plot of Tesla Log Returns",
       xlab = "Theoretical Quantiles",
       ylab = "Sample Quantiles",
       col = "red",
       pch = 4
       )
qqline(TESLA_RETURNS,
       col = "blue",
       lwd = 2)
grid()

qqnorm(SMP_ETF_RETURNS,
       main = "QQ Plot of S&P Log Returns",
       xlab = "Theoretical Quantiles",
       ylab = "Sample Quantiles",
       col = "green",
       pch = 4
)
qqline(SMP_ETF_RETURNS,
       col = "blue",
       lwd = 2)
grid()

#Makes initial parameters for student t-distribution to be optimised
start_params_func <- function(data) {
  parameters <- numeric(3)
  parameters[1] = mean(data)
  parameters[2] = sd(data)
  parameters[3] = 5
  return(parameters)
}

#Creates the log-likelihood function of the distributions to be optimised. 
#Returns negative of the actual log-likelihood function
neg_log_likelihood <- function(params, data) {
  mu <- params[1]
  sd <- params[2]
  df <- params[3]
  
  if(sd <= 0 || df <= 0) {
    #to prevent nan errors - prevents optimiser from taking negative values
    return(1e10)
  }
  
  log_likelihood <- sum(dt(((data - mu)/sd), df = df, log = TRUE) - log(sd))
                        
  #We need to return -log likelihood since the optimise function minimises and
  #we need to maximise for MLE
  return(-log_likelihood)
}

optimise_t <- function(data, start_parameters) {
  mle_result <- optim(
    par = start_parameters,
    fn = neg_log_likelihood,
    data = data,
    method = "BFGS",
    hessian = T
  )
  
  return(mle_result)
}
#Computing the MLE
TESLA_t_START_PARAMS <- start_params_func(TESLA_RETURNS)
TESLA_MLE <- optimise_t(TESLA_RETURNS, TESLA_t_START_PARAMS)
TESLA_t_END_PARAMS <- TESLA_MLE$par

SMP_t_START_PARAMS <- start_params_func(SMP_ETF_RETURNS)
SMP_MLE <- optimise_t(SMP_ETF_RETURNS, SMP_t_START_PARAMS)
SMP_t_END_PARAMS <- SMP_MLE$par

#Create the QQ plot with the optimised T-distribution variables:
t_qq_plot <- function(data, params) {
  mu <- params[1]
  sigma <- params[2]
  df <- params[3]
  
  #standardise data again using the parameters estimated by the MLE
  standardised <- (data - mu) / sigma
  
  #generate quantiles for t distribution
  n <- length(data)
  probs <- seq(1/(n+1), n/(n+1), length.out = n)
  theoretical <- qt(probs, df = df)
  
  #sort data into quantiles
  sample_quantiles <- sort(standardised)
  
  plot(theoretical , sample_quantiles, main= "QQ Plot - Optimised Student T vs Data",
       xlab = "Theoretical t Quantiles", 
       ylab = "Sample Quantiles", 
       col = "green")
  #red for Tesla, Green for S&P
  #reference line
  q_sample <- quantile(sample_quantiles, c(0.25, 0.75))
  q_theory <- qt(c(0.25, 0.75), df = df)
  slope <- diff(q_sample / diff(q_theory))
  intercept <- q_sample[1] - slope * q_theory[1]
  abline(intercept, slope, col = "blue", lwd = 2)
  grid()
}
t_qq_plot(TESLA_RETURNS, TESLA_t_END_PARAMS)
t_qq_plot(SMP_ETF_RETURNS, SMP_t_END_PARAMS)

#perform chi squared goodness of fit test to assess the t distribution fit
chi_sq_gof_t <- function(data, params, num_bins = 50) {
  mu    <- params[1]
  sigma <- params[2]
  nu    <- params[3]
  n     <- length(data)
  
  #creating quantile bins:
  probs     <- seq(0, 1, length.out = num_bins + 1)
  bin_edges <- qt(probs, df = nu) * sigma + mu
  
  #Calculate bin frequencies from observations
  observed <- hist(data, breaks = bin_edges, plot = FALSE)$counts
  
  #Step 3 — compute expected frequencies
  #each bin has equal probability by construction so 
  #expected count = n / num_bins for each bin
  expected <- rep(n / num_bins, num_bins)
  
  #Chi_sq_hypothesis test
  chi_sq_result <- chisq.test(observed, p = rep(1/num_bins, num_bins))
  
  return(chi_sq_result)
}
#perform chi_sq goodness of fit test
tesla_chi_sq <- chi_sq_gof_t(TESLA_RETURNS, TESLA_t_END_PARAMS, num_bins = 50)
SMP_chi_sq <- chi_sq_gof_t(SMP_ETF_RETURNS, SMP_t_END_PARAMS, num_bins = 50)
tesla_chi_sq
SMP_chi_sq

#PART 5:
#1. Perform f_test to check whether population variances can be assumed as equal
#2. Regardless of f_test result, perform two sample of means test
#f test
f_test_result <- var.test(TESLA_RETURNS, SMP_ETF_RETURNS,
                          alternative = "two.sided", conf.level = 0.95)
print(f_test_result)

#two sample mean test
two_sample_mean_test <- t.test(TESLA_RETURNS * 250,
                            SMP_ETF_RETURNS * 250, alternative = "two.sided", 
                            var.equal = T, conf.level = 0.95)

print(two_sample_mean_test)

#PART 6: Hypothesis test
#H0, annualised average return = 7%
#H1, annualised average return > 7% 

tesla_annualised_return <- mean(TESLA_RETURNS)*250
SMP_ETF_annualised_return <- mean(SMP_ETF_RETURNS)*250

t_test_statistic <- t.test(TESLA_RETURNS * 250, 
                           mu = 0.07, 
                           alternative = "greater",
                           conf.level = 0.95)
print(t_test_statistic)
