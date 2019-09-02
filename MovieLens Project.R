
################################
# Create edx set, validation set
################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################
# Initial Setup complete
################################


# Create train and test subsets from the edx set
# Test set will be 20% of the edx set
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1,p = 0.2, list = FALSE)
train_set <- edx[-test_index,]
test_set <- edx[test_index,]

# Make sure userId and movieId in test set are also in train set
test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Calculating mean rating in train set
mu <- mean(train_set$rating)

# Computing b_i for each movie as mean of residuals
b_i <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))


# Computing b_u for each user as mean of residuals
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/n())

# Computing b_t for each day as mean of residuals
b_t <- train_set %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_i, by = "movieId") %>% 
  mutate(time=as_datetime(timestamp)) %>% #Converting epoch time to human readable time
  group_by(day=floor_date(time,"day")) %>% #rounding date-time down to nearest day
  summarise(b_t=sum(rating-b_i-b_u-mu)/n())

# Computing b_g for each genre as mean of residuals and also
# Using cross validation to determine the optimal standard Error cut off value
# So that genres that have high standard errors on their mean ratings are not assigned a b_g value
ses <- seq(0,1,0.1)
rmses <- sapply(ses, function(s){
  b_g <- train_set %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_i, by = "movieId") %>% 
    mutate(day = floor_date(as_datetime(timestamp),"day")) %>% #rounding date-time down to nearest day
    left_join(b_t, by="day") %>%
    mutate(b_t=replace_na(b_t,0)) %>% #Replacing NA values with 0
    group_by(genres) %>%
    summarise(b_g=(sum(rating-b_i-b_u-b_t-mu))/(n()),se = sd(rating)/sqrt(n())) %>%
    filter(se<=s) # Retaining b_g values that correspond to Standard Error less than or equal to S 
  
  # Making predictions on the test set
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(day = floor_date(as_datetime(timestamp),"day")) %>% #rounding date-time down to nearest day
    left_join(b_t, by="day") %>%
    mutate(b_t=replace_na(b_t,0)) %>% #Replacing NA values with 0
    left_join(b_g, by="genres") %>%
    mutate(b_g=replace_na(b_g,0)) %>% #Replacing NA values with 0
    mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
    .$pred
  
  # Calculating RMSE for predictions made on test subset
  RMSE(predicted_ratings, test_set$rating)
  return(RMSE(predicted_ratings, test_set$rating))
})

# Storing the optimal standard error error value i.e the one which gives lowest RMSE
s_e <- ses[which.min(rmses)]

#performing cross-validation using different lambda values
# This step can take over 5 minutes
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  
  # Computing b_i for each movie as mean of residuals
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  # Computing b_u for each user as mean of residuals
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  # Computing b_t for each day as mean of residuals
  b_t<-train_set %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_i, by = "movieId") %>% 
    mutate(time=as_datetime(timestamp)) %>% #rounding date-time down to nearest day
    group_by(day=floor_date(time,"day")) %>% 
    summarise(b_t=sum(rating-b_i-b_u-mu)/(n()+l))
  
  # Computing b_g for each genre as mean of residuals and also
  b_g<-train_set %>% 
    left_join(b_u, by="userId") %>% 
    left_join(b_i, by = "movieId") %>% 
    mutate(day=floor_date(as_datetime(timestamp),"day")) %>% #rounding date-time down to nearest day
    left_join(b_t, by="day") %>%
    mutate(b_t=replace_na(b_t,0)) %>% #Replacing NA values with 0
    group_by(genres) %>%
    summarise(b_g=(sum(rating-b_i-b_u-b_t-mu))/(n()+l),se = sd(rating)/sqrt(n())) %>%
    filter(se<=s_e)
  
  # Making predictions on the test set
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(day=floor_date(as_datetime(timestamp),"day")) %>% #rounding date-time down to nearest day
    left_join(b_t, by="day") %>%
    mutate(b_t=replace_na(b_t,0)) %>% #Replacing NA values with 0
    left_join(b_g, by="genres") %>%
    mutate(b_g=replace_na(b_g,0)) %>% #Replacing NA values with 0
    mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
    .$pred
  
  # Calculating RMSE for predictions made on test subset
  RMSE(predicted_ratings, test_set$rating)
  return(RMSE(predicted_ratings, test_set$rating))
})

# Storing the optimal Lambda value i.e the one which gives lowest RMSE
lambda <- lambdas[which.min(rmses)]

###############################################
# Training final model using edx dataset

mu <- mean(edx$rating) # Calculating mean rating in edx set

# Computing b_i for each movie as mean of residuals
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = sum(rating - mu)/(n()+lambda))

# Computing b_u for each user as mean of residuals
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))

# Computing b_t for each day as mean of residuals
b_t <-edx %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_i, by = "movieId") %>% 
  mutate(time=as_datetime(timestamp)) %>% 
  group_by(day=floor_date(time,"day")) %>% 
  summarise(b_t=sum(rating-b_i-b_u-mu)/(n()+lambda))

# Computing b_g for each genre as mean of residuals and
# Removing b_g for genres that have Standard Errors on mean ratings higher than cutoff
b_g <- edx %>% 
  left_join(b_u, by="userId") %>% 
  left_join(b_i, by = "movieId") %>% 
  mutate(day=floor_date(as_datetime(timestamp),"day")) %>% #rounding date-time down to nearest day
  left_join(b_t, by="day") %>%
  mutate(b_t=replace_na(b_t,0)) %>% #Replacing NA values with 0
  group_by(genres) %>%
  summarise(b_g=(sum(rating-b_i-b_u-b_t-mu))/(n()),se = sd(rating)/sqrt(n())) %>%
  filter(se<=s_e)

#########################################################################
# Predicitng the ratings on the validation set
predicted_ratings <- validation %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  mutate(day=floor_date(as_datetime(timestamp),"day")) %>% #rounding date-time down to nearest day
  left_join(b_t, by="day") %>%
  mutate(b_t=replace_na(b_t,0)) %>% #Replacing NA values with 0
  left_join(b_g, by="genres") %>%
  mutate(b_g=replace_na(b_g,0)) %>% #Replacing NA values with 0
  mutate(pred = mu + b_i + b_u + b_t + b_g) %>%
  .$pred

# Calculating RMSE for predictions made on validation set    
RMSE_final <- RMSE(predicted_ratings, validation$rating)
RMSE_final
