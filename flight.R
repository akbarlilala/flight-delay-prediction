
library(dplyr)
library(readr)

flights <- read_csv("~/Desktop/flight_data_2018_2024.csv")
library(randomForest)

# Select necessary columns
flights_selected <- flights %>%
  select(FlightDate, Marketing_Airline_Network, Origin, Dest,
         CRSDepTime, DepDelay, Cancelled, Diverted)

# Filter only completed flights
flights_clean <- flights_selected %>%
  filter(Cancelled == 0, Diverted == 0, !is.na(DepDelay), !is.na(CRSDepTime))

flights_clean <- flights_clean %>%
  mutate(
    CRSDepTime = as.numeric(CRSDepTime),
    DepHour = as.integer(CRSDepTime / 100),
    SignificantDelay = ifelse(DepDelay > 15, 1, 0)
  )

# Select columns
flights_final <- flights_clean %>%
  select(SignificantDelay, Marketing_Airline_Network, Origin, Dest, DepHour)

# Sample 50,000 rows
set.seed(123)
flights_sampled <- sample_n(flights_final, 50000)

# Train/test split
set.seed(42)
train_indices <- sample(seq_len(nrow(flights_sampled)), size = 0.7 * nrow(flights_sampled))
train_data <- flights_sampled[train_indices, ]
test_data <- flights_sampled[-train_indices, ]

# Reduce levels 
top_airlines <- names(sort(table(train_data$Marketing_Airline_Network), decreasing = TRUE))[1:10]
top_origins <- names(sort(table(train_data$Origin), decreasing = TRUE))[1:20]
top_dests <- names(sort(table(train_data$Dest), decreasing = TRUE))[1:20]

train_filtered <- train_data %>%
  filter(Marketing_Airline_Network %in% top_airlines,
         Origin %in% top_origins,
         Dest %in% top_dests)

test_filtered <- test_data %>%
  filter(Marketing_Airline_Network %in% top_airlines,
         Origin %in% top_origins,
         Dest %in% top_dests)

# Convert categorical vars to factors
train_filtered <- train_filtered %>%
  mutate(
    SignificantDelay = as.factor(SignificantDelay),
    Marketing_Airline_Network = as.factor(Marketing_Airline_Network),
    Origin = as.factor(Origin),
    Dest = as.factor(Dest),
    DepHour = as.numeric(DepHour)
  )

test_filtered <- test_filtered %>%
  mutate(
    SignificantDelay = as.factor(SignificantDelay),
    Marketing_Airline_Network = as.factor(Marketing_Airline_Network),
    Origin = as.factor(Origin),
    Dest = as.factor(Dest),
    DepHour = as.numeric(DepHour)
  )

# Train Random Forest model
rf_model <- randomForest(
  SignificantDelay ~ Marketing_Airline_Network + Origin + Dest + DepHour,
  data = train_filtered,
  ntree = 100,
  importance = TRUE
)

pred_rf <- predict(rf_model, newdata = test_filtered)

# Confusion matrix
confusion <- table(Predicted = pred_rf, Actual = test_filtered$SignificantDelay)
print(confusion)

# Accuracy
accuracy <- sum(diag(confusion)) / sum(confusion)
cat("Accuracy:", round(accuracy * 100, 2), "%\n")


# Plot
par(mar = c(5, 8, 4, 2))
varImpPlot(rf_model, 
           main = "Variable Importance",
           pch = 16, 
           col = "darkblue", 
           cex = 0.8,
           n.var = min(20, ncol(rf_model$importance))) 


