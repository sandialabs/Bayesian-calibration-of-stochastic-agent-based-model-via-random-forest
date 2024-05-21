library(scoringutils)
env <- Sys.getenv("PROJLOC")
source(paste0(env, "src/forConnor/utils/load_data.R"))

smoothing <- 7

# -- Get arguments if they exist
pfile1 <- paste0(env, "/results/calibration/calibration_predictions_h.csv")
prediction_data1 <- read.csv(pfile1, header = FALSE)
pfile2 <- paste0(env, "/results/calibration/calibration_predictions_d.csv")
prediction_data2 <- read.csv(pfile2, header = FALSE)
dlen <- ncol(prediction_data1)
prediction_data1 <- as.matrix(prediction_data1)
prediction_data2 <- as.matrix(prediction_data2)

data_real <- read.csv(paste0(env, "/data/observed_chicago.csv"), header = TRUE, stringsAsFactors = FALSE)
data_real <- na.omit(data_real)
data_real_temp <- data_real
data_real <- data_real[2:nrow(data_real), ]
data_real$hospitalizations <- diff(data_real_temp$hospitalizations)
data_real$deaths <- diff(data_real_temp$deaths)

# %%
## ---- arrange data into data.frame for scoreutils
df_len <- prod(dim(prediction_data1))
df_continuous <- data.frame(
  target_end_date = character(),
  target_type = character(),
  sample = integer(),
  prediction = double(),
  true_value = double()
)
pb <- txtProgressBar(min = 0, max = df_len)
count <- 0
for (i in seq_len(dim(prediction_data1)[1])) {
  for (j in 1:dlen) {
    df_continuous[nrow(df_continuous) + 1, 1] <- data_real$date[j]
    df_continuous[nrow(df_continuous), 2] <- "hosps"
    df_continuous[nrow(df_continuous), 3] <- i
    df_continuous[nrow(df_continuous), 4] <- prediction_data1[i, j]
    df_continuous[nrow(df_continuous), 5] <- data_real$hospitalizations[j]
    df_continuous[nrow(df_continuous) + 1, 1] <- data_real$date[j]
    df_continuous[nrow(df_continuous), 2] <- "deaths"
    df_continuous[nrow(df_continuous), 3] <- i
    df_continuous[nrow(df_continuous), 4] <- prediction_data2[i, j]
    df_continuous[nrow(df_continuous), 5] <- data_real$deaths[j]
    count <- count + 1
    setTxtProgressBar(pb, count)
  }
}
close(pb)

# %%
# ---- get all scores
all_scores <- score(df_continuous)
print(all_scores)

# ---- write all scores
sfile <- paste0(env, "/results/calibration/scores.csv")
write.csv(all_scores, sfile)
