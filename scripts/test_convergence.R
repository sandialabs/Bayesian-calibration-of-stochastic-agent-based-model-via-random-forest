library(mcgibbsit)
env <- Sys.getenv("PROJLOC")

# Options
ifile <- paste0(env, "/results/calibration/calibration.csv")
print(paste("loading from:", ifile))

# %%
d <- read.csv(ifile, header = TRUE)
d <- as.matrix(d)
nc <- ncol(d)
d <- d[, 1:(nc - 2)]

# %%
## ---- do convergence analysis
out <- mcgibbsit(d, q = 0.5, r = 0.025)
print(out)

# %%
## ---- save to file
sfile <- paste0(env, "/results/calibration/convergence.csv")
print(paste("writing to:", sfile))
write.table(out$resmatrix, sfile)

# %%

