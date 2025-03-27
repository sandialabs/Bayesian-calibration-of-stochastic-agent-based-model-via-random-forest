import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import pickle

import os
env = os.environ["PROJLOC"]

plt.style.use(env+"/src/utils/rf_mcmc.mplstyle")
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

import sys
sys.path.append(env + "/src/utils")
from load_data import get_data, get_params, get_real_data
sys.path.append(env + "/src/surrogate")
import plots as prp
import utils as pru

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

show_plots = True

# %%
######################## LOAD DATA ##############################
data_raw = get_data(abc=False)
data_params = get_params(abc=False, drop_vars=True)

variables = data_raw.columns.get_level_values(0).unique()

transformer1 = StandardScaler().fit(data_params)
params = data_params.copy()
params.iloc[:,:] = transformer1.transform(params)

save_loc = env+"/plots/stein_calibration_gaussian_process/surrogate/"
res_loc = env+"/results/stein_calibration_gaussian_process/surrogate/"

## Split into train/test
data_train_input, data_test_input, data_train_output, data_test_output = train_test_split(params,data_raw,random_state=0)

# %%
######################## GAUSSIAN PROCESS ##############################
load_name = res_loc + "best_estimator.pkl"
with open(load_name, "rb") as file:
    gp1 = pickle.load(file)
gp1.fit(data_train_input, data_train_output)

# How well do we do in recovering the data compared to linear_regression?
preds = gp1.predict(data_test_input)

# How well do we do in recovering the actual data?
true_data = data_test_output

rel_err = pru.absolute_relative_err(true_data, preds)
_,rel_err_median,_ = pru.err_statistics(rel_err)

# Analyze parameters
print("R^2 (data): {}".format(r2_score(true_data,preds)))
print("Median rel abs err: \n{}".format(rel_err_median))

# %%
# Example of reconstruction accuracy
output = "deaths"
# output = "hospitalizations"

# Fit random forest
gp1.fit(params, data_raw)

gp_pred_data = gp1.predict(params)

true_data = data_raw
abs_rel_err = pru.absolute_relative_err(true_data,gp_pred_data)[output]

param_five,param_med,param_nfive = pru.err_statistics(abs_rel_err,axis=0)
time_five,time_med,time_nfive = pru.err_statistics(abs_rel_err,axis=1)

prp.plot_statistics_over_axes(param_five,param_med,param_nfive,time_five,time_med,time_nfive,0,savefile=save_loc+"gp_err_stats.pdf",show_plots=show_plots, xlabel="Date")

# %%
ticks = pd.to_datetime(data_raw['hospitalizations'].columns)

gp1.fit(data_train_input, data_train_output)
preds = gp1.predict(data_test_input)

scaled_test_preds = data_test_output
scaled_preds = scaled_test_preds.copy()
scaled_preds.iloc[:,:] = preds
scaled_err = pru.absolute_relative_err(scaled_test_preds,scaled_preds)

np.random.seed(0)
for i in np.random.choice(data_test_output.index,size=10,replace=False):
  stp_i = scaled_test_preds.loc[i]
  sp_i = scaled_preds.loc[i]

  prp.plot_data_comparisons(stp_i,sp_i,scaled_err.loc[i],variables,pset=None,savefile=save_loc+f"gp_reconstruct_{i}.pdf",show_plots=show_plots,pred_name="GP Reconstruction")
# %%
# Plot histogram of surrogate relative errors
ax = sns.histplot(scaled_err.T.median())
ax.set(xlabel="Median Absolute Relative Error")
plt.tight_layout()
plt.savefig(save_loc+"gp_rel_err_hist.pdf")
plt.show()
