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
from load_data import get_data, get_params
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

transformer1 = StandardScaler().fit(data_raw)
data_processed = data_raw.copy()
data_processed.iloc[:,:] = transformer1.transform(data_raw)

save_loc = env+"/plots/surrogate/"
# %%
######################## PCA ##############################
var_exp = .95
# Keep up to __% explained
data_components,data_ws,_ = pru.get_pca_data(data_processed,var_exp)

print("For {} explained variance, {} components are selected.".format(var_exp, data_components.shape[1]))

# Show components (on raw data)
comp_components,comp_ws,_ = pru.get_pca_data(data_processed,.95)

## Split into train/test
data_train_input, data_test_input, data_train_output, data_test_output = train_test_split(data_params,data_ws,random_state=0)
train_preds, _, _, test_preds = train_test_split(data_processed,data_processed,random_state=0)

# %%
# Scree plot
plt.figure()
ncomps = 15
var_over_comps = [pru.get_pca_data(data_processed,var_exp,i)[2] for i in range(1,ncomps+1)]
sns.lineplot(x=range(1,ncomps+1), y=var_over_comps, marker='o')
plt.ylabel("Variance explained")
plt.xlabel("Number of PCA components")
plt.tight_layout()
plt.savefig(save_loc+"pca_scree.pdf")
plt.show()

# %%
# Compare data reconstructed with PCA with real data
cdata_train_input, cdata_test_input, cdata_train_output, cdata_test_output = train_test_split(data_params,data_ws,random_state=0)
compare_train_preds, _, _, compare_test_preds = train_test_split(data_processed,data_processed,random_state=0)
# scaled_test_preds = pru.inverse_transform_data(data_processed,transformer1)
# scaled_preds = data_processed.copy()
# scaled_preds.iloc[:,:] = pca_pred_data
comp_scaled_test_preds = pru.inverse_transform_data(compare_test_preds,transformer1)
comp_scaled_preds = data_processed.copy()
temp_components,temp_ws,_ = pru.get_pca_data(data_processed)
comp_scaled_preds.iloc[:,:] = pru.reconstruct_data_from_pca(temp_ws, temp_components, transformer1)
comp_scaled_preds = comp_scaled_preds.loc[comp_scaled_test_preds.index]

np.random.seed(0)
random_inds = np.random.choice(compare_test_preds.index,size=10,replace=False)
for i in random_inds:
  stp_i = comp_scaled_test_preds.loc[i]
  sp_i = comp_scaled_preds.loc[i]
  sp_err = pru.absolute_relative_err(stp_i,sp_i)

  prp.plot_data_comparisons(stp_i,sp_i,sp_err,variables,pset=None,savefile=save_loc+f"pca_reconstruct_{i}.pdf",show_plots=show_plots,pred_name="PCA Compression")

# %%
######################## RANDOM FOREST ##############################
load_name = env+"/results/surrogate/best_estimator.pkl"
with open(load_name, "rb") as file:
    rf1 = pickle.load(file)

ticks = pd.to_datetime(data_raw['hospitalizations'].columns)
# for i in range(data_raw.shape[0]):

# data_train_input, data_test_input, data_train_output, data_test_output = train_test_split(data_params,data_ws,random_state=0)
# train_preds, _, _, test_preds = train_test_split(data_processed,data_processed,random_state=0)
rf1.fit(cdata_train_input, cdata_train_output)
pred_ws = rf1.predict(cdata_test_input)
# temp_components,temp_ws,_ = pru.get_pca_data(compare_train_preds)
temp_components,temp_ws,_ = pru.get_pca_data(data_processed)
preds = pru.reconstruct_data_from_pca(pred_ws,data_components,transformer1)

scaled_test_preds = pru.inverse_transform_data(compare_test_preds,transformer1)
scaled_preds = scaled_test_preds.copy()
# scaled_preds.iloc[:,:] = pru.inverse_transform_data(preds,transformer1)
scaled_preds.iloc[:,:] = preds
scaled_err = pru.absolute_relative_err(scaled_test_preds,scaled_preds)

# np.random.seed(0)
# for i in np.random.choice(np.arange(test_preds.shape[0]),size=10):
for i in random_inds:
  stp_i = scaled_test_preds.loc[i]
  sp_i = scaled_preds.loc[i]

  prp.plot_data_comparisons(stp_i,sp_i,scaled_err.loc[i],variables,pset=None,savefile=save_loc+f"rf_reconstruct_{i}.pdf",show_plots=show_plots,pred_name="RF Reconstruction")
# %%
# Plot histogram of surrogate relative errors
ax = sns.histplot(scaled_err.T.median())
ax.set(xlabel="Median Absolute Relative Error")
plt.tight_layout()
plt.savefig(save_loc+f"rf_rel_err_hist.pdf")
plt.show()

