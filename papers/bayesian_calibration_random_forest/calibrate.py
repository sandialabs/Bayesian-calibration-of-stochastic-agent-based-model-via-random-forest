import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings(action="ignore",category=UserWarning)
warnings.filterwarnings(action="ignore",category=RuntimeWarning)

import os
env = os.environ["PROJLOC"]

from pymcmcstat.MCMC import MCMC
from pymcmcstat.propagation import observation_sample

import sys
sys.path.append(env+"/src/utils")
from load_data import get_data, get_params, get_real_data

smoothing = 7
prediction_samples = 500
nsimu = 50000

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
data_real = get_real_data()
data_real = data_real.diff()
train_data = get_data(abc=False)
parameters = get_params(abc=False, drop_vars=True)
hosp_len = data_real.shape[0]
comparison_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.to_numpy()

comparison_data = comparison_data[:,1:]
comparison_data[0] = np.convolve(comparison_data[0], np.ones(smoothing)/smoothing, 'same')
comparison_data[1] = np.convolve(comparison_data[1], np.ones(smoothing)/smoothing, 'same')
res_loc = env+"/results/bayesian_calibration_random_forest/"

# Load model
surr_loc = res_loc + "surrogate/surrogate.pkl"
with open(surr_loc, "rb") as file:
    loaded_model = pickle.load(file)
smodel = loaded_model["rf"]
pca_components = loaded_model["pca_components"]
transformer = loaded_model["transformer"]
pca_components = loaded_model["pca_components"]

# Restrict pca components to when we have data
full_time = pd.to_datetime(train_data.columns.get_level_values(1))
times = np.arange(full_time.unique().shape[0])
data_time = pd.to_datetime(data_real.index)
time_mask = np.array([True if (ft in data_time) else False for ft in full_time])

# Define surrogate model function
pca_components_all_time = np.zeros(len(pca_components))
pca_components = pca_components[time_mask]
def surrogate_func(ps):
    coefs = smodel.predict(ps.reshape((1,-1)))
    pca_components_all_time[time_mask] = pca_components @ coefs.flatten()
    raw_result = transformer.inverse_transform(pca_components_all_time.reshape((1,-1)))
    result = raw_result.flatten()[time_mask]
    result = np.vstack((result[:hosp_len], result[hosp_len:]))
    result = np.diff(result)
    return result

def ssfun(ps,data):
    res = np.array(data.ydata).squeeze() - surrogate_func(ps)
    return (res**2).sum(axis=1)

def prior_function(ps,low,high):
    return np.zeros(1)

# %%
# ------------------------------------------------------------------------------ MCMC
# ------------------------------------------------------------------------------
sampler = MCMC()
x = np.arange(comparison_data.shape[1])
y = comparison_data
sampler.data.add_data_set(x,y[0])
sampler.data.add_data_set(x,y[1])

# Find MLE from ABM runs
errs = np.linalg.norm(train_data.diff(axis=1).loc[:,time_mask].iloc[:,2:] - comparison_data.flatten(), axis=1)
ps0_loc = np.argsort(errs)[0]
ps0 = parameters.iloc[ps0_loc].to_numpy()

# Add parameters
for i in range(parameters.shape[1]):
    sampler.parameters.add_model_parameter(
        name = parameters.columns[i],
        theta0 = ps0[i],
        minimum = parameters.iloc[:,i].min(),
        maximum = parameters.iloc[:,i].max(),
    )

# Simulation setup
sampler.simulation_options.define_simulation_options(
    nsimu = nsimu,
    updatesigma = 1,
    results_filename = res_loc + "calibration/calibrator.json",
    save_to_json = True,
)
S2_ols = ssfun(ps0,sampler.data) / (len(comparison_data) - parameters.shape[1])
N0 = np.ones(2)
N = comparison_data.shape[1]
sampler.model_settings.define_model_settings(
    sos_function = ssfun,
    prior_function = prior_function,
    N = N,
    N0 = N0,
    S20 = S2_ols,
    sigma2 = S2_ols
)

# Sample
sampler.run_simulation()

# %%
# Save CSV
results = sampler.simulation_results.results
chain = np.array(results['chain'])
s2chain = np.array(results['s2chain'])
sschain = np.array(results['sschain'])
save_data = np.hstack((chain, s2chain, sschain))
df = pd.DataFrame(save_data, columns=parameters.columns.to_list() + ["sigma2_h", "sigma2_d", "sumofsquares_h", "sumofsquares_d"])
df.to_csv(res_loc+"calibration/calibration.csv",index=False)

# Save predictions
final_samples = chain[-prediction_samples:,:]
posterior_samples = pd.DataFrame(final_samples, columns=parameters.columns)
final_s2 = s2chain[-prediction_samples:]
predictions = [observation_sample(final_s2[i].reshape((-1,1)), surrogate_func(final_samples[i]), 0) for i in range(prediction_samples)]
predictions_h = pd.DataFrame(np.array(predictions)[:,0,:])
predictions_d = pd.DataFrame(np.array(predictions)[:,1,:])
predictions_h.to_csv(res_loc+"calibration/calibration_predictions_h.csv",index=False,header=False)
predictions_d.to_csv(res_loc+"calibration/calibration_predictions_d.csv",index=False,header=False)
posterior_samples.to_csv(res_loc+"calibration/posterior_samples.csv",index=False)
