import numpy as np
import pandas as pd
import pickle
from functools import partial
import warnings
warnings.filterwarnings(action="ignore",category=UserWarning)
warnings.filterwarnings(action="ignore",category=RuntimeWarning)

import os
env = os.environ["PROJLOC"]

import sys
sys.path.append(env+"/src/utils")
from load_data import get_data, get_params
sys.path.append(env+"/src/calibration")
from optimisers import adam
from potentials import potential_grad_modelU2
from posteriors import log_posteriorU

seeds = 6
nparticles = 200
smoothing = 7
prediction_samples = 500
nsimu = 10000
lr = 1e-3
verbosity = 100

# ------------------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------------------
data_real = get_real_data()
data_real = data_real.diff()
train_data = get_data(abc=False)
parameters = get_params(abc=False, drop_vars=True)
param_min = parameters.min().to_numpy()
scaled_parameters = parameters - param_min
scaled_max = scaled_parameters.max().to_numpy()
parameters = scaled_parameters / scaled_max
hosp_len = data_real.shape[0]
comparison_data = data_real.reindex(columns=["hospitalizations","deaths"]).T.to_numpy()

comparison_data = comparison_data[:,1:]
comparison_data[0] = np.convolve(comparison_data[0], np.ones(smoothing)/smoothing, 'same')
comparison_data[1] = np.convolve(comparison_data[1], np.ones(smoothing)/smoothing, 'same')
res_loc = env+"/results/stein_calibration_gaussian_process/calibration/"
sur_loc = env+"/results/stein_calibration_gaussian_process/surrogate/"

# Load model
surr_loc = sur_loc + "surrogate.pkl"
with open(surr_loc, "rb") as file:
    loaded_model = pickle.load(file)
smodel = loaded_model["gp"]
transformer = loaded_model["transformer"]

# Restrict data to chicago time frame
full_time = pd.to_datetime(train_data.columns.get_level_values(1))
times = np.arange(full_time.unique().shape[0])
data_time = pd.to_datetime(data_real.index)
time_mask = np.array([True if (ft in data_time) else False for ft in full_time])

# Define surrogate model function
def surrogate_func(ps):
    params = ps * scaled_max + param_min
    tps = transformer.transform(params)
    raw_result = smodel.predict(tps)
    result = raw_result[:, time_mask]
    result = np.stack((result[:,:hosp_len], result[:,hosp_len:]), axis=0)
    result = np.moveaxis(result, 1, 0)
    result = np.diff(result)
    return result.squeeze()

def ssfun(ps,data):
    res = np.array(data["ydata"]).squeeze() - surrogate_func(ps)
    return (res**2).sum(axis=1)

# %%
# ------------------------- STEIN VI --------------------------------
x = np.arange(comparison_data.shape[1]).flatten()
y = comparison_data
cdata = {"xdata": x, "ydata": y}

# Find MLE from ABM runs
errs = np.linalg.norm(train_data.diff(axis=1).loc[:,time_mask].iloc[:,2:] - comparison_data.flatten(), axis=1)
ps0_loc = np.argsort(errs)[0]
ps0 = parameters.iloc[ps0_loc].to_numpy()

for seed in range(seeds):
    filename = f"stein_calibration_seed{seed}"

    S2_ols = ssfun(ps0.reshape((1,-1)),cdata) / (comparison_data.shape[1] - parameters.shape[1])

    hs = 1e-6*np.ones(4)
    avals = np.zeros(4)
    bvals = np.ones(4)
    def run_params_func(params):
        result = surrogate_func(params)
        dict_result = dict()
        for i in range(params.shape[0]):
            dict_result[params[i].tobytes()] = result[i]
        return dict_result

    pgm = potential_grad_modelU2

    potential_grad = partial(pgm,data=comparison_data,run_params_func=run_params_func,hs=hs,s2=S2_ols,avals=avals,bvals=bvals)
    lp = partial(log_posteriorU,data=comparison_data,s2=S2_ols,avals=avals,bvals=bvals)

    # Initialize parameters from prior
    rng = np.random.default_rng(seed)
    particles = rng.uniform(avals,bvals,(nparticles,parameters.shape[1]))

    # Use ADAM optimizer to advance particles with gradient ascent
    particles, particles_over_time, logposteriors = adam(lp, potential_grad, particles, lr, max_iter=nsimu, verb=verbosity)

    # %%
    # Rescale
    scaled_particles = particles * scaled_max + param_min
    scaled_particles_over_time = [p * scaled_max + param_min for p in particles_over_time]
    scaled_avals = avals * scaled_max + param_min
    scaled_bvals = bvals * scaled_max + param_min
    scaled_mle_params = ps0.reshape((1,-1)) * scaled_max + param_min
    curves = surrogate_func(particles)

    # %%
    # Save results
    np.save(res_loc+filename+".npy", scaled_particles)
    particle_df = pd.DataFrame(scaled_particles, columns=parameters.columns)
    particle_df.to_csv(res_loc+filename+".csv", index=False)

    curve_ticks = pd.to_datetime(data_real.dropna().index).strftime("%m/%d")
    curve_df = pd.concat([
         pd.DataFrame(curves[:,0].T, index=curve_ticks),
         pd.DataFrame(curves[:,1].T, index=curve_ticks),
    ])
    curve_df["type"] = "hospitalizations"
    curve_df.iloc[len(curve_ticks):,-1] = "deaths"
    np.save(res_loc+filename+"_curves.npy", np.array(list(curves)))
    curve_df.to_csv(res_loc+filename+"_curves.csv")

    np.save(res_loc+filename+"_logposteriors.npy", logposteriors)
    np.save(res_loc+filename+"_avals.npy", scaled_avals)
    np.save(res_loc+filename+"_bvals.npy", scaled_bvals)
    np.save(res_loc+filename+"_mle_params.npy", scaled_mle_params)
    np.save(res_loc+filename+"_sparticles_over_time.npy", scaled_particles_over_time)
    np.save(res_loc+filename+"_particles_over_time.npy", particles_over_time)
