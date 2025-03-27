import pickle
import numpy as np
import os
env = os.environ["PROJLOC"]

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import PairwiseKernel, Matern, RBF

import sys
sys.path.append(env + "/src/utils")
sys.path.append(env + "/src/surrogate")
import utils as pru
from load_data import get_data, get_params, get_real_data

# Load data
data_raw = get_data(abc=False)
data_params = get_params(abc=False, drop_vars=True)

variables = data_raw.columns.get_level_values(0).unique()

# Z-scale data
transformer = StandardScaler().fit(data_params)
params = data_params.copy()
params.iloc[:,:] = transformer.transform(params)

data_train_input, data_test_input, data_train_output, data_test_output = train_test_split(params, data_raw, random_state=0)

# %%
## Hyperparameter tuning
gp = GaussianProcessRegressor(normalize_y=True)
parameters = {
    "alpha": np.linspace(1e-3,1,20),
    "kernel": [
        PairwiseKernel(metric="polynomial"), # Mean RE: 0.028, KMean RE: 0.046
        PairwiseKernel(metric="laplacian"),  # Mean RE: 0.028, KMean RE: 0.059
        RBF(),
        Matern(),
    ]
}
grid = GridSearchCV(gp, parameters, verbose=2, cv=7, n_jobs=-1)
grid.fit(data_train_input, data_train_output)

# %%
save_name = env+"/results/stein_calibration_gaussian_process/surrogate/best_estimator.pkl"
with open(save_name, "wb") as file:
    pickle.dump(grid.best_estimator_, file)

print("Best estimator: {}".format(grid.best_estimator_))
print("Best score: {}".format(grid.best_score_))

print("Best test score: {}".format(grid.score(data_test_input, data_test_output)))
true_preds = grid.predict(data_test_input)

true_data = data_test_output
rel_err = pru.absolute_relative_err(true_data, true_preds)
_,rel_med,_ = pru.err_statistics(rel_err['hospitalizations'])
print("Median rel abs err (hospitalizations): \n{}".format(rel_med))
_,rel_med,_ = pru.err_statistics(rel_err['deaths'])
print("Median rel abs err (deaths): \n{}".format(rel_med))
