import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate
from scipy.stats import sobol_indices, uniform, norm


######################## DATA PREP ##############################
import os
env = os.environ["PROJLOC"]
import sys
sys.path.append(env + "/src/utils")
sys.path.append(env + "/src/surrogate")
import utils as pru
from load_data import get_data, get_params, get_real_data

# Load data
data_raw = get_data(abc=False)
data_params = get_params(abc=False, drop_vars=True)

# Z scale
transformer = StandardScaler().fit(data_params)
params = data_params.copy()
params.iloc[:,:] = transformer.transform(params)

# Train, test split
data_train_input, data_test_input, data_train_output, data_test_output = train_test_split(params, data_raw, random_state=0)

resdir = env+"/results/stein_calibration_gaussian_process/"

######################## TRAIN MODEL ##############################
load_name = resdir+"surrogate/best_estimator.pkl"
with open(load_name, "rb") as file:
    gp = pickle.load(file)

# Gaussian process
gp.fit(data_train_input, data_train_output)
gp_permute = permutation_importance(gp, data_test_input, data_test_output, random_state=0).importances_mean
######################## PREDICTION ##############################
prediction = gp.predict(data_test_input).astype(int)

# Compare predictions with ABM data
truth = data_test_output.to_numpy().astype(int)
rel_err = pru.absolute_relative_err(truth,prediction)

# Errors
err_five,err_median,err_nfive = pru.err_statistics(rel_err)

print("=====================================================")
print("Parameters:")
print("------------------------")
print(gp.get_params())
print("=====================================================")
print("Median relative absolute error: {}".format(err_median))
print("5% quantile relative absolute error: {}".format(err_five))
print("95% quantile relative absolute error: {}".format(err_nfive))
print("=====================================================")
print("Feature importance (permutation):")
print("------------------")
permute_imp = pd.Series(gp_permute, index=data_train_input.columns).sort_values()[::-1]
print(permute_imp)
def run_sobol_surrogate(X):
    preds = gp.predict(X.T)
    return preds.T

si = sobol_indices(
    func=run_sobol_surrogate,
    n=2**12,
    dists = [uniform(loc=mi,scale=ma-mi) for mi,ma in zip(params.min(), params.max())]
)

si_first = si.first_order.mean(axis=0)
si_total = si.total_order.mean(axis=0)
print("------------------")
print("Feature importance (sobol first):")
print("------------------")
first_imp = pd.Series(si_first, index=data_train_input.columns).sort_values()[::-1]
print(first_imp)
print("------------------")
print("Feature importance (sobol total):")
print("------------------")
total_imp = pd.Series(si_total, index=data_train_input.columns).sort_values()[::-1]
print(total_imp)

# %%
pname = resdir+"surrogate/permutation_importance.csv"
fname = resdir+"surrogate/sobol_first_importance.csv"
tname = resdir+"surrogate/sobol_total_importance.csv"
permute_imp.to_csv(pname)
first_imp.to_csv(fname)
total_imp.to_csv(tname)

######################## CROSS VALIDATION ##############################

def cv_scorer(estimator, X, y, axis=None):
    truth = y
    prediction = estimator.predict(X)
    rel_err = pru.absolute_relative_err(truth,prediction)
    err_five,err_median,err_nfive = pru.err_statistics(rel_err,axis=axis)
    score = {"5%":err_five, "median":err_median, "95%":err_nfive}
    return score

cv_folds = 7
cv_results = cross_validate(gp, params, data_raw, cv=cv_folds, scoring=cv_scorer, return_estimator=True, return_indices=True)
print("=====================================================")
print(f"Cross validation with {cv_folds} folds")
print("------------------")
print("5% quantile relative errors of data reconstruction:")
print(cv_results['test_5%'])
print("------------------")
print("median relative errors of data reconstruction:")
print(cv_results['test_median'])
print("------------------")
print("95% quantile relative errors of data reconstruction:")
print(cv_results['test_95%'])
print("------------------")
col_names = ["5% quantile", "Median", "95% quantile"]
cv_df = pd.DataFrame(
    [[tf,tm,tnf] for tf,tm,tnf in zip(cv_results['test_5%'],cv_results['test_median'],cv_results['test_95%'])],
    columns=col_names
)
ax = cv_df.plot(
    kind="box",
    logy=True,
    grid=True,
    figsize=(5,7),
    title=f"Cross validation of GP surrogate ({cv_folds} folds)",
    ylabel="($\\frac{|y - \\hat{y}|}{|y|}$)"
)
fig = ax.get_figure()
fig.tight_layout()

save_name = env+"/plots/stein_calibration_gaussian_process/surrogate/cross_validation.pdf"
fig.savefig(save_name)

fig.show()

# %%
# Save model final
gp.fit(params, data_raw)
save_dict = {"gp": gp, "transformer": transformer}

model_name = resdir+"surrogate/surrogate.pkl"
with open(model_name, "wb") as file:
    pickle.dump(save_dict, file)
