# %%
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
from load_data import get_data, get_params

# Load data
data_raw = get_data(abc=False)
data_params = get_params(abc=False, drop_vars=True)

# Z scale
transformer = StandardScaler().fit(data_raw)
data_processed = data_raw.copy()
data_processed.iloc[:,:] = transformer.transform(data_raw)

# PCA of data
data_components, data_coefs, _ = pru.get_pca_data(data_processed,pca_var_explained=.95)

# Train, test split
data_train_input, data_test_input, data_train_output, data_test_output = train_test_split(data_params,data_coefs,random_state=0)
_, _, _, test_samples = train_test_split(data_processed,data_processed,random_state=0)

resdir = env+"/results/bayesian_calibration_random_forest/"

######################## TRAIN MODEL ##############################
load_name = resdir+"surrogate/best_estimator.pkl"
with open(load_name, "rb") as file:
    rf = pickle.load(file)

# Random forest
rf.fit(data_train_input, data_train_output)
rf_gini = rf.feature_importances_
rf_permute = permutation_importance(rf, data_test_input, data_test_output, random_state=0).importances_mean
######################## PREDICTION ##############################
# Predict PCA coefficients for test data
pred_coefs = rf.predict(data_test_input)
preds = test_samples.copy()
preds.iloc[:,:] = (data_components @ pred_coefs.T).T

# Compare predictions with ABM data
truth = pru.inverse_transform_data(test_samples,transformer)
prediction = pru.inverse_transform_data(preds,transformer)
rel_err = pru.absolute_relative_err(truth,prediction)

# Errors
err_five,err_median,err_nfive = pru.err_statistics(rel_err)

print("=====================================================")
print("PCA: {} components".format(data_components.shape[1]))
print("=====================================================")
print("Parameters:")
print("------------------------")
print(rf.get_params())
print("=====================================================")
print("Median relative absolute error: {}".format(err_median))
print("5% quantile relative absolute error: {}".format(err_five))
print("95% quantile relative absolute error: {}".format(err_nfive))
print("=====================================================")
print("Feature importance (impurity):")
print("------------------")
gini_imp = pd.Series(rf_gini, index=data_train_input.columns).sort_values()[::-1]
print(gini_imp)
print("------------------")
print("Feature importance (permutation):")
print("------------------")
permute_imp = pd.Series(rf_permute, index=data_train_input.columns).sort_values()[::-1]
print(permute_imp)
def run_sobol_surrogate(X):
    pred_coefs = rf.predict(X.T)
    preds = (data_components @ pred_coefs.T).T
    prediction = pru.inverse_transform_data(preds,transformer)
    return prediction.T

si = sobol_indices(
    func=run_sobol_surrogate,
    n=2**12,
    dists = [norm(loc=m,scale=s) for m,s in zip(data_params.mean(), data_params.std())]
    # dists = [uniform(loc=mi,scale=ma-mi) for mi,ma in zip(data_params.min(), data_params.max())]
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
gname = resdir+"surrogate/gini_importance.csv"
pname = resdir+"surrogate/permutation_importance.csv"
fname = resdir+"surrogate/sobol_first_importance.csv"
tname = resdir+"surrogate/sobol_total_importance.csv"
gini_imp.to_csv(gname)
permute_imp.to_csv(pname)
first_imp.to_csv(fname)
total_imp.to_csv(tname)

######################## CROSS VALIDATION ##############################

def cv_scorer(estimator, X, y, axis=None):
    true_traj = pru.reconstruct_data_from_pca(y, data_components)
    pred_traj = pru.reconstruct_data_from_pca(estimator.predict(X), data_components)
    truth = pru.inverse_transform_data(true_traj,transformer)
    prediction = pru.inverse_transform_data(pred_traj,transformer)
    rel_err = pru.absolute_relative_err(truth,prediction)
    err_five,err_median,err_nfive = pru.err_statistics(rel_err,axis=axis)
    score = {"5%":err_five, "median":err_median, "95%":err_nfive}
    return score

cv_folds = 7
cv_results = cross_validate(rf, data_params, data_coefs, cv=cv_folds, scoring=cv_scorer, return_estimator=True, return_indices=True)
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
    title=f"Cross validation of PCA+RF surrogate ({cv_folds} folds)",
    ylabel="($\\frac{|y - \\hat{y}|}{|y|}$)"
)
fig = ax.get_figure()
fig.tight_layout()

save_name = env+"/plots/bayesian_calibration_random_forest/surrogate/cross_validation.pdf"
fig.savefig(save_name)

fig.show()

# %%
######################## SAVE FAILED PARAMETERS ##############################

parameter_errors = pd.DataFrame(
    np.ones(data_params.shape[0]),
    index=data_params.index,
    columns=["Median abs rel err (over time)"]
)

for test_inds,estimator in zip(cv_results['indices']['test'],cv_results['estimator']):
    test_inputs = data_params.iloc[test_inds,:]
    test_outputs = data_coefs[test_inds,:]
    scores = cv_scorer(estimator, test_inputs, test_outputs, axis=1)['median']
    parameter_errors.iloc[test_inds,:] = scores.reshape((-1,1))

# Plot failed parameters
parameter_errors.reset_index().plot(
    kind="scatter",
    x="instance",
    y="Median abs rel err (over time)",
    ylabel="median ($\\frac{|y - \\hat{y}|}{|y|}$) over time",
    xlabel="Parameter set"
)

save_name = resdir+"surrogate/parameter_errors.csv"
parameter_errors.to_csv(save_name)

# %%
# Save model final
rf.fit(data_params, data_coefs)
save_dict = {"pca_components": data_components, "rf": rf, "transformer": transformer}

model_name = resdir+"surrogate/surrogate.pkl"
with open(model_name, "wb") as file:
    pickle.dump(save_dict, file)
