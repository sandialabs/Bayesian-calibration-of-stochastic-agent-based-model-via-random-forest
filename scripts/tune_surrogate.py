import pickle
import os
env = os.environ["PROJLOC"]

from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

import sys
sys.path.append(env + "/src/utils")
sys.path.append(env + "/src/surrogate")
import utils as pru
from load_data import get_data, get_params

# Load data
data_raw = get_data(abc=False)
data_params = get_params(abc=False, drop_vars=True)

variables = data_raw.columns.get_level_values(0).unique()

# Z-scale data
transformer = StandardScaler().fit(data_raw)
data_processed = data_raw.copy()
data_processed.iloc[:,:] = transformer.transform(data_raw)

data_components, data_ws, _ = pru.get_pca_data(data_processed)

data_train_input, data_test_input, data_train_output, data_test_output = train_test_split(data_params,data_ws,random_state=0)
train_trajectories, _, _, test_trajectories = train_test_split(data_processed,data_processed,random_state=0)

# %%
## Hyperparameter tuning

def rf_accuracy(y_true, y_pred):
  scaled_y_true = pru.inverse_transform_data(y_true,transformer)
  scaled_y_pred = pru.inverse_transform_data(y_pred,transformer)
  scaled_err = pru.absolute_relative_err(scaled_y_true,scaled_y_pred)
  _,scaled_median,_ = pru.err_statistics(scaled_err)
  return scaled_median

rf1 = RandomForestRegressor()
parameters = {
    'n_estimators': [10, 50, 100, 200, 300, 400, 500], # default: 100
    'criterion': ['squared_error','absolute_error'], # default: 'squared_error'
    'max_depth': [None, 1, 5, 10], # default: None
    'min_samples_leaf': [1, 3, 5, 8, 10], # default: 1
    'max_features': [1, 3, 5, 8, 10], # default: 1
    'oob_score': [False, True] # default: False
}
grid = GridSearchCV(rf1, parameters, cv=5, n_jobs=5)
grid.fit(data_train_input, data_train_output)

# %%
save_name = env+"/results/surrogate/best_estimator.pkl"
with open(save_name, "wb") as file:
    pickle.dump(grid.best_estimator_, file)

print("Best estimator: {}".format(grid.best_estimator_))
print("Best score: {}".format(grid.best_score_))

print("Best test score: {}".format(grid.score(data_test_input, data_test_output)))
pred_ws = grid.predict(data_test_input)
preds = pru.reconstruct_data_from_pca(pred_ws, data_components)

true_data = pru.inverse_transform_data(test_trajectories,transformer)
true_preds = pru.inverse_transform_data(preds,transformer)
rel_err = pru.absolute_relative_err(true_data, true_preds)
_,rel_med,_ = pru.err_statistics(rel_err['hospitalizations'])
print("Median rel abs err: \n{}".format(rel_med))
_,rel_med,_ = pru.err_statistics(rel_err['deaths'])
print("Median rel abs err: \n{}".format(rel_med))

