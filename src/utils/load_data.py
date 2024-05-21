import numpy as np
import pandas as pd
import os
env = os.environ["PROJLOC"]

def get_params(abc=False,drop_vars=True):
    if abc:
        params_loc = env+"/data/abc_parameters.csv"
    else:
        params_loc = env+"/data/citycovid_parameters.csv"
    params = pd.read_csv(params_loc, index_col=0)
    if drop_vars:
        parameter_mask = np.isclose(params.var(), 0) == False
        params = params.iloc[:,parameter_mask]
    return params

def get_data(abc=False):
    if abc:
        data_loc = env+"/data/abc_results.csv"
    else:
        data_loc = env+"/data/citycovid_results.csv"
    data = pd.read_csv(data_loc, index_col=0, header=[0,1])
    return data

