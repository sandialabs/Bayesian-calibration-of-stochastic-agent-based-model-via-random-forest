import numpy as np
jnp = np

def finite_diff_params(params, hs):
    n_params = len(params)
    all_params = np.zeros((n_params, n_params))
    for i in range(n_params):
        temp_params = np.copy(params)
        temp_params[i] += hs[i]
        all_params[i] = temp_params
    return all_params

def calculate_finite_diff(params, all_runs, hs):
    prediction = all_runs[params.tobytes()]
    model_grad = jnp.zeros((len(params), *prediction.shape))
    for i in range(len(params)):
        temp_params = np.copy(params)
        temp_params[i] += hs[i]
        temp_pred = all_runs[temp_params.tobytes()]
        # Finite difference in direction of parameter i:
        model_grad[i] = (temp_pred - prediction) / hs[i]
    return model_grad

