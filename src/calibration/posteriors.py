import numpy as jnp
np = jnp

def log_posteriorG(params, data, prediction, s2, ms, ss):
    """
        Derivation:
        -----------
        log of MVNormal likelihood and normal priors:
            -(data - pred) * Sigma * (data - pred) - sum_j [(p_j-m_j)/s_j]^2
        assume diagonal covariance Sigma for likelihood:
            -sum_i( (data_i - pred_i)^2 / s2) - sum_j [(p_j-m_j)/s_j]^2
    """
    likelihood = np.apply_along_axis(np.divide, 0, -(prediction - data)**2, s2)
    priors = [-(p-m)**2 / s**2 for p,m,s in zip(params,ms,ss)]
    return jnp.sum(likelihood) + jnp.sum(priors)
    # return jnp.sum(likelihood)

def log_posteriorU(params, data, prediction, s2, avals, bvals):
    """
        Derivation:
        -----------
        log of MVNormal likelihood and normal priors:
            -(data - pred) * Sigma * (data - pred) - sum_j {0 if inside or -inf outside}
        assume diagonal covariance Sigma for likelihood:
            -sum_i( (data_i - pred_i)^2 / s2) - sum_j {0 if inside or -inf outside}
    """
    likelihood = np.apply_along_axis(np.divide, 0, -(prediction - data)**2, s2)
    priors = [0 if (a <= p <= b) else -np.inf for p,a,b in zip(params,avals,bvals)]
    return jnp.sum(likelihood) + jnp.sum(priors)
    # return jnp.sum(likelihood)

def grad_log_posteriorG(params, data, prediction, grad_prediction, s2, ms, ss):
    """
        Derivation:
        -----------
        grad:
            d/dp[ -sum_i( (data_i - pred_i)^2 / s2) - sum_j [(p_j-m_j)/s_j]^2 ]
        result:
            -2*sum_i( (data_i - pred_i) / s2) * d/dp(pred_i) - 2*[ (p_j-m_j)/s_j^2]
    """
    grad_log_likelihood = -2*jnp.sum((prediction - data) * np.apply_along_axis(np.divide, 1, grad_prediction, s2), axis=1)
    grad_log_priors = jnp.array([-2*(p-m) / s**2 for p,m,s in zip(params.flatten(),ms.flatten(),ss.flatten())])
    return grad_log_likelihood + grad_log_priors
    # return grad_log_likelihood

def grad_log_posteriorU(params, data, prediction, grad_prediction, s2, avals, bvals, strength=1e10):
    """
        Derivation:
        -----------
        grad:
            d/dp[ -sum_i( (data_i - pred_i)^2 / s2) - sum_j {0 if inside or -inf outside} ]
        result:
            -2*sum_i( (data_i - pred_i) / s2) * d/dp(pred_i) - {inf if on lower bound or -inf if on upper}
    """
    grad_log_likelihood = -2*jnp.sum((prediction - data) * np.apply_along_axis(np.divide, 1, grad_prediction, s2), axis=1)
    grad_log_priors = []
    for p,a,b in zip(params.flatten(),avals.flatten(),bvals.flatten()):
        scale = b - a
        if p - a < 0:
            grad_log_priors.append(scale*strength)
        elif p - b > 0:
            grad_log_priors.append(-scale*strength)
        else:
            grad_log_priors.append(0)
    grad_log_priors = np.array(grad_log_priors)
    if np.ndim(grad_log_likelihood) == 2:
        grad_log_priors = grad_log_priors.reshape((-1, 1))
    return grad_log_likelihood + grad_log_priors
    # return grad_log_likelihood
