from finite_diff import calculate_finite_diff, finite_diff_params
from posteriors import grad_log_posteriorG, grad_log_posteriorU
from kernels import svgd_kernel
import numpy as np
jnp = np

def potential_grad_modelG(particles, data=None, run_params_func=None, hs=None, s2=None, ms=None, ss=None):
    """
    compute Stein potential
    """
    nparticles = particles.shape[0]

    # Calculate all needed runs
    all_params = np.vstack([finite_diff_params(prtcl, hs) for prtcl in particles])
    all_params = np.vstack([particles, all_params])

    # Consolidate to avoid duplicates
    all_params = np.unique(all_params, axis=0)

    # Run list of params
    all_runs = run_params_func(all_params)

    # grad log posterior
    grad_beta = []
    for prtcl in particles:
        # Access correct params for this case
        prediction = all_runs[prtcl.tobytes()]
        grad_prediction = calculate_finite_diff(prtcl, all_runs, hs)
        grad_ll = grad_log_posteriorG(
            prtcl,
            data=data,
            prediction=prediction,
            grad_prediction=grad_prediction,
            s2=s2, ms=ms, ss=ss
        )
        grad_beta.append(grad_ll)
    grad_beta = jnp.array(grad_beta)

    # calculate the kernel matrix contributions
    if nparticles > 1:
        Kmat, dKmat = svgd_kernel(particles)
    else:
        Kmat = np.ones((1,1))
        dKmat = np.zeros((1,particles.shape[1]))

    # assemble potential and return
    grad_beta = (jnp.matmul(Kmat, grad_beta) + dKmat) / nparticles
    grad_beta = grad_beta / nparticles
    return grad_beta, all_runs

def potential_grad_modelU(particles, data=None, run_params_func=None, hs=None, s2=None, avals=None, bvals=None):
    """
    compute Stein potential
    """
    nparticles = particles.shape[0]

    # Calculate all needed runs
    all_params = np.vstack([finite_diff_params(prtcl, hs) for prtcl in particles])
    all_params = np.vstack([particles, all_params])

    # Consolidate to avoid duplicates
    all_params = np.unique(all_params, axis=0)

    # Run list of params
    all_runs = run_params_func(all_params)

    # grad log posterior
    grad_beta = []
    for prtcl in particles:
        # Access correct params for this case
        prediction = all_runs[prtcl.tobytes()]
        grad_prediction = calculate_finite_diff(prtcl, all_runs, hs)
        grad_ll = grad_log_posteriorU(
            prtcl,
            data=data,
            prediction=prediction,
            grad_prediction=grad_prediction,
            s2=s2, avals=avals, bvals=bvals
        )
        grad_beta.append(grad_ll)
    grad_beta = jnp.array(grad_beta)

    # calculate the kernel matrix contributions
    if nparticles > 1:
        Kmat, dKmat = svgd_kernel(particles)
    else:
        Kmat = np.ones((1,1))
        dKmat = np.zeros((1,particles.shape[1]))

    # assemble potential and return
    grad_beta = (jnp.matmul(Kmat, grad_beta) + dKmat) / nparticles
    # grad_beta = grad_beta / nparticles
    return grad_beta, all_runs

def potential_grad_modelU2(particles, data=None, run_params_func=None, hs=None, s2=None, avals=None, bvals=None):
    """
    compute Stein potential
    """
    nparticles = particles.shape[0]

    # Calculate all needed runs
    all_params = np.vstack([finite_diff_params(prtcl, hs) for prtcl in particles])
    all_params = np.vstack([particles, all_params])

    # Consolidate to avoid duplicates
    all_params = np.unique(all_params, axis=0)

    # Run list of params
    all_runs = run_params_func(all_params)

    # grad log posterior
    grad_beta = []
    for prtcl in particles:
        # Access correct params for this case
        prediction = all_runs[prtcl.tobytes()]
        grad_prediction = calculate_finite_diff(prtcl, all_runs, hs)
        grad_ll = grad_log_posteriorU(
            prtcl,
            data=data,
            prediction=prediction,
            grad_prediction=grad_prediction,
            s2=s2, avals=avals, bvals=bvals
        )
        grad_beta.append(grad_ll)
    grad_beta = jnp.array(grad_beta)

    # calculate the kernel matrix contributions
    if nparticles > 1:
        Kmat, dKmat = svgd_kernel(particles)
    else:
        Kmat = np.ones((1,1))
        dKmat = np.zeros((1,particles.shape[1]))

    # assemble potential and return
    grad_beta = (jnp.tensordot(Kmat, grad_beta, axes=1) + dKmat[:,:,jnp.newaxis]) / nparticles
    grad_beta = np.mean(grad_beta, axis=2)
    # grad_beta = grad_beta / nparticles
    return grad_beta, all_runs
