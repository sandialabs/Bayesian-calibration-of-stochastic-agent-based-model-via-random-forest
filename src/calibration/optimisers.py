import numpy as np
jnp = np

def adam(lposterior, potential_grad, particles, stepsize, max_iter=100, eps=1.e-6, beta1=0.8, beta2=0.99, verb=0, keep=True, print_progress=True):
    """
    adam optmizer
    """
    nparticles = particles.shape[0]

    m = 0.0
    v = 0.0
    keepers = []
    logposteriors = []
    for iter in range(max_iter):
        grad_beta,all_runs = potential_grad(particles)
        if verb>0:
            if iter%verb == 0:
                ave_log_posterior = np.sum([lposterior(p, prediction=all_runs[p.tobytes()]) for p in particles]) / nparticles
                ave_particle = np.mean(particles,axis=0)
                if print_progress:
                    print(iter,ave_log_posterior,ave_particle)
                if keep:
                    keepers.append(particles)
                    logposteriors.append(ave_log_posterior)
        # Step
        m = beta1 * m + (1 - beta1) * grad_beta
        v = beta2 * v + (1 - beta2) * grad_beta**2
        mhat = m / (1 - beta1**(iter+1))
        vhat = v / (1 - beta2**(iter+1))
        particles = particles + stepsize * mhat / (jnp.sqrt(vhat) + eps)

    if keep:
        return particles, keepers, logposteriors
    else:
        return particles

def adamR(lposterior, potential_grad, particles, stepsize, restarts=2, rfunc=lambda n,lr: lr, max_iter=100, eps=1.e-6, beta1=0.8, beta2=0.99, verb=0, keep=True, print_progress=True):
    """
    adam optmizer (with restarts)
    """
    nparticles = particles.shape[0]

    keepers = []
    logposteriors = []
    for res in range(restarts):
        m = 0.0
        v = 0.0
        for iter in range(max_iter):
            grad_beta,all_runs = potential_grad(particles)
            if verb>0:
                if iter%verb == 0:
                    ave_log_posterior = np.sum([lposterior(p, prediction=all_runs[p.tobytes()]) for p in particles]) / nparticles
                    ave_particle = np.mean(particles,axis=0)
                    if print_progress:
                        print(iter,ave_log_posterior,ave_particle)
                    if keep:
                        keepers.append(particles)
                        logposteriors.append(ave_log_posterior)
            # Step
            m = beta1 * m + (1 - beta1) * grad_beta
            v = beta2 * v + (1 - beta2) * grad_beta**2
            mhat = m / (1 - beta1**(iter+1))
            vhat = v / (1 - beta2**(iter+1))
            particles = particles + stepsize * mhat / (jnp.sqrt(vhat) + eps)
        stepsize = rfunc((res+1)*max_iter, stepsize)

    if keep:
        return particles, keepers, logposteriors
    else:
        return particles

def adagrad(lposterior, potential_grad, particles, stepsize, max_iter=100, verb=0, keep=True):
    """
    adagrad optmizer
    """
    nparticles = particles.shape[0]

    sum_sq_grad = 0.0
    keepers = []
    logposteriors = []
    for iter in range(max_iter):
        grad_beta,all_runs = potential_grad(particles)
        if verb>0:
            if iter%verb == 0:
                ave_log_posterior = np.sum([lposterior(p, prediction=all_runs[p.tobytes()]) for p in particles]) / nparticles
                ave_particle = np.mean(particles,axis=0)
                print(iter,ave_log_posterior,ave_particle)
                if keep:
                    keepers.append(particles)
                    logposteriors.append(ave_log_posterior)
        # Step
        sum_sq_grad += grad_beta**2
        particles = particles + stepsize * grad_beta / np.sqrt(sum_sq_grad/(iter+1))

    if keep:
        return particles, keepers, logposteriors
    else:
        return particles
