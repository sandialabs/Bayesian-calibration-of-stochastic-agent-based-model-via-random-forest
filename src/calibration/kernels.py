from scipy.spatial.distance import pdist, squareform
import numpy as jnp
np = jnp

def svgd_kernel(beta,h=-1):
    """
    Calculate kernel matrix and its gradient: K, \nabla_x k
    """

    # compute kernel matrix using rbf
    sq_dist = pdist(beta)
    pairwise_dists = squareform(sq_dist)**2
    if h < 0:
        # if h < 0, using median to evaluate bandwidth from data
        h = jnp.median(pairwise_dists)
        h = jnp.sqrt(0.5 * h / jnp.log(beta.shape[0]+1))

    Kmat  = jnp.exp( -pairwise_dists / h**2 / 2)
    dKfull = Kmat[:,:,None] * (beta[:,None,:] - beta[None,:,:]) / h**2
    dKmat = dKfull.sum(axis=1)
    return (Kmat, dKmat)
