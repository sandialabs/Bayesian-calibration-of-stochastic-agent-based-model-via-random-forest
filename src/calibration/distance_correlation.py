import numpy as np
from scipy.spatial.distance import cdist

NWARN=10000 # if using more than this (approximately), things might become very costly


def _compute_centered_matrix(spl):
    """
    centers a matrix
    """
    num_samples = spl.shape[0]
    if len(spl.shape) == 1:
        Amat = np.abs(np.subtract.outer(spl,spl))
    elif len(spl.shape) == 2:
        Amat = cdist(spl, spl, metric='euclidean')
    else:
        print('_compute_centered_matrix works with order 1 and 2 tensors')
        quit()

    # compute means
    Alin = np.mean(Amat, axis=1)
    Acol = np.mean(Amat, axis=0)
    Amn = np.mean(Amat)
    # subtract/add means (linewise, columnwise, overall)
    Amat = Amat - Alin.reshape(num_samples,1)
    Amat = (Amat.T - Acol.reshape(num_samples,1)).T
    Amat = Amat+Amn
    return Amat

def distcorr(spl,rv_sizes=None):
    """
    Compute distance correlation between random vectors, 
    possibly of different dimensionalities

    Args:
        spl - 2D array of samples, each row is a sample
        rv_sizes - array/list of rv dimensions (assumed all equal to 1 if not provided)

    Output:
       Returns a 2D array of distance correlations between pairs of random vectors;
       only entries 0<=j<i<no. of random vectors are populated

    References:
       * http://en.wikipedia.org/wiki/Distance_correlation
       * Szekely & Rizzo, "Brownian Distance Covariance", The Annals of Applied Statistics
         2009, Vol. 3, No. 4, 1236–1265, DOI: 10.1214/09-AOAS312
    """
    num_samples = spl.shape[0]
    if (num_samples > NWARN):
        print('Warning ! This might be a lengthy calculation: num_samples='+str(num_samples))

    if rv_sizes is not None:
        assert sum(rv_sizes)==spl.shape[1]
        num_rvs = len(rv_sizes)
        rv_idx = np.insert(np.cumsum(rv_sizes),0,0)
    else:
        num_rvs = spl.shape[1]
        rv_idx = np.arange(num_rvs+1)

    dCor = np.zeros((num_rvs, num_rvs))
    As = [_compute_centered_matrix(spl[:,rv_idx[0]:rv_idx[1]])]
    dSigX = [np.sqrt(np.sum(As[0]**2))/num_samples]
    for i in range(1,num_rvs):

        if rv_idx[i+1]-rv_idx[i]>1:
            As.append(_compute_centered_matrix(spl[:,rv_idx[i]:rv_idx[i+1]]))
        else:
            As.append(_compute_centered_matrix(spl[:,rv_idx[i]]))
        dSigX.append(np.sqrt(np.sum(As[-1]**2))/num_samples)

        if i%10 == 0:
            print('{},'.format(i),end='',flush=True)

        for j in range(i):
            dCov = np.sum(np.multiply(As[i], As[j]))/(num_samples * num_samples)
            dCor[i,j] = dCov/(dSigX[i] * dSigX[j])

    if num_rvs>9:
        print('\n')

    As.clear()
    return np.sqrt(dCor)

if __name__ == "__main__":

    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2,2,figsize=(6,6))

    num_samples = 1000

    def plot_xy(x, y, dC, ax):
        ax.plot(x, y, '.', ms=1)
        ax.set_aspect('equal')
        ax.set_title('{:.2f}'.format(dC[1,0]))

    def yx2(nspl, ax=None, noise=0.01):
        x = np.random.uniform(low=-1, high=1, size=nspl)
        y = x**2+np.random.uniform(low=-noise, high=noise, size=nspl)
        dCor = distcorr(np.vstack((x,y)).T)
        if ax is not None:
            plot_xy(x, y, dCor, ax)
        return x, y, dCor
    
    def xycirc(nspl, loc=1, ax=None, noise=0.01):
        theta = np.random.uniform(low=0, high=2*np.pi, size=nspl)
        radius = np.random.normal(loc=loc, scale=noise, size=nspl)
        x = radius*np.cos(theta)
        y = radius*np.sin(theta)
        dCor = distcorr(np.vstack((x,y)).T)
        if ax is not None:
            plot_xy(x, y, dCor, ax)
        return x, y, dCor

    # test 1: x\in [-1,1], y=x^2+U[-0.5,0.5]
    x, y, dCor = yx2(num_samples, noise=0.5, ax=axs[0,0])

    # test 2: x\in [-1,1], y=x^2+U[-0.05,0.05]
    x, y, dCor = yx2(num_samples, noise=0.05, ax=axs[1,0])

    # test 3: circle with gaussian noise sig=0.1
    x, y, dCor = xycirc(num_samples, noise=0.1, ax=axs[0,1])

    # test 4: circle with gaussian noise sig=0.01
    x, y, dCor = xycirc(num_samples, noise=0.01, ax=axs[1,1])

    for ax in axs.T[0]:
        ax.set_xlim([-1.25,1.25])
        ax.set_ylim([-1,1.5])

    for ax in axs.T[1]:
        ax.set_xlim([-1.25,1.25])
        ax.set_ylim([-1.25,1.25])

    plt.tight_layout()
    plt.savefig('distcorr_test1D.png')
    plt.close(fig)

    # test 5
    num_samples = 10000
    cov = [[4,1],[1,1]]
    xy=np.random.multivariate_normal([0,0], cov, size=num_samples)
    z = xy[:,0]+np.random.uniform(low=-0.3, high=0.3, size=num_samples)
    xyz = np.hstack((xy,np.atleast_2d(z).T))
    dCor1 = distcorr(xyz)
    dCor2 = distcorr(xyz,rv_sizes=[2,1])

    fig, ax = plt.subplots(1,1,figsize=(6,6))
    plt.plot(xy[:,0],xy[:,1],'s',label='x vs y:{:.2f}'.format(dCor1[1,0]))
    plt.plot(xy[:,0],z,'s',label='x vs z: {:.2f}'.format(dCor1[2,0]))
    plt.text(3,-6,'xy vs z: {:.2f}'.format(dCor2[1,0]),fontsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig('distcorr_test2D.png')
    plt.close(fig)


