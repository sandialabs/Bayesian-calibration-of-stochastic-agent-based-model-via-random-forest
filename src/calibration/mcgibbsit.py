import numpy as np
import scipy
import pandas as pd

def mcgibbsit(data, q = 0.5, r = 0.025, s = 0.95, eps = 0.001, correct_cor=True):
    """
    Python reimplementation of mcgibbsit R package
        https://cran.r-project.org/web/packages/mcgibbsit/index.html

    Inputs:
       data: input chain data. shape - (niters, nchains, nparams)
       q: quantile to check for autocorrelation
       r: iterations after burn in parameter
       s: niterations parameter
       eps: burn in epsilon parameter

    Outputs:
    """
    # minimum number of iterations
    phi = scipy.stats.norm.ppf(0.5 * (1 + s))
    nmin = int(np.ceil((q * (1 - q) * phi**2)/r**2))

    # Num particles
    nchains = data.shape[1]

    res_matrix = np.zeros((data.shape[2], 6))
    for j in range(data.shape[2]):
        kthin = 0
        bic = 1
        quant = np.quantile(data[:,:,j], q)
        dichot = data[:,:,j] <= quant

        while bic >= 0:
            kthin += 1
            tmp = np.zeros((dichot.shape[1],2,2,2))
            for i in range(dichot.shape[1]):
                d = dichot[::kthin,i]
                three_moves = np.vstack((d[:-2],d[1:-1],d[2:])).T
                states, counts = np.unique(three_moves, axis=0, return_counts=True)
                for i1,n1 in enumerate([False, True]):
                    for i2,n2 in enumerate([False, True]):
                        for i3,n3 in enumerate([False, True]):
                            match_loc = np.where(np.all(states == [n1,n2,n3], axis=1))[0]
                            if len(match_loc) != 0:
                                tmp[i,i1,i2,i3] = counts[match_loc]

            ## add all of the transition matrixes together
            testtran = np.sum(tmp, axis=0)

            ## compute the likelihoood
            g2 = 0
            for i1 in range(2):
                for i2 in range(2):
                    for i3 in range(2):
                        if testtran[i1, i2, i3] != 0:
                          fitted = (np.sum(testtran[i1, i2, :]) * np.sum(testtran[:, i2, i3])) / (np.sum(testtran[:, i2, :]))
                          g2 = g2 + testtran[i1, i2, i3] * np.log(testtran[i1, i2, i3]/fitted) * 2

            ## compute bic
            bic = g2 - np.log( np.sum(testtran) - 2 ) * 2

        ## estimate the parameters of the first-order markov chain
        alpha = np.sum(testtran[0, 1, :]) / (np.sum(testtran[0, 0, :]) + np.sum(testtran[0, 1, :]))
        beta  = np.sum(testtran[1, 0, :]) / (np.sum(testtran[1, 0, :]) + np.sum(testtran[1, 1, :]))

        ## compute burn in
        if np.isnan(alpha):
            alpha = 0
        if np.isnan(beta):
            beta = 0
        tempburn = np.log((eps * (alpha + beta)) / max(alpha, beta)) / (np.log(np.abs(1 - alpha - beta)))

        nburn = int(np.ceil(tempburn) * kthin)

        ## compute iterations after burn in
        tempprec = ((2 - alpha - beta) * alpha * beta * phi**2) / (((alpha + beta)**3) * r**2)
        nkeep  = np.ceil(tempprec * kthin)

        ## compute the correlation
        if nchains > 1 & correct_cor:
            varmat = np.cov(dichot.T)
            denom = np.mean(np.diag(varmat))  # overall variance
            np.fill_diagonal(varmat, 0)
            numer = np.mean(np.array(varmat)) # overall covariance
            rho = numer / denom
        else:
            rho = 1.0

        ## inflation factors
        iratio = (nburn + nkeep) / nmin
        R = ( 1 + rho * (nchains - 1) )
        res_matrix[j,0] = np.ceil( nburn * nchains ) # M
        res_matrix[j,1] = np.ceil( nkeep * R       ) # N
        res_matrix[j,2] = np.sum(res_matrix[j,0:2])  # total
        res_matrix[j,3] = nmin                       # nmin
        res_matrix[j,4] = round(iratio, 3)          # I
        if nchains > 1 & correct_cor:
            res_matrix[j,5] = round(R, 3)           # R
        else:
            res_matrix[j,5] = np.nan                 # R
        res_df = pd.DataFrame(res_matrix, columns=["M", "N", "total", "Nmin", "I", "R"])
    return res_df
