import numpy as np
from sklearn.decomposition import PCA

def get_pca_data(data,pca_var_explained=0.95,n_components=None):
    """
    Thin wrapper on scikit-learn PCA to return the components and
    corresponding coefficients.

    Parameters
    ----------
    data: dataset of ABM runs
        - Rows are different ABM parameter runs
        - Columns are time points of hospitalizations then deaths
    n_components: number of components
        - this takes precedence over pca_var_explained
    return_pca_obj: return the scikit-learn object

    Returns
    -------
    components: PCA components
    coefs: PCA component coefficients (for reconstruction)
    var_exp: variance explained by these components and coefficients
    """
    if n_components != None:
        pca = PCA(n_components=n_components)
    else:
        pca = PCA(n_components=pca_var_explained)
    pca.fit(data)

    # Grab components, coefficients
    components = pca.components_.T
    coefs = pca.transform(data)
    var_exp = np.sum(pca.explained_variance_ratio_)

    return components, coefs, var_exp

def reconstruct_data_from_pca(coefficients, components, transform=None):
    """
    Combine coefficients a_i with components c_i to approximate data

    Original PCA decomposition is:

    data = a_i*c_i + ... + a_2n*c_2n

    Parameters
    ----------
    coefficients: PCA component coefficients (for reconstruction)
        - can have multiple rows (representing multiple samples)
    components: PCA components (as columns)
    transform: Any scikit-learn transformation prior to PCA

    Returns
    -------
    reconstructed_data: data built from PCA components and coefficients
    """
    if transform == None:
        reconstructed_data = coefficients @ components.T
    else:
        reconstructed_data = transform.inverse_transform(coefficients @ components.T)
    return reconstructed_data

def inverse_transform_data(data,transform,threshold=1e-5):
    """
    Thin wrapper on scikit-learn inverse transform
    - Maintains pandas format
    - Rounds off small numbers

    Parameters
    ----------
    data: pandas dataframe
    transform: scikit-learn transformation to invert

    Returns
    -------
    transformed_data: inverse transformed data
    """
    transformed_data = data.copy()
    if isinstance(transformed_data,np.ndarray):
        transformed_data = transform.inverse_transform(data)
        transformed_data[np.abs(transformed_data) < threshold] = 0
    else:
        transformed_data.iloc[:,:] = transform.inverse_transform(data)
        transformed_data.mask(np.abs(transformed_data) < threshold, 0, inplace=True)
    return transformed_data

def absolute_relative_err(truth, prediction):
    """
    Calculate the absolute relative error of y and yhat:

    abs_rel_err = abs(y - yhat) / abs(y)

    * Infs are mapped to nan

    Parameters
    ----------
    truth: true data
    prediction: predicted data

    Returns
    -------
    abs_rel_err: absolute relative error
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        abs_rel_err = np.abs(truth - prediction) / np.abs(truth)
    if isinstance(abs_rel_err,np.ndarray):
        abs_rel_err[abs_rel_err == np.inf] = np.nan
    else:
        abs_rel_err.replace(np.inf,np.nan,inplace=True)
    return abs_rel_err

def err_statistics(abs_rel_err,axis=None):
    """
    Calculate median, 5% and 95% quantiles of error

    * Nans are ignored, numpy/pandas robust

    Parameters
    ----------
    err: calculated error
    axis: axis to calculate on (None = operate on both)

    Returns
    -------
    five: 5% quantile of error (across rows and columns)
    median: median of error (across rows and columns)
    nfive: 95% quantile of error (across rows and columns)
    """
    if isinstance(abs_rel_err, np.ndarray) | (axis == None):
        five = np.nanquantile(abs_rel_err,.05,axis=axis)
        median = np.nanmedian(abs_rel_err,axis=axis)
        nfive = np.nanquantile(abs_rel_err,.95,axis=axis)
    else:
        five = abs_rel_err.quantile(.05,axis=axis)
        median = abs_rel_err.median(axis=axis)
        nfive = abs_rel_err.quantile(.95,axis=axis)
    return five,median,nfive
