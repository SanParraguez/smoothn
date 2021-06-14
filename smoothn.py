# -*- coding: utf-8 -*-
"""
Santiago Parraguez Cerda
University of Chile - 2020
santiago.parraguez@ug.uchile.cl
v1.1

z = smoothn(y)

Tool for smooth and replace missing data by interpolation using the discrete cosine transform.

[1] Garcia D. (2010) Robust smoothing of gridded data in one and higher dimensions with missing values.
Computational Statistics & Data Analysis, 2010; 54:1167-1178.
"""
# ====== IMPORT LIBRARIES ======
import numpy as np
from scipy.fftpack import dctn, idctn
from scipy.optimize import minimize
from scipy.stats import median_abs_deviation
from scipy.ndimage.morphology import distance_transform_edt
# ==============================
def smoothn(y, s=None, tolz=1e-3, z0=None, w=None, di=None, rbst=False) -> np.ndarray:
    """
    Fast smooths an array of data based on the cosine discrete transform. Allows to choose the smoothing parameter,
    or calculate it using a GCVscore. Also it implements a robust iteration (False by default) to ignore outlier points.

    :param y: data to be smoothed and filled
    :type y: np.ma.masked_array or np.ndarray
    :param float s: smoothing parameter, calculated automatically if not provided to minimize gcv score
    :param float tolz: tolerance of iteration over z (prediction of data)
    :param np.ndarray z0: initial guess for z, calculated with nearest neighbor if not provided
    :param np.ndarray w: weights array, if not provided assumes all data has the same confidence
    :param tuple di: grid dimensions, assumed regular if not given
    :param bool rbst: indicates if the robust iteration is executed
    :return: gives the smoothed gridded data, with missing values filled with the prediction done
    """
    # Create MaskedArray in case it contains (NaN, Inf)
    y = _to_masked(y)

    # Define dimensions
    n = y.size
    d = y.ndim
    shape = y.shape
    nf = y.count()

    # Create spacing vector
    if di is None:
        di = np.ones(d)
    di = di/np.max(di)

    # Create weights matrix
    if w is None:
        w = np.ones(shape)
    w = w/np.max(w) * ~y.mask
    if np.any(w < 0):
        raise ValueError('All weights must be >=0')

    # Define if it has a weight matrix and if the smoothing factor needs to be calculated
    isweight = np.any(w < 1)
    isauto = s is None
    isrobust = rbst

    # Create Lambda tensor (see reference) and calculate Gamma if S is given
    lamb = _create_lambda(y, di)
    if not isauto:
        gamm = 1 / (1 + s * lamb ** 2)
        s_lim = None
    else:
        # Set limit values for S to be calculated
        gamm = None
        h_lim = np.array([1-1e-5, 1e-5])
        s_lim = (((1 + np.sqrt(1 + 8 * h_lim ** (2 / d))) / (4 * h_lim ** (2 / d))) ** 2 - 1) / 16

    # ===== Initial values for iteration =====
    w_tot = w.copy()
    # Initial condition for z if is weighted
    if isweight:
        # Initialize z
        if z0:
            z = z0.copy()
        else:
            z = _initz(y)
    else:
        z = np.zeros(shape)

    z0 = z

    y_fill = y.filled(0.0)  # Assign arbitrary value to missing data
    # ===== START ITERATION =====
    # Set limits of iterations
    n_robust_iter = 3
    max_iter = 1000
    for i in range(n_robust_iter):

        aow = np.sum(w_tot)/n
        miss = np.inf

        for k in range(max_iter):

            if miss < tolz:
                break

            # Calculate discrete cosine transform
            xdc = w_tot * (y_fill - z) + z
            dcty = dctn(xdc, norm='ortho')

            # Estimate value of s
            if isauto:
                s = minimize(_gcv_score, x0=np.array(1.0),
                             args=(y, w_tot, n, nf, dcty, lamb, aow), bounds=[s_lim]).x
                gamm = 1 / (1 + s * lamb ** 2)

            # Calculate inverse discrete cosine transform
            z = idctn(gamm * dcty, norm='ortho')

            # Check jump, if it's less than tolerance iterations are terminated
            res = z0-z
            miss = isweight * np.linalg.norm(res, 2) / np.linalg.norm(z, 2)
            z0 = z

        if isrobust:
            u = _studentized(y_fill, z, ~y.mask, s)
            w_tot = w * _biweight(u)
            isweight = True
        else:
            break

    return z

def _gcv_score(s, y, w_tot, n, nf, dcty, lamb, aow=None) -> float:
    """
    Calculates the Generalised cross-validation (GCV) score based on the weights used.

    :param float s: smoothing parameter of the algorithm
    :param np.ma.MaskedArray y: masked data
    :param np.array w_tot: array of weights assigned to data
    :param int n: size of the array y, not calculated to save resources
    :param int nf: number of finite elements in y, not calculated to save resources
    :param np.ndarray dcty: discrete cosine transform of the data, considering the prediction and weights
    :param np.ndarray lamb: Lambda tensor of the data y
    :param float aow: amount of weight in the data, is the mean of the weights
    :return: returns the calculated GCV score for the data given
    """
    gamm = 1 / (1 + s * lamb ** 2)

    if aow > 0.9:
        a = dcty*(gamm-1)
        rss = np.linalg.norm(a, 2)**2
    else:
        y_hat = idctn(dcty*gamm, norm='ortho')
        dy = y.compressed() - y_hat[~y.mask]
        rss = np.linalg.norm(np.sqrt(w_tot[~y.mask])*dy, 2)**2

    trh = gamm.sum()
    gcv = rss/nf/(1-trh/n)**2
    return gcv

def _to_masked(y_) -> np.ma.masked_array:
    """
    Verify that y_ has a mask, it it doesn't have one it creates it from non-finite values

    :param y_: array of values to mask
    :type y_: np.ma.masked_array or np.ndarray
    :return: masked data of y_
    """
    if not np.ma.isMaskedArray(y_):
        y_ = np.ma.MaskedArray(y_, ~np.isfinite(y_))
    return y_

def _create_lambda(y, dx) -> np.ndarray:
    """
    Creates a Lambda tensor for the inpaintn(x) implementation. Lambda contains the eingenvalues of the difference
    matrix used in this penalized least squares process.

    :param np.ndarray y: data to process
    :param tuple dx: spatial difference between steps
    :return: array with Lambda tensor (see reference)
    """
    d = y.ndim
    lamb_ = np.zeros(y.shape)
    for i in range(d):
        siz0 = np.ones(d, dtype=np.int)
        siz0[i] = y.shape[i]
        lamb_ += (2 - 2 * np.cos(np.pi * np.arange(y.shape[i]).reshape(siz0) / y.shape[i])) / dx[i] ** 2
    return lamb_

def _initz(y) -> np.ndarray:
    """
    Initialize the estimation values of y with a nearest interpolation. Implemented for 1D, 2D and 3D.

    :param np.ma.MaskedArray y:
    :return: array with the first guess of z
    """
    if np.any(y.mask):
        _, li = distance_transform_edt(y.mask, return_indices=True)
        z0 = np.zeros(y.shape)
        z0[~y.mask] = y[~y.mask]
        if y.ndim == 1:
            z0[y.mask] = y[li[0][y.mask]]
        elif y.ndim == 2:
            z0[y.mask] = y[li[0][y.mask], li[1][y.mask]]
        elif y.ndim == 3:
            z0[y.mask] = y[li[0][y.mask], li[1][y.mask], li[2][y.mask]]
        else:
            raise NotImplementedError(f'Algorithm not implemented for {y.ndim}-dimensional data')
    else:
        z0 = y
    return z0

def _studentized(y, z, mask, s) -> np.ndarray:
    """
    Returns the Studentized residual approximated by the median absolute deviation and the diagonal elements
    of the hat matrix H.

    :param np.ndarray y: data received
    :param np.ndarray z: data estimation
    :param np.ndarray mask: original mask of the data (y.mask)
    :param float s: smoothing factor
    :return: array with studentized residuals of the approximation
    """
    r = (y - z)
    n = r.ndim
    mad_ = median_abs_deviation(r[mask], axis=None)
    a = np.sqrt(1 + 16 * s)
    h = (np.sqrt(1 + a) / (np.sqrt(2) * a)) ** n
    u = r / (1.4826 * mad_ * np.sqrt(1 - h))
    return u

def _biweight(u, c=4.685) -> np.ndarray:
    """
    Function to calculate the bisquare weighting of the residuals.

    :param np.ndarray u: Studentized residual adjusted for standard deviation
    :param float c: residual limit
    :return: array with biweights of the residuals
    """
    subset = np.less(np.abs(u), c)
    bw = (1 - (u / c) ** 2) ** 2 * subset
    return bw
