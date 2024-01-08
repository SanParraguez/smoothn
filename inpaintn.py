# -*- coding: utf-8 -*-
"""
Implementation of inpaintn(y)
"""
__all__ = [
    'inpaintn'
]

# ====== IMPORT LIBRARIES ======

import numpy as np
from scipy.ndimage import label, sum_labels
from .smoothn import smoothn


# ================================================================================================

def inpaintn(data, s=None, structure=None, max_size=None,  robust=False, **kwargs):
    """
    Fill missing values with a smoothn algorithm, based on the cosine discrete transform.
    Returns the filled array.

    Parameters
    ----------
    data : np.ma.MaskedArray or np.ndarray
        n-dimensional data to be filled.
    s : None or float
        Smoothing parameter, calculated automatically if not provided to minimize GCV score.
    structure : int or tuple
        Structure to count empty spaces. See scipy.ndimage.label function.
    max_size : int
        Maximum number of pixels in hole to be filled. Defaults to None, which fills every invalid
        value
    robust : bool
        Indicates if a robust iteration is executed to avoid outliers.
    """
    if isinstance(data, np.ndarray):
        data = np.ma.masked_invalid(data)

    z = data.copy()

    # fill complete array
    if max_size in [None, 0, -1]:

        z[data.mask] = smoothn(data, s=s, robust=robust, **kwargs)[data.mask]

    # fill small holes with max_size
    elif max_size > 0:

        if structure is None:
            structure = np.ones(tuple([3] * data.ndim), dtype=np.int16)
        labeled, n_labels = label(data.mask, structure=structure)

        sizes = sum_labels(np.ones_like(data), labeled, index=range(n_labels + 1))
        mask_size = sizes <= max_size
        mask_clean = mask_size[labeled]

        z[mask_clean] = smoothn(data, s=s, robust=robust, **kwargs)[mask_clean]

    # raise otherwise
    else:
        raise ValueError(f'max_size should be either 0 or a positive int, found {max_size}')

    return z


