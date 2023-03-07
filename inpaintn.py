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

def inpaintn(data, s=None, structure=None, robust=False, max_size=5, grid=None):
    """
    Fill missing values with a smoothn algorithm, based on the cosine discrete transform.
    Returns the filled array.

    Parameters
    ----------
    data : np.ma.MaskedArray
        n-dimensional data to be filled.
    s : None or float
        Smoothing parameter, calculated automatically if not provided to minimize GCV score.
    structure : int or tuple
        Structure to count empty spaces. See scipy.ndimage.label function.
    robust : bool
        Indicates if a robust iteration is executed to avoid outliers.
    max_size : int
        Maximum number of pixels in hole to be filled.
    grid : tuple, list
        Grid dimensions, assumed regular [np.ones(data.ndim)] if not given.
    """
    if structure is None:
        structure = np.ones(tuple([3]*data.ndim), dtype=np.int16)
    labeled, n_labels = label(data.mask, structure=structure)
    sizes = sum_labels(np.ones_like(data), labeled, index=range(n_labels + 1))
    mask_size = sizes <= max_size
    mask_clean = mask_size[labeled]

    z = data.copy()
    z[mask_clean] = smoothn(data, s, robust=robust, di=grid)[mask_clean]

    return z


