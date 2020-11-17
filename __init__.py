# -*- coding: utf-8 -*-
"""
=======================================================
===                   SMOOTHN                       ===
=======================================================
---  Santiago Parraguez Cerda                       ---
---  University of Bremen, Germany                  ---
---  mail: sanparra@uni-bremen.de                   ---
---  2023                                           ---
=======================================================

Tool for smooth and replace missing data by interpolation using the discrete cosine transform.

[1] Garcia D. (2010) Robust smoothing of gridded data in one and higher dimensions with missing values.
Computational Statistics & Data Analysis, 2010; 54:1167-1178.
"""
__name__ = 'smoothn'
__version__ = '1.1'
__release__ = '2023-03'

__all__ = [
    'smoothn'
]
# ===== IMPORTS =======================================
from .smoothn import smoothn
