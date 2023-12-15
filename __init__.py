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

[2] Wang, G., Garcia, D., Liu, Y., de Jeu, R., and Johannes Dolman, A. (2012). A three-dimensional gap filling method
for large geophysical datasets: Application to global satellite soil moisture observations.
Environmental Modelling & Software, 2012; 30:139-142.
"""
__name__ = 'smoothn'
__version__ = '1.1.2'
__release__ = '2023-12'

__all__ = [
    'smoothn',
    'inpaintn'
]
# ===== IMPORTS =======================================
from .smoothn import smoothn
from .inpaintn import inpaintn
