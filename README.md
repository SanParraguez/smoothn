# Smoothn(y)

This code is a Python implementation of the algorithm described by García (2010).

The smoothn function achieves a fast, robust smoothing to gridded data in multiple dimensions. It can handle missing data, providing either a solution for predict data with a low smoothing factor. The robust approach works fine for outlier points, wich are not detected as missing values, so the approximation does not tries to overfits to that points.

[1] Garcia D. (2010) Robust smoothing of gridded data in one and higher dimensions with missing values. Computational Statistics & Data Analysis, 2010; 54:1167-1178.
