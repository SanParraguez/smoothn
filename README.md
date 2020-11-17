# Robust smoothing algorithm with missing values

This code is a Python implementation of the algorithm SMOOTHN(Y) described by García (2010). The original source code (Matlab) can be found [here](https://www.biomecardio.com/en/download.html).

The smoothn function achieves a fast, robust smoothing to gridded data in multiple dimensions using the discrete cosine transform to fit the points. It can handle missing data, providing either a solution for predict data with a low smoothing factor. The robust approach works fine for outlier points, wich are not detected as missing values, so the approximation does not tries to overfits to that points.

[1] Garcia D. (2010) Robust smoothing of gridded data in one and higher dimensions with missing values. Computational Statistics & Data Analysis, 2010; 54:1167-1178.
[doi:10.1016/j.csda.2009.09.020](https://doi.org/10.1016/j.csda.2009.09.020).
