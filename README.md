# Robust smoothing algorithm with missing values

This code is a Python implementation of the ``smoothn(y)`` algorithm described by García (2010). The original source code (Matlab) can be found [here](https://www.biomecardio.com/en/download.html).

The smoothn function achieves a fast and robust smoothing of multidimensional gridded data by using the discrete cosine transform to fit the points. It can handle missing data, providing either a solution for predict data with a low smoothing factor. The robust approach works well for outlier points, wich are not detected as missing values, so the approximation does not tries to overfits to those points.

## References

[1] Garcia, D. (2010). Robust smoothing of gridded data in one and higher dimensions with missing values. Computational Statistics & Data Analysis; 54:1167-1178.
[doi:10.1016/j.csda.2009.09.020](https://doi.org/10.1016/j.csda.2009.09.020).

[2] Wang, G., Garcia, D., Liu, Y., de Jeu, R., and Johannes Dolman, A. (2012). A three-dimensional gap filling method for large geophysical datasets: Application to global satellite soil moisture observations. Environmental Modelling & Software; 30:139-142. [doi:10.1016/j.envsoft.2011.10.015](https://doi.org/10.1016/j.envsoft.2011.10.015)
