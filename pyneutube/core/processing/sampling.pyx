# core/processing/sampling.pyx

"""
sampling.pyx

Utilities for sampling image data by coordinates or resampling images.
"""

# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: language_level=3

import numpy as np
cimport numpy as cnp
from scipy.ndimage import map_coordinates
cimport cython

from libc.math cimport NAN, ceil, floor

cnp.import_array()

# Define numpy array types
ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.intp_t ITYPE_t

# The signal volume is float32 whenever that is exact, so the samplers are
# compiled for both widths. Every sample is widened to `double` before any
# arithmetic, so both specialisations agree on identical values.
ctypedef fused IMAGE_t:
    cnp.float32_t
    cnp.float64_t

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def sample_voxels_minor(cnp.ndarray image,
#                           double[:, :] coords,
#                           int order=1):
#    """
#    Optimized version using memory views for better performance.
#    Works with any dimensionality of input image.
#    """
#    cdef:
#        Py_ssize_t n_points = coords.shape[0]
#        Py_ssize_t n_dims = coords.shape[1]
#        Py_ssize_t i, j
#        double[:, :] coords_zyx_view
#        cnp.ndarray[double, ndim=2, mode="c"] coords_zyx
#        cnp.ndarray[double, ndim=2, mode="c"] coords_transposed
#    
#    # Create coordinate arrays
#    coords_zyx = np.empty((n_points, n_dims), dtype=np.float64, order='C')
#    coords_zyx_view = coords_zyx
#    
#    # Reverse coordinate order with memory view for speed
#    for i in range(n_points):
#        for j in range(n_dims):
#            coords_zyx_view[i, j] = coords[i, n_dims - 1 - j]
#    
#    # Transpose for map_coordinates
#    coords_transposed = np.transpose(coords_zyx).copy(order='C')
#    
#    return map_coordinates(image, coords_transposed, order=order,
#                          mode='constant', cval=0.0, prefilter=(order > 1))

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double trilinear_interpolate(IMAGE_t[:, :, ::1] image,
                                         double x, double y, double z,
                                         int width, int height, int depth) nogil:
    """Fast trilinear interpolation for 3D images."""
    cdef:
        int x0, x1, y0, y1, z0, z1
        double wx, wy, wz
        double c00, c01, c10, c11, c0, c1, val
    
    # Get integer coordinates
    x0 = <int>floor(x)
    x1 = x0 + 1
    y0 = <int>floor(y)
    y1 = y0 + 1
    z0 = <int>floor(z)
    z1 = z0 + 1
    
    # Check bounds
    if x0 < 0 or x1 >= width or y0 < 0 or y1 >= height or z0 < 0 or z1 >= depth:
        return NAN
    
    # Calculate weights
    wx = x - x0
    wy = y - y0
    wz = z - z0
    
    # Trilinear interpolation
    c00 = image[z0, y0, x0] * (1.0 - wx) + image[z0, y0, x1] * wx
    c01 = image[z0, y1, x0] * (1.0 - wx) + image[z0, y1, x1] * wx
    c10 = image[z1, y0, x0] * (1.0 - wx) + image[z1, y0, x1] * wx
    c11 = image[z1, y1, x0] * (1.0 - wx) + image[z1, y1, x1] * wx
    
    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy
    
    val = c0 * (1.0 - wz) + c1 * wz
    
    return val


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double nearest_neighbor(double[:, ::1] image, double x, double y) nogil:
    """Nearest neighbor interpolation for 2D images."""
    cdef:
        int ix, iy
        int height = image.shape[0]
        int width = image.shape[1]
    
    ix = <int>(x + 0.5)
    iy = <int>(y + 0.5)
    
    if ix < 0 or ix >= width or iy < 0 or iy >= height:
        return NAN
    
    return image[iy, ix]


@cython.boundscheck(False)
@cython.wraparound(False)
def sample_voxels(IMAGE_t[:, :, ::1] image,
                   double[:, ::1] coords,
                   int order=1):
    """
    Fast 3D sampling with custom interpolation.

    Parameters
    ----------
    image : 3D array
        Input image, float32 or float64. Samples are widened to double before
        any arithmetic, so both widths agree on identical values.
    coords : array, shape (N, 3)
        Coordinates as (x, y, z)
    order : int
        0 for nearest neighbor, 1 for linear
    
    Returns
    -------
    values : array, shape (N,)
        Sampled values
    """
    cdef:
        Py_ssize_t n_points = coords.shape[0]
        Py_ssize_t i
        double x, y, z
        int ix, iy, iz

        int depth = image.shape[0]
        int height = image.shape[1]
        int width = image.shape[2]

        cnp.ndarray[double, ndim=1] result = np.empty(n_points, dtype=np.float64)
        double[:] result_view = result
    
    with nogil:
        if order == 0:
            # Nearest neighbor for 3D
            for i in range(n_points):
                x = coords[i, 0]
                y = coords[i, 1]
                z = coords[i, 2]
                # Simple nearest neighbor
                ix = <int>(x + 0.5)
                iy = <int>(y + 0.5)
                iz = <int>(z + 0.5)
                if (ix >= 0 and ix < width and 
                    iy >= 0 and iy < height and 
                    iz >= 0 and iz < depth):
                    result_view[i] = image[iz, iy, ix]
                else:
                    result_view[i] = NAN
        else:
            # Trilinear interpolation
            for i in range(n_points):
                x = coords[i, 0]
                y = coords[i, 1]
                z = coords[i, 2]
                result_view[i] = trilinear_interpolate(image, x, y, z, width, height, depth)
    
    return result

