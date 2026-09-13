# cython: language_level=3
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

from abc import abstractmethod

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport ceil, exp, fabs, isnan, sqrt

from typing import Literal, Optional

from pyneutube.core.processing.transform cimport rotate_by_theta_psi_fast

from .config import Defaults

# Type definitions
np.import_array()
ctypedef np.float64_t DTYPE_t


cdef class SegmentFilter:
    """
    Base class for radial filters, optimized with Cython.
    """
    cdef dict _cache
    cdef public int z_dim
    cdef public double step
    cdef public double max_dist2
    cdef public object _coords_2d  # Changed to object type
    cdef public object _dist2_2d
    cdef public object _weights_2d
    
    def __cinit__(self, double step=0.2, double max_dist2=2.66, int z_dim=Defaults.SEG_LENGTH):
        if step <= 0:
            raise ValueError("Step size must be positive.")
        if max_dist2 <= 0:
            raise ValueError("Maximum squared distance must be positive.")
        
        self.z_dim = z_dim
        self.step = step
        self.max_dist2 = max_dist2
        self._cache = {}
        self._initialize_filter()
    
    cdef void _initialize_filter(self):
        """Initialize the filter grid and weights."""
        cdef tuple cache_key = (self.step, self.max_dist2, type(self))
        
        if cache_key in self._cache:
            cached_data = self._cache[cache_key]
            self._coords_2d = cached_data[0]
            self._dist2_2d = cached_data[1]
            self._weights_2d = cached_data[2]
        else:
            coords_2d, dist2_2d = self._build_grid()
            weights_2d = self._call_kernel_func(dist2_2d)
            
            self._cache[cache_key] = (coords_2d, dist2_2d, weights_2d)
            self._coords_2d = coords_2d
            self._dist2_2d = dist2_2d
            self._weights_2d = weights_2d
    
    cdef tuple _build_grid(self):
        """Build coordinate grid efficiently."""
        cdef:
            double max_dist = ceil(sqrt(self.max_dist2))
            int n_points = <int>(2 * max_dist / self.step) + 1
            double[:, ::1] X = np.zeros((n_points, n_points), dtype=np.float64)
            double[:, ::1] Y = np.zeros((n_points, n_points), dtype=np.float64)
            double[:, ::1] dist2 = np.zeros((n_points, n_points), dtype=np.float64)
            int i, j, count = 0
            double val
        
        # Create meshgrid
        for i in range(n_points):
            val = -max_dist + i * self.step
            for j in range(n_points):
                X[i, j] = -max_dist + j * self.step
                Y[i, j] = val
                dist2[i, j] = X[i, j]**2 + Y[i, j]**2
                if dist2[i, j] < self.max_dist2:
                    count += 1
        
        # Extract valid coordinates
        cdef double[:, ::1] coords = np.zeros((count, 2), dtype=np.float64)
        cdef double[:] dist2_valid = np.zeros(count, dtype=np.float64)
        cdef int k = 0
        
        for i in range(n_points):
            for j in range(n_points):
                if dist2[i, j] < self.max_dist2:
                    coords[k, 0] = X[i, j]
                    coords[k, 1] = Y[i, j]
                    dist2_valid[k] = dist2[i, j]
                    k += 1
        
        return np.asarray(coords), np.asarray(dist2_valid)
    
    cdef object _call_kernel_func(self, object dist2):
        """Wrapper for Python-implemented kernel_func"""
        return self.kernel_func(dist2)
    
    @abstractmethod
    def kernel_func(self, dist2):
        """Abstract method to be implemented by subclasses"""
        pass
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple generate_filter_by_seg(self, object seg, str rel_pos):
        """Generate 3D filter field for a segment."""
        cdef tuple field_2d = self._discrete_field_2d_scaling(seg, False, 0)
        cdef tuple field_3d = self._discrete_field_3d(seg, field_2d[0], field_2d[1], field_2d[2])

        cdef np.ndarray coords_3d = field_3d[0]
        if rel_pos == 'global':
            origin_arr = seg.start_coord
            coords_3d += origin_arr

            return coords_3d, field_3d[1], field_3d[2]

        return field_3d
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef tuple _discrete_field_3d(self, object seg,
                                double[:,::1] coords_2d,
                                double[:] dist2_2d,
                                double[:] weights_2d):
        """Generate 3D field from 2D coordinates."""
        cdef:
            Py_ssize_t num_coords = coords_2d.shape[0]
            Py_ssize_t zdim = self.z_dim
            Py_ssize_t total_points = num_coords * zdim
            double[:,::1] coords_3d = np.empty((total_points, 3), dtype=np.float64)
            double[:] weights_3d = np.empty(total_points, dtype=np.float64)
            double[:] dist2_3d = np.empty(total_points, dtype=np.float64)
            Py_ssize_t i, j, idx
            double z_step = (seg.length - 1.0) / (zdim - 1)
        
        # Build 3D coordinates
        with nogil:
            for i in range(num_coords):
                for j in range(zdim):
                    idx = i * zdim + j
                    
                    coords_3d[idx, 0] = coords_2d[i, 0]
                    coords_3d[idx, 1] = coords_2d[i, 1]
                    coords_3d[idx, 2] = j * z_step
                    weights_3d[idx] = weights_2d[i] / zdim
                    dist2_3d[idx] = dist2_2d[i]
        
        # Apply rotation if needed
        if seg.theta != 0 or seg.psi != 0:
            coords_3d = rotate_by_theta_psi_fast(coords_3d, seg.theta, seg.psi, None)
        
        # Convert memoryviews to numpy arrays before returning
        coords_3d_arr = np.asarray(coords_3d)
        dist2_3d_arr = np.asarray(dist2_3d)
        weights_3d_arr = np.asarray(weights_3d)

        return coords_3d_arr, dist2_3d_arr, weights_3d_arr
    
    @abstractmethod
    def _discrete_field_2d_scaling(self, object seg, bint rotate=False, double z=0):
        """Abstract method to be implemented by subclasses"""
        pass

    def __call__(self, seg: object, is_2d: bool = False, z: Optional[float] = None, rel_pos: Literal['global', 'local']='global'):
        if is_2d:
            return self._discrete_field_2d_scaling(seg, rotate=True, z=z)
        else:
            return self.generate_filter_by_seg(seg, rel_pos)

    def __repr__(self):
        return f"{self.__class__.__name__}(step={self.step}, max_dist2={self.max_dist2}, z_dim={self.z_dim})"
    

cdef class MexicanHatFilter(SegmentFilter):
    """Optimized Mexican Hat filter implementation."""
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def kernel_func(self, np.ndarray[DTYPE_t, ndim=1] dist2):
        """Mexican Hat kernel function."""
        return (1 - dist2) * np.exp(-dist2)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef tuple _discrete_field_2d_scaling(self, object seg, bint rotate=False, double z=0):
        """Scale 2D coordinates according to segment parameters."""
        cdef:
            Py_ssize_t n = self._coords_2d.shape[0]
            double[:,::1] coords = np.asarray(self._coords_2d).copy()
            double[:] dist2 = np.asarray(self._dist2_2d).copy()
            double[:] weights = np.asarray(self._weights_2d).copy()

            double radius = seg.radius
            double scale = seg.scale
            double radius_scale = radius * scale
            double sqrt_r = sqrt(radius)
            double sqrt_s = sqrt(sqrt(scale))

            double[:, ::1] coords_3d = np.empty((n, 3), dtype=np.float64)
            Py_ssize_t i
            double val, enorm, norm, alpha
            
            double sum_abs_weights = 0.0
        
        with nogil:
            for i in range(n):
                val = weights[i]
                if val > 0.0:
                    coords[i,0] *= radius_scale
                    coords[i,1] *= radius
                    weights[i]  = val * sqrt_r * sqrt_s
                else:
                    enorm = dist2[i]
                    norm  = sqrt(enorm)
                    alpha = norm - 1.0
                    if enorm < 1.0:
                        alpha /= sqrt(enorm)
                    elif enorm > 4.0:
                        alpha = (alpha / sqrt(enorm)) * 2.0
                    coords[i,0] *= (radius_scale / norm) * (1.0 + alpha)
                    coords[i,1] *= (radius / norm) * (1.0 + alpha)

                sum_abs_weights += fabs(weights[i])

            for i in range(n):
                weights[i] /= sum_abs_weights
        
        if rotate:
            with nogil:
                for i in range(n):
                    coords_3d[i,0] = coords[i,0]
                    coords_3d[i,1] = coords[i,1]
                    coords_3d[i,2] = z
            if seg.theta != 0 or seg.psi != 0:
                coords_3d = rotate_by_theta_psi_fast(coords_3d, seg.theta, seg.psi, None)
            coords = coords_3d

        return coords, dist2, weights
    # cpdef tuple _discrete_field_2d_scaling(self, object seg, bint rotate=False, double z=0):
    #     """Scale 2D coordinates according to segment parameters."""
    #     cdef:
    #         np.ndarray[DTYPE_t, ndim=2] coords = np.asarray(self._coords_2d).copy()
    #         np.ndarray[DTYPE_t, ndim=1] dist2 = np.asarray(self._dist2_2d).copy()
    #         np.ndarray[DTYPE_t, ndim=1] weights = np.asarray(self._weights_2d).copy()
    #         np.ndarray positive_mask = weights > 0
    #         double radius = seg.radius
    #         double scale = seg.scale
    #         np.ndarray coords_3d
        
    #     # Positive region scaling
    #     coords[positive_mask, 0] *= radius * scale
    #     coords[positive_mask, 1] *= radius
    #     weights[positive_mask] *= sqrt(radius) * sqrt(sqrt(scale))
        
    #     # Negative region scaling
    #     cdef np.ndarray negative_mask = ~positive_mask
    #     cdef np.ndarray norm = np.sqrt(dist2[negative_mask])
    #     cdef np.ndarray alpha = norm - 1
        
    #     coords[negative_mask, 0] *= radius * scale / norm
    #     coords[negative_mask, 1] *= radius / norm
        
    #     # Adjust coordinates
    #     cdef np.ndarray enorm = dist2[negative_mask]
    #     alpha[enorm < 1] /= np.sqrt(enorm[enorm < 1])
    #     alpha[enorm > 4] /= np.sqrt(enorm[enorm > 4])
    #     alpha[enorm > 4] *= 2
    #     coords[negative_mask] *= 1 + alpha[:, None]
        
    #     # Normalize weights
    #     weights /= np.sum(np.abs(weights))
        
    #     if rotate and (seg.theta != 0 or seg.psi != 0):
    #         # Add z coordinate and rotate
    #         coords_3d = np.column_stack((
    #             coords,
    #             np.full(coords.shape[0], z, dtype=np.float64)
    #         ))
    #         coords_3d = rotate_by_theta_psi_fast(coords_3d, seg.theta, seg.psi, None)
    #         coords = coords_3d
        
    #     return coords, dist2, weights

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double correlation_score(np.ndarray[DTYPE_t, ndim=1] image_intensities,
                             np.ndarray[DTYPE_t, ndim=1] filter_weights):
    """Compute correlation coefficient."""
    cdef:
        Py_ssize_t i, n = image_intensities.shape[0]
        double sum_xy = 0.0, sum_x = 0.0, sum_y = 0.0
        double sum_x2 = 0.0, sum_y2 = 0.0
        double x, y, mean_x, mean_y
        double[:] ii_view = image_intensities
        double[:] fw_view = filter_weights

    # Match NeuTube's darray_corrcoef_n(): ignore NaNs in sums, but keep
    # the full array length as the mean denominator.
    for i in range(n):
        if not isnan(ii_view[i]):
            sum_x += ii_view[i]
        if not isnan(fw_view[i]):
            sum_y += fw_view[i]
    mean_x = sum_x / n
    mean_y = sum_y / n

    # Calculate covariance and variances on valid pairs only.
    for i in range(n):
        if isnan(ii_view[i]) or isnan(fw_view[i]):
            continue
        x = ii_view[i] - mean_x
        y = fw_view[i] - mean_y
        sum_xy += x * y
        sum_x2 += x * x
        sum_y2 += y * y

    if sum_x2 == 0 or sum_y2 == 0:
        return 0.0

    return sum_xy / (sqrt(sum_x2) * sqrt(sum_y2))


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double dot_score(np.ndarray[DTYPE_t, ndim=1] image_intensities,
                       np.ndarray[DTYPE_t, ndim=1] filter_weights):
    """Compute NeuTube-style dot product, ignoring invalid samples."""
    cdef:
        Py_ssize_t i, n = image_intensities.shape[0]
        double score = 0.0
        double[:] ii_view = image_intensities
        double[:] fw_view = filter_weights

    for i in range(n):
        if isnan(ii_view[i]) or isnan(fw_view[i]):
            continue
        score += ii_view[i] * fw_view[i]

    return score

# cpdef double correlation_score(np.ndarray[DTYPE_t, ndim=1] image_intensities,
#                              np.ndarray[DTYPE_t, ndim=1] filter_weights):
#     """Compute correlation coefficient."""
#     cdef double std_int = np.std(image_intensities)
#     if std_int == 0:
#         return -1.0
#     return np.corrcoef(image_intensities, filter_weights)[0, 1]

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mean_intensity_score(np.ndarray[DTYPE_t, ndim=1] image_intensities,
                                 np.ndarray[DTYPE_t, ndim=1] filter_weights):
    """Compute mean intensity for positive weights."""
    cdef:
        Py_ssize_t i, n = filter_weights.shape[0]
        double total = 0.0
        int count = 0
        double[:] fw_view = filter_weights
        double[:] ii_view = image_intensities

    for i in range(n):
        if fw_view[i] > 0 and not isnan(ii_view[i]):
            total += ii_view[i]
            count += 1

    if count == 0:
        return 0.0
    return total / count
# cpdef double mean_intensity_score(np.ndarray[DTYPE_t, ndim=1] image_intensities,
#                                 np.ndarray[DTYPE_t, ndim=1] filter_weights):
#     """Compute mean intensity for positive weights."""
#     cdef np.ndarray mask = filter_weights > 0
#     if mask.sum() == 0:
#         return 0.0
#     return np.mean(image_intensities[mask])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double mean_intensity_neg_weight_score(np.ndarray[DTYPE_t, ndim=1] image_intensities,
                                 np.ndarray[DTYPE_t, ndim=1] filter_weights):
    """
    Compute mean intensity for negative weights.
    Generally used for outer fiber signal.
    """
    cdef:
        Py_ssize_t i, n = filter_weights.shape[0]
        double total = 0.0
        int count = 0
        double[:] fw_view = filter_weights
        double[:] ii_view = image_intensities

    for i in range(n):
        if fw_view[i] < 0:
            total += ii_view[i]
            count += 1

    if count == 0:
        return 0.0
    return total / count
