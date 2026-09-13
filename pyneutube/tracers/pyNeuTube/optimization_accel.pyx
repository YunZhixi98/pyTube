# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport cython
cimport numpy as cnp

from libc.math cimport sqrt
from scipy.optimize import OptimizeResult

from pyneutube.core.processing.sampling import sample_voxels

from .config import Defaults, Optimization
from .filters import MexicanHatFilter, dot_score


ctypedef cnp.float64_t DTYPE_t

_OPTIMIZATION_SEG_FILTER = MexicanHatFilter()


@cython.cdivision(True)
cdef inline double _vec_norm4(double[:] vec) noexcept nogil:
    return sqrt(
        vec[0] * vec[0]
        + vec[1] * vec[1]
        + vec[2] * vec[2]
        + vec[3] * vec[3]
    )


cdef inline double _dot4(double[:] lhs, double[:] rhs) noexcept nogil:
    return (
        lhs[0] * rhs[0]
        + lhs[1] * rhs[1]
        + lhs[2] * rhs[2]
        + lhs[3] * rhs[3]
    )


cdef class SegmentOptimizer:
    cdef object image
    cdef object score_func
    cdef int maxiter
    cdef double min_gradient
    cdef double min_direction
    cdef double alpha0
    cdef double ro
    cdef double c1
    cdef bint verbose
    cdef object seg_filter
    cdef double stop_grad
    cdef object low
    cdef object high
    cdef object delta
    cdef object weight
    cdef object _x_buf
    cdef object _g0_buf
    cdef object _g1_buf
    cdef object _gradient_buf
    cdef object _direction_buf
    cdef object _trial_buf
    cdef object _org_x_buf
    cdef object _scaled_dir_buf
    cdef double _last_gradient_base_score

    def __init__(
        self,
        image,
        *,
        score_func=dot_score,
        maxiter=Optimization.MAX_ITER,
        min_gradient=1e-3,
        min_direction=1e-3,
        alpha0=0.2,
        ro=0.8,
        c1=0.01,
        weight=None,
        verbose=False,
    ):
        cdef:
            double norm
            double[:] delta_view

        self.image = np.asarray(image)
        self.score_func = score_func
        self.maxiter = int(maxiter)
        self.min_gradient = float(min_gradient)
        self.min_direction = float(min_direction)
        self.alpha0 = float(alpha0)
        self.ro = float(ro)
        self.c1 = float(c1)
        self.verbose = bool(verbose)

        self.low = np.array(
            [Defaults.MIN_SEG_RADIUS, -np.inf, -np.inf, 0.2],
            dtype=np.float64,
        )
        self.high = np.array(
            [50.0, np.inf, np.inf, 20.0],
            dtype=np.float64,
        )
        self.delta = np.array(
            [
                Optimization.DELTA_RADIUS,
                Optimization.DELTA_THETA,
                Optimization.DELTA_PSI,
                Optimization.DELTA_SCALE,
            ],
            dtype=np.float64,
        )
        if weight is None:
            delta_view = self.delta
            norm = _vec_norm4(delta_view)
            self.weight = np.array(
                [
                    delta_view[0] / norm,
                    delta_view[1] / norm,
                    delta_view[2] / norm,
                    delta_view[3] / norm,
                ],
                dtype=np.float64,
            )
        else:
            self.weight = np.asarray(weight, dtype=np.float64)

        self.seg_filter = _OPTIMIZATION_SEG_FILTER
        self.stop_grad = Optimization.LINE_SEARCH_STOP_GRADIENT
        self._x_buf = np.empty(4, dtype=np.float64)
        self._g0_buf = np.empty(4, dtype=np.float64)
        self._g1_buf = np.empty(4, dtype=np.float64)
        self._gradient_buf = np.empty(4, dtype=np.float64)
        self._direction_buf = np.empty(4, dtype=np.float64)
        self._trial_buf = np.empty(4, dtype=np.float64)
        self._org_x_buf = np.empty(4, dtype=np.float64)
        self._scaled_dir_buf = np.empty(4, dtype=np.float64)
        self._last_gradient_base_score = 0.0

    @staticmethod
    cdef void _apply_x_to_seg(object seg, double[:] x) except *:
        seg.radius = x[0]
        seg.theta = x[1]
        seg.psi = x[2]
        seg.scale = x[3]

    cdef double _score_at(self, double[:] x_local, object tmpseg) except *:
        SegmentOptimizer._apply_x_to_seg(tmpseg, x_local)
        coords3d, _, weights = self.seg_filter(tmpseg)
        intensities = sample_voxels(self.image, coords3d)
        return float(self.score_func(intensities, weights))

    @staticmethod
    cdef void _copy4(double[:] src, double[:] dst) noexcept:
        dst[0] = src[0]
        dst[1] = src[1]
        dst[2] = src[2]
        dst[3] = src[3]

    @staticmethod
    cdef void _conjugate_update_direction(
        double[:] gradient,
        double[:] prev_gradient,
        double[:] direction,
    ) noexcept:
        cdef double grad_delta0 = gradient[0] - prev_gradient[0]
        cdef double grad_delta1 = gradient[1] - prev_gradient[1]
        cdef double grad_delta2 = gradient[2] - prev_gradient[2]
        cdef double grad_delta3 = gradient[3] - prev_gradient[3]
        cdef double denom = _dot4(prev_gradient, prev_gradient)
        cdef double beta = 0.0

        if denom > 0.0:
            beta = (
                gradient[0] * grad_delta0
                + gradient[1] * grad_delta1
                + gradient[2] * grad_delta2
                + gradient[3] * grad_delta3
            ) / denom
            if beta < 0.0:
                beta = 0.0

        direction[0] = direction[0] * beta + gradient[0]
        direction[1] = direction[1] * beta + gradient[1]
        direction[2] = direction[2] * beta + gradient[2]
        direction[3] = direction[3] * beta + gradient[3]

    cdef void _compute_fd_gradient_into(
        self,
        double[:] x_view,
        object tmpseg,
        double[:] gradient_view,
    ) except *:
        cdef:
            double[:] delta_view = self.delta
            double base_score = self._score_at(x_view, tmpseg)
            double right_score
            double left_score
            double step
            double grad_i
            int i

        self._last_gradient_base_score = base_score

        for i in range(4):
            step = delta_view[i]
            x_view[i] += step
            right_score = self._score_at(x_view, tmpseg)
            x_view[i] -= step

            grad_i = (right_score - base_score) / step
            if grad_i < 0.0:
                x_view[i] -= step
                left_score = self._score_at(x_view, tmpseg)
                if left_score < base_score:
                    grad_i = 0.0
                else:
                    grad_i = (base_score - left_score) / step
                x_view[i] += step
            elif grad_i > 0.0:
                x_view[i] -= step
                left_score = self._score_at(x_view, tmpseg)
                if left_score > base_score:
                    grad_i = (base_score - left_score) / step
                x_view[i] += step
            else:
                x_view[i] -= step
                left_score = self._score_at(x_view, tmpseg)
                grad_i = (base_score - left_score) / step
                x_view[i] += step

            gradient_view[i] = grad_i

    cpdef cnp.ndarray compute_fd_gradient(self, cnp.ndarray[DTYPE_t, ndim=1] x, object tmpseg):
        cdef:
            double[:] x_view = x
            cnp.ndarray[DTYPE_t, ndim=1] gradient = self._gradient_buf
            double[:] gradient_view = gradient

        self._compute_fd_gradient_into(x_view, tmpseg, gradient_view)

        return np.array(gradient, copy=True)

    cdef tuple _line_search_var_backtrack_impl(
        self,
        double[:] x_org_view,
        double[:] direction_view,
        double start_score,
        double[:] start_grad_view,
        object tmpseg,
    ) except *:
        cdef:
            double[:] low_view = self.low
            double[:] high_view = self.high
            double[:] weight_view = self.weight
            cnp.ndarray[DTYPE_t, ndim=1] x_trial = self._trial_buf
            cnp.ndarray[DTYPE_t, ndim=1] org_x = self._org_x_buf
            cnp.ndarray[DTYPE_t, ndim=1] scaled_dir = self._scaled_dir_buf
            double[:] org_x_view = org_x
            double[:] x_trial_view = x_trial
            double[:] scaled_dir_view = scaled_dir
            double dir_len
            double alpha
            double gd_dot
            double gd_dot_c1
            double score_trial
            double wolfe1
            int i

        SegmentOptimizer._copy4(x_org_view, org_x_view)

        for i in range(4):
            direction_view[i] *= weight_view[i]

        dir_len = _vec_norm4(direction_view)
        if dir_len <= self.min_direction:
            return False, start_score

        alpha = self.alpha0 / dir_len
        gd_dot = _dot4(start_grad_view, direction_view)
        gd_dot_c1 = gd_dot * self.c1

        while True:
            if alpha * self.ro * dir_len < self.stop_grad:
                SegmentOptimizer._apply_x_to_seg(tmpseg, org_x_view)
                return False, start_score

            for i in range(4):
                scaled_dir_view[i] = direction_view[i] * alpha
                x_trial_view[i] = org_x_view[i] + scaled_dir_view[i]
                if x_trial_view[i] < low_view[i]:
                    x_trial_view[i] = low_view[i]
                elif x_trial_view[i] > high_view[i]:
                    x_trial_view[i] = high_view[i]

            score_trial = self._score_at(x_trial_view, tmpseg)

            alpha *= self.ro
            wolfe1 = alpha / self.ro * gd_dot_c1
            if wolfe1 < 0.0:
                wolfe1 = 0.0
            if score_trial >= start_score + wolfe1:
                return True, score_trial

    cpdef tuple line_search_var_backtrack(
        self,
        cnp.ndarray[DTYPE_t, ndim=1] x_org,
        cnp.ndarray[DTYPE_t, ndim=1] direction,
        double start_score,
        cnp.ndarray[DTYPE_t, ndim=1] start_grad,
        object tmpseg,
    ):
        cdef:
            double[:] x_org_view = x_org
            double[:] direction_view = direction
            double[:] start_grad_view = start_grad
            bint improved
            double score_trial

        improved, score_trial = self._line_search_var_backtrack_impl(
            x_org_view,
            direction_view,
            start_score,
            start_grad_view,
            tmpseg,
        )
        if improved:
            return True, np.array(self._trial_buf, copy=True), score_trial
        return False, np.array(self._org_x_buf, copy=True), start_score

    cpdef void fit(self, object seg):
        cdef:
            cnp.ndarray[DTYPE_t, ndim=1] x = self._x_buf
            cnp.ndarray[DTYPE_t, ndim=1] g0 = self._g0_buf
            cnp.ndarray[DTYPE_t, ndim=1] g1 = self._g1_buf
            cnp.ndarray[DTYPE_t, ndim=1] update_direction = self._direction_buf
            double[:] update_direction_view
            double[:] g0_view
            double[:] g1_view
            double[:] x_view = x
            double final_score
            double dir_len
            double gnorm
            double new_score
            int iter_count = 0
            bint improved
            object tmpseg = seg.copy()

        x_view[0] = seg.radius
        x_view[1] = seg.theta
        x_view[2] = seg.psi
        x_view[3] = seg.scale

        self._compute_fd_gradient_into(x_view, tmpseg, g0)
        SegmentOptimizer._copy4(g0, update_direction)
        final_score = self._last_gradient_base_score

        while True:
            improved = False
            update_direction_view = update_direction
            dir_len = _vec_norm4(update_direction_view)
            if dir_len >= self.min_direction:
                improved, new_score = self._line_search_var_backtrack_impl(
                    x_view,
                    update_direction_view,
                    final_score,
                    g0,
                    tmpseg,
                )
                if improved:
                    SegmentOptimizer._copy4(self._trial_buf, x_view)
                    final_score = new_score
                    if self.verbose:
                        print(f"iter {iter_count + 1}: score {final_score:.6g}")

            if not improved:
                g0_view = g0
                gnorm = _vec_norm4(g0_view)
                if gnorm > self.min_gradient:
                    SegmentOptimizer._copy4(g0_view, update_direction)
                    update_direction_view = update_direction
                    improved, new_score = self._line_search_var_backtrack_impl(
                        x_view,
                        update_direction_view,
                        final_score,
                        g0,
                        tmpseg,
                    )
                    if improved:
                        SegmentOptimizer._copy4(self._trial_buf, x_view)
                        final_score = new_score
                        if self.verbose:
                            print(f"iter {iter_count + 1} (fallback): score {final_score:.6g}")

                if not improved:
                    break

            iter_count += 1
            if iter_count >= self.maxiter:
                break

            self._compute_fd_gradient_into(x_view, tmpseg, g1)
            g1_view = g1
            g0_view = g0
            update_direction_view = update_direction
            SegmentOptimizer._conjugate_update_direction(g1_view, g0_view, update_direction_view)
            SegmentOptimizer._copy4(g1_view, g0_view)

        SegmentOptimizer._apply_x_to_seg(seg, x_view)
        seg.theta = seg.theta % (2 * np.pi)
        seg.psi = seg.psi % (2 * np.pi)


def optimize_segment_C(seg, image, score_func=dot_score, var_init=None):
    """Fit radius, orientation, and scale using the C-style optimizer port."""
    if var_init is not None:
        seg.radius, seg.theta, seg.psi, seg.scale = var_init

    optimizer = SegmentOptimizer(
        image,
        score_func=score_func,
        maxiter=Optimization.MAX_ITER,
    )
    optimizer.fit(seg)
    return OptimizeResult(
        success=True,
        x=np.array([seg.radius, seg.theta, seg.psi, seg.scale], dtype=np.float64),
    )
