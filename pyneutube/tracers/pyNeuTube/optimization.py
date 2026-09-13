from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy.optimize import OptimizeResult, minimize

from pyneutube.core.processing.sampling import sample_voxels

from .config import Defaults, Optimization
from .filters import MexicanHatFilter, correlation_score, dot_score
from .tracing_base import BaseTracingSegment

try:
    from .optimization_accel import SegmentOptimizer as _CythonSegmentOptimizer
    from .optimization_accel import optimize_segment_C as _optimize_segment_C_accel
except ImportError:
    _CythonSegmentOptimizer = None
    _optimize_segment_C_accel = None

_OPTIMIZATION_SEG_FILTER = MexicanHatFilter()


def optimize_segment(
    seg: BaseTracingSegment,
    image: np.ndarray,
    score_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]] = dot_score,
    var_init=None,
):
    """Fit radius, orientation, and scale for a tracing segment."""
    if var_init is None:
        var_init = [seg.radius, seg.theta, seg.psi, seg.scale]

    var_bound = np.array(
        [
            (Defaults.MIN_SEG_RADIUS, 50.0),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (0.2, 20.0),
        ]
    )

    seg_filter = _OPTIMIZATION_SEG_FILTER
    tmpseg = seg.copy()
    # init_dir = set_orientation(seg.theta, seg.psi)

    def _score_func(x: np.ndarray) -> float:
        tmpseg.radius, tmpseg.theta, tmpseg.psi, tmpseg.scale = x
        # new_dir = set_orientation(tmpseg.theta, tmpseg.psi)
        # dot_product = np.dot(init_dir, new_dir)
        # penalty = 1000.0 * abs(dot_product) if dot_product < 0 else 0.0  
        penalty = 0 # temporarily disable the penalty

        coords_3d, _, weights_3d = seg_filter(tmpseg)
        intensities = sample_voxels(image, coords_3d)
        score = score_func(intensities, weights_3d)

        return -score + penalty

    return minimize(
        fun=_score_func,
        x0=var_init,
        method="L-BFGS-B",
        jac="3-point",
        bounds=var_bound,
        tol=1e-3,
        options={"maxiter": Optimization.MAX_ITER},
    )



def optimize_segment_C(
    seg: BaseTracingSegment,
    image: np.ndarray,
    score_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]] = dot_score,
    var_init=None,
):
    """Fit radius, orientation, and scale using the C-style optimizer port."""
    if _optimize_segment_C_accel is not None:
        return _optimize_segment_C_accel(
            seg,
            image,
            score_func=score_func,
            var_init=var_init,
        )

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


class SegmentOptimizer:
    """Port the four-parameter NeuTube perceptor optimizer used by tracing."""

    def __init__(
        self,
        image: np.ndarray,
        *,
        score_func: Callable[[np.ndarray, np.ndarray], float] = dot_score,
        maxiter: int = Optimization.MAX_ITER,
        min_gradient: float = 1e-3,
        min_direction: float = 1e-3,
        alpha0: float = 0.2,
        ro: float = 0.8,
        c1: float = 0.01,
        weight: Optional[np.ndarray] = None,
        verbose: bool = False,
    ):
        self.image = np.asarray(image)
        self.score_func = score_func
        self.maxiter = int(maxiter)
        self.min_gradient = float(min_gradient)
        self.min_direction = float(min_direction)
        self.alpha0 = float(alpha0)
        self.ro = float(ro)
        self.c1 = float(c1)
        self.verbose = bool(verbose)

        self.var_bound = np.array(
            [
                (Defaults.MIN_SEG_RADIUS, 50.0),
                (-np.inf, np.inf),
                (-np.inf, np.inf),
                (0.2, 20.0),
            ],
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
            self.weight = self.delta / np.linalg.norm(self.delta)
        else:
            self.weight = np.asarray(weight, dtype=np.float64)
        self.seg_filter = _OPTIMIZATION_SEG_FILTER
        self.stop_grad = Optimization.LINE_SEARCH_STOP_GRADIENT
        self._last_gradient_base_score = 0.0

    @staticmethod
    def _apply_x_to_seg(seg: BaseTracingSegment, x: np.ndarray) -> None:
        seg.radius = float(x[0])
        seg.theta = float(x[1])
        seg.psi = float(x[2])
        seg.scale = float(x[3])

    def _score_at(self, x_local: np.ndarray, tmpseg: BaseTracingSegment) -> float:
        self._apply_x_to_seg(tmpseg, x_local)
        coords3d, _, weights = self.seg_filter(tmpseg)
        intensities = sample_voxels(self.image, coords3d)
        return float(self.score_func(intensities, weights))

    @staticmethod
    def _conjugate_update_direction(
        gradient: np.ndarray,
        prev_gradient: np.ndarray,
        direction: np.ndarray,
    ) -> None:
        grad_delta = gradient - prev_gradient
        denom = float(np.dot(prev_gradient, prev_gradient))
        beta = 0.0 if denom <= 0.0 else float(np.dot(gradient, grad_delta) / denom)
        if beta < 0.0:
            beta = 0.0

        direction *= beta
        direction += gradient

    def compute_fd_gradient(
        self,
        x: np.ndarray,
        tmpseg: BaseTracingSegment,
        score_at: Optional[Callable[[np.ndarray, BaseTracingSegment], float]] = None,
    ) -> np.ndarray:
        if score_at is None:
            score_at = self._score_at

        n = int(x.size)
        gradient = np.zeros_like(x, dtype=np.float64)
        base_score = score_at(x, tmpseg)
        self._last_gradient_base_score = float(base_score)

        for i in range(n):
            step = self.delta[i]
            x[i] += step
            right_score = score_at(x, tmpseg)
            x[i] -= step

            grad_i = (right_score - base_score) / step
            if grad_i < 0.0:
                x[i] -= step
                left_score = score_at(x, tmpseg)
                if left_score < base_score:
                    grad_i = 0.0
                else:
                    grad_i = (base_score - left_score) / step
                x[i] += step
            elif grad_i > 0.0:
                x[i] -= step
                left_score = score_at(x, tmpseg)
                if left_score > base_score:
                    grad_i = (base_score - left_score) / step
                x[i] += step
            else:
                x[i] -= step
                left_score = score_at(x, tmpseg)
                grad_i = (base_score - left_score) / step
                x[i] += step

            gradient[i] = grad_i

        return gradient

    def line_search_var_backtrack(
        self,
        x_org: np.ndarray,
        direction: np.ndarray,
        start_score: float,
        start_grad: np.ndarray,
        tmpseg: BaseTracingSegment,
        *,
        score_at: Optional[Callable[[np.ndarray, BaseTracingSegment], float]] = None,
    ) -> Tuple[bool, np.ndarray, float]:
        if score_at is None:
            score_at = self._score_at

        if self.weight is not None:
            direction *= self.weight

        dir_len = float(np.linalg.norm(direction))
        if dir_len <= self.min_direction:
            return False, x_org.copy(), start_score

        alpha = self.alpha0 / dir_len
        gd_dot = float(np.dot(start_grad, direction))
        gd_dot_c1 = gd_dot * self.c1
        low = self.var_bound[:, 0]
        high = self.var_bound[:, 1]

        org_x = x_org.copy()
        x_trial = np.empty_like(org_x)
        scaled_dir = np.empty_like(direction)

        while True:
            if alpha * self.ro * dir_len < self.stop_grad:
                self._apply_x_to_seg(tmpseg, org_x)
                return False, org_x.copy(), start_score

            np.multiply(direction, alpha, out=scaled_dir)
            np.add(org_x, scaled_dir, out=x_trial)
            np.clip(x_trial, low, high, out=x_trial)
            score_trial = score_at(x_trial, tmpseg)

            alpha *= self.ro
            wolfe1 = max(alpha / self.ro * gd_dot_c1, 0.0)
            if score_trial >= start_score + wolfe1:
                return True, x_trial.copy(), score_trial

    def fit(self, seg: BaseTracingSegment) -> None:
        x = np.array([seg.radius, seg.theta, seg.psi, seg.scale], dtype=np.float64)

        tmpseg = seg.copy()
        score_at = self._score_at
        g0 = self.compute_fd_gradient(x, tmpseg, score_at)
        update_direction = g0.copy()
        final_score = self._last_gradient_base_score
        iter_count = 0

        while True:
            improved = False
            dir_len = float(np.linalg.norm(update_direction))
            if dir_len >= self.min_direction:
                improved, x_new, new_score = self.line_search_var_backtrack(
                    x,
                    update_direction,
                    final_score,
                    g0,
                    tmpseg,
                    score_at=score_at,
                )
                if improved:
                    x = x_new
                    final_score = new_score
                    if self.verbose:
                        print(f"iter {iter_count + 1}: score {final_score:.6g}")

            if not improved:
                gnorm = float(np.linalg.norm(g0))
                if gnorm > self.min_gradient:
                    update_direction = g0.copy()
                    improved, x_new, new_score = self.line_search_var_backtrack(
                        x,
                        update_direction,
                        final_score,
                        g0,
                        tmpseg,
                        score_at=score_at,
                    )
                    if improved:
                        x = x_new
                        final_score = new_score
                        if self.verbose:
                            print(f"iter {iter_count + 1} (fallback): score {final_score:.6g}")

                if not improved:
                    break

            iter_count += 1
            if iter_count >= self.maxiter:
                break

            g1 = self.compute_fd_gradient(x, tmpseg, score_at)
            self._conjugate_update_direction(g1, g0, update_direction)
            g0 = g1.copy()

        self._apply_x_to_seg(seg, x)
        seg.theta = seg.theta % (2 * np.pi)
        seg.psi = seg.psi % (2 * np.pi)


PythonSegmentOptimizer = SegmentOptimizer


if _CythonSegmentOptimizer is not None:
    SegmentOptimizer = _CythonSegmentOptimizer
