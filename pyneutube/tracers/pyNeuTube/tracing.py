# tracers/pyNeuTube/tracing.py

"""
tracing.py

Definition of tracing segments and utilities of tracing.
"""

from functools import lru_cache
from itertools import islice
from typing import Callable, List, Tuple, Literal, Union
from tqdm import tqdm
from math import exp, sqrt

import numpy as np
from scipy.optimize import minimize

from pyneutube.core.math_utils import get_bounding_box
from pyneutube.core.processing.sampling import sample_voxels
from pyneutube.core.processing.transform import rotate_by_theta_psi_fast

from .config import TraceStatus, Defaults, TraceDirection
from .filters import MexicanHatFilter, correlation_score, dot_score, mean_intensity_score
from .tracing_base import BaseTracingSegment
from .seg_utils import set_coordinates, set_orientation
from .geometry import point_in_seg
from .optimization import optimize_segment, optimize_segment_C
from .tracing_utils import label_tracing_mask
# from filters0 import correlation_score, mean_intensity_score


_SEG_FILTER = MexicanHatFilter()
_ORIENTATION_SEG_FILTER = MexicanHatFilter(max_dist2=0.81)
_HEMISPHERE_UNIFORM_POINTS = 200
_HEMISPHERE_REFINE_COARSE_POINTS = 96
_HEMISPHERE_REFINE_TOP_K = 6
_HEMISPHERE_REFINE_ANGLE = np.deg2rad(6.0)
# _MULTI_TRIAL_THETA_OFFSETS = (0.0, np.pi / 8, -np.pi / 8, np.pi / 16, -np.pi / 16)
# _MULTI_TRIAL_PSI_OFFSETS = (0.0, np.pi / 4, -np.pi / 4, np.pi / 8, -np.pi / 8)


# C original
@lru_cache(maxsize=16)
def _orientation_search_schedule(length: float = Defaults.SEG_LENGTH) -> tuple[tuple[float, tuple[float, ...]], ...]:
    schedule = []
    for theta in np.arange(0.1, np.pi * 0.75, 0.2):
        psi_step = 2.0 / length / np.sin(theta)
        psi_values = tuple(float(value) for value in np.arange(0, 2 * np.pi, psi_step))
        schedule.append((float(theta), psi_values))
    return tuple(schedule)


@lru_cache(maxsize=16)
def _hemisphere_uniform_orientation_schedule(
    num_points: int = _HEMISPHERE_UNIFORM_POINTS,
) -> tuple[tuple[float, tuple[float, ...]], ...]:
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    indices = np.arange(num_points)
    z = np.linspace(0.0, 1.0, num=num_points, endpoint=False) + 0.5 / num_points
    theta_values = np.arccos(z)
    psi_values = np.mod(indices * golden_angle, 2.0 * np.pi)
    return tuple(
        (float(theta), (float(psi),))
        for theta, psi in zip(theta_values, psi_values, strict=True)
    )


def _orientation_vector_to_theta_psi(vector: np.ndarray) -> tuple[float, float]:
    direction = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(direction)
    if norm == 0:
        raise ValueError("Orientation vector must be non-zero.")
    direction = direction / norm
    if direction[2] < 0.0:
        direction = -direction
    theta = float(np.arccos(np.clip(direction[2], -1.0, 1.0)))
    psi = float(np.arctan2(direction[0], -direction[1]) % (2 * np.pi))
    return theta, psi


@lru_cache(maxsize=1024)
def _vector_refined_orientation_candidates(
    theta: float,
    psi: float,
    angle_step: float = _HEMISPHERE_REFINE_ANGLE,
) -> tuple[tuple[float, float], ...]:
    center = np.asarray(_cached_orientation_vector(float(theta), float(psi)), dtype=np.float64)
    center /= np.linalg.norm(center)
    reference = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(center, reference))) > 0.95:
        reference = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    tangent_u = np.cross(center, reference)
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(center, tangent_u)

    candidates = []
    for offset_u in (-1.0, 0.0, 1.0):
        for offset_v in (-1.0, 0.0, 1.0):
            refined = center + angle_step * (offset_u * tangent_u + offset_v * tangent_v)
            candidates.append(_orientation_vector_to_theta_psi(refined))
    return tuple(candidates)


@lru_cache(maxsize=4096)
def _cached_orientation_vector(theta: float, psi: float) -> tuple[float, float, float]:
    direction = set_orientation(theta, psi)
    return (float(direction[0]), float(direction[1]), float(direction[2]))


class TracingSegment(BaseTracingSegment):
    def __init__(self, radius: float, coord: np.ndarray, length: float = Defaults.SEG_LENGTH, 
                 theta: float = 0.0, psi: float = 0.0, 
                 scale: float = 1.0, 
                 alignment: Literal['center', 'start', 'end'] = 'center',
                 direction = None):
        """
        Initialize a tracing segment.
        A segment is a cylinder in 3D space with a specified radius, length, and orientation.

        Parameters
        ----------
        radius : float
            Radius of the segment.
        coord : np.ndarray
            Position of the segment in 3D space as a numpy array of shape (3,).
        length : float
            Length of the segment. Defaults to 11.
        theta : float
            Rotation angle along x-axis (counter-clockwise) of the segment in radians. Defaults to 0.0. Range: [0, 蟺].
        psi : float
            Rotation angle along z-axis (counter-clockwise) of the segment in radians. Defaults to 0.0. Range: [0, 2蟺].
        scale : float
            Scale factor to control the cross-sectional shape of the segment. Defaults to 1.0.
            If scale=1.0, the cross-section is a circle. If scale != 1.0, the cross-section is an ellipse,
            where x = y * scale.
        alignment : str: 'center' | 'start' | 'end'
            Alignment of the segment with respect to the position.
        """
        self.radius = radius
        self.length = length
        # self.theta, self.psi = normalize_euler_zx(theta, psi)
        self.theta, self.psi = theta%(2*np.pi), psi%(2*np.pi)
        self.scale = scale

        self.start_coord = None  # start position
        self.center_coord = None
        self.end_coord = None
        self.dir_v = None  # direction vector (unit vector used for rotation)

        self.trace_direction = direction

        self._set_orientation()
        self._set_coordinate(coord, alignment=alignment)

        self.score = -1
        self.mean_intensity = None

        # for NeuronReconstruction
        self.ball_radius = None

    def _set_coordinate(self, coord: np.ndarray, alignment: Literal['center', 'start', 'end']) -> None:
        """
        Set the **start** position of the segment in 3D space.
        
        Parameters
        ----------
        coord : np.ndarray
            Position of the segment in 3D space as a numpy array of shape (3,).
        alignment : str: 'center' | 'start' | 'end'
            Alignment of the segment with respect to the position.
            Defaults to 'center'.
        """
        coord = np.asarray(coord, dtype=np.float64)
        self.start_coord, self.center_coord, self.end_coord = set_coordinates(coord, self.dir_v, self.length, alignment)
        
    def _set_orientation(self) -> None:
        """
        Update the direction vector of the segment in 3D space.
        """
        self.dir_v = np.asarray(
            _cached_orientation_vector(float(self.theta), float(self.psi)),
            dtype=np.float64,
        )

        return
    
    def _set_ball_radius(self) -> float:
        """
        compute ball radius for NeuronReconstruction primary distance filter
        """
        ball_radius = sqrt((self.radius * max(self.scale, 1.0))**2 + (self.length)**2) / 2 # note this is self.length not self.length/2
        self.ball_radius = ball_radius

        return ball_radius
    
    def copy(self) -> "TracingSegment":
        new_seg = self.__class__(
            radius=self.radius,
            coord=self.start_coord,
            length=self.length,
            theta=self.theta,
            psi=self.psi,
            scale=self.scale,
            alignment='start'
        )

        new_seg.trace_direction = self.trace_direction

        new_seg.score = self.score
        new_seg.mean_intensity = self.mean_intensity
        
        return new_seg


    def flip_segment(self) -> None:
        """
        Flip the segment direction.
        """
        self.dir_v = -self.dir_v
        self._set_coordinate(self.end_coord, alignment='start')
        # self.theta, self.psi = normalize_euler_zx(np.pi - self.theta, self.psi + np.pi)
        self.theta, self.psi = (np.pi - self.theta) % (2*np.pi), (self.psi + np.pi) % (2*np.pi)
        # self.theta += np.pi  # C source code

        return
    
    def centroid_shift(self, image: np.ndarray) -> None:
        """
        Perform mean shift using filtered value of image intensity
        """
        coords_3d, _, _ = _SEG_FILTER(self)
        intensities = sample_voxels(image, coords_3d)
        valid_mask = np.isfinite(intensities)
        if np.any(valid_mask):
            valid_coords = coords_3d[valid_mask]
            valid_intensities = intensities[valid_mask]
            if np.sum(valid_intensities) != 0:
                centroid = np.average(valid_coords, weights=valid_intensities, axis=0)
                self._set_coordinate(centroid, 'center')

        return
    
    def orientation_grid_search(self, image: np.ndarray) -> None:
        """
        Brute-force search initial orientation of the TracingSegment.
        """
        best_score = -np.inf
        best_theta, best_psi = self.theta, self.psi
        center_coord = self.center_coord.copy()
        half_length = (self.length - 1.0) * 0.5

        # For a fixed radius/scale/length, the orientation filter weights are
        # invariant and only the local coordinates need rotation per candidate.
        base_seg = self.copy()
        base_seg.theta = 0.0
        base_seg.psi = 0.0
        base_seg._set_orientation()
        base_coords_3d, _, weights_3d = _ORIENTATION_SEG_FILTER(base_seg, rel_pos="local")

        search_mode = getattr(Defaults, "ORIENTATION_SEARCH_MODE", "grid")
        if search_mode == "grid":
            for theta, psi_values in _orientation_search_schedule(self.length):
                for psi in psi_values:
                    dir_v = np.asarray(
                        _cached_orientation_vector(float(theta), float(psi)),
                        dtype=np.float64,
                    )
                    start_coord = center_coord - half_length * dir_v
                    coords_3d = rotate_by_theta_psi_fast(base_coords_3d, theta, psi, None)
                    coords_3d += start_coord
                    intensities = sample_voxels(image, coords_3d)
                    score = correlation_score(intensities, weights_3d)

                    if score > best_score:
                        best_score = score
                        best_theta, best_psi = theta, psi
        elif search_mode in {"hemisphere_uniform", "hemisphere_uniform_refine"}:
            def score_candidate(theta: float, psi: float) -> float:
                dir_v = np.asarray(
                    _cached_orientation_vector(float(theta), float(psi)),
                    dtype=np.float64,
                )
                start_coord = center_coord - half_length * dir_v
                coords_3d = rotate_by_theta_psi_fast(base_coords_3d, theta, psi, None)
                coords_3d += start_coord
                intensities = sample_voxels(image, coords_3d)
                return correlation_score(intensities, weights_3d)

            def update_best(theta: float, psi: float) -> float:
                nonlocal best_score, best_theta, best_psi
                score = score_candidate(theta, psi)
                if score > best_score:
                    best_score = score
                    best_theta, best_psi = theta, psi
                return score

            coarse_scores = []
            if search_mode == "hemisphere_uniform":
                schedule = _hemisphere_uniform_orientation_schedule()
            else:
                schedule = _hemisphere_uniform_orientation_schedule(_HEMISPHERE_REFINE_COARSE_POINTS)
            for theta, psi_values in schedule:
                psi = psi_values[0]
                score = update_best(theta, psi)
                coarse_scores.append((score, theta, psi))

            if search_mode == "hemisphere_uniform_refine":
                coarse_scores.sort(reverse=True, key=lambda item: item[0])
                seen = {
                    (round(theta, 12), round(psi_values[0], 12))
                    for theta, psi_values in schedule
                }
                for _score, theta, psi in coarse_scores[:_HEMISPHERE_REFINE_TOP_K]:
                    for refined_theta, refined_psi in _vector_refined_orientation_candidates(theta, psi):
                        key = (round(refined_theta, 12), round(refined_psi, 12))
                        if key in seen:
                            continue
                        seen.add(key)
                        update_best(refined_theta, refined_psi)
        else:
            raise ValueError(f"Unknown orientation search mode: {search_mode!r}")

        self.theta, self.psi = best_theta, best_psi
        self._set_orientation()
        self._set_coordinate(center_coord, 'center')

        return

    def length_search(self, image: np.ndarray) -> None:
        """
        Brute-force search length of the TracingSegment on the side.
        """
        for length in np.linspace(self.length, 0, num=round(self.length) + 1, endpoint=True):
            coords_2d, _, weights_2d = _SEG_FILTER(self, is_2d=True, z=float(length))
            coords_2d = np.asarray(coords_2d)
            weights_2d = np.asarray(weights_2d)
            coords_2d += self.start_coord
            intensities = sample_voxels(image, coords_2d)
            score = correlation_score(intensities, weights_2d)
            if score > 0.5:
                self.length = length
                break
        self._set_coordinate(self.start_coord, 'start')
        return

    def fit_segment(self, image: np.ndarray,
                    score_func: Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]] = dot_score,
                    ) -> bool:

        max_x = [self.radius, self.theta, self.psi, self.scale]

        max_params = optimize_segment_C(self, image, score_func, var_init=max_x)

        if max_params.success:
            self.radius, self.theta, self.psi, self.scale = max_params.x
        self.score, self.mean_intensity = self.score_segment(image, [correlation_score, mean_intensity_score])

        self._set_orientation()
        self._set_coordinate(self.start_coord, 'start')

        return max_params.success

    def score_segment(self, image: np.ndarray, 
                      score_func: Union[List[Callable[[np.ndarray, np.ndarray], Union[np.ndarray, float]]],
                                        Callable[[np.ndarray, np.ndarray], 
                                        Union[np.ndarray, float]]] = correlation_score) -> Union[float, List[float]]:
        """
        """
        coords_3d, _, weights_3d = _SEG_FILTER(self)
        intensities = sample_voxels(image, coords_3d)
        scores = []
        if isinstance(score_func, list):
            for func in score_func:
                scores.append(func(intensities, weights_3d))
            return scores
        else:
            score = score_func(intensities, weights_3d)
            return score
    
    def get_norm_min_score(self, min_score: float) -> float:
        """

        """
        norm_min_score = min_score * (1.0 + 1.0 / (2.0 + exp(4.0 - self.radius * sqrt(self.scale))))

        return norm_min_score

    
    def _dilate_segment(self, ratio: float = 1.5, diff: float = 0.0, max_diff: float = 3.0) -> None:
        """
        Dilate the segment's radius and adjust its scale factor.

        Parameters
        ----------
        ratio : float
            Multiplicative factor to apply to the radius
        diff : float
            Additive factor to apply to the radius
        max_diff : float
            Maximum allowed additive change to the radius

        Notes
        -----
        The dilation affects both the y-radius (radius) and x-radius (radius * scale).
        The scale factor is updated to maintain the elliptical proportions.
        """

        rby = self.radius
        rbx = self.radius * self.scale
        nrby = rby * ratio + diff
        nrbx = rbx * ratio + diff

        if max_diff > 0:
            nrbx = min(nrbx, rbx + max_diff)
            nrby = min(nrby, rby + max_diff)

        self.scale = nrbx / nrby
        self.radius = nrby

        return
    
    def __repr__(self):
        return f"TracingSegment(radius={self.radius:.3f}, coord={self.start_coord}, length={self.length:.3f}, theta={self.theta:.3f}, psi={self.psi:.3f}, scale={self.scale:.3f})"

    
class SegmentChain:
    """
    A chain of connected TracingSegment objects (e.g. one branch of a neuron).
    Enforces that each segment's start == the previous segment's end.
    """
    def __init__(
        self,
        segments: Union[List[TracingSegment], TracingSegment, None] = None,
    ):
        """
        Parameters
        ----------
        segments : list[TracingSegment] | TracingSegment
        """
        self._segments: List[TracingSegment] = []

        if isinstance(segments, TracingSegment):
            self._segments = [segments]
        elif isinstance(segments, list):
            self._segments = list(segments)

        # control the tracing process of the head and tail of the chain
        self._trace_status = [TraceStatus.NORMAL, TraceStatus.NORMAL]
        self._trace_step = Defaults.TRACE_STEP
        self._stop_seg_trace_score = Defaults.STOP_SEG_TRACE_SCORE
        self._max_seg_num = Defaults.MAX_SEG_NUM
        self._blocked_by_init_hit = False

        self.mean_intensity = None
        self.mean_score = None
        self.label_bboxes = None

    def _invalidate_label_bbox(self) -> None:
        self.label_bboxes = None

    def get_label_bbox(
        self,
        start: int | None = None,
    end: int | None = None,
    ) -> np.ndarray | None:
        if self.label_bboxes is None or len(self) == 0:
            return None

        if start is None:
            start = 0
        if end is None:
            end = len(self) - 1
        if start > end:
            return None

        selected = self.label_bboxes[start:end + 1]
        valid = selected[:, 0] >= 0
        if not np.any(valid):
            return None

        selected = selected[valid]
        return np.array(
            [
                selected[:, 0].min(),
                selected[:, 1].max(),
                selected[:, 2].min(),
                selected[:, 3].max(),
                selected[:, 4].min(),
                selected[:, 5].max(),
            ],
            dtype=np.intp,
        )

    def _remove_segment(self, idx: int) -> None:
        self._segments.pop(idx)
        self._invalidate_label_bbox()

    def append(self, segment: TracingSegment):
        """
        Append a segment to the end of the chain.
        """
        if not isinstance(segment, TracingSegment):
            raise TypeError("Can only add TracingSegment objects")
        self._segments.append(segment)
        self._invalidate_label_bbox()

    def insert(self, idx: int, segment: TracingSegment):
        """
        Insert a segment at position `idx` in the chain and re-validate connectivity.
        """
        self._segments.insert(idx, segment)
        self._invalidate_label_bbox()

    @property
    def segments(self) -> List[TracingSegment]:
        """Get a copy of the internal list of segments."""
        return list(self._segments)

    def to_coords(self, orientation: Literal['xyz', 'zyx'] = 'xyz', return_indices: bool = False) -> np.ndarray:
        """
        Return an array of 3D coordinates representing the startpoints of the segment chain.
        (including the endpoint of the last segment)

        Parameters
        ----------
        orientation : 'xyz' or 'zyx'
            Axis order for output coordinates.

        Returns
        -------
        np.ndarray
            Array of shape (N+1, 3), where N is the number of segments. `N+1` includes the tip node.
        """
        n = len(self)
        if n == 0:
            if return_indices:
                return [], []
            else:
                return []

        if orientation not in ('xyz', 'zyx'):
            raise ValueError("Orientation must be 'xyz' or 'zyx'.")


        # coords = np.empty((n + 1, 3), dtype=np.float64)
        # for i, seg in enumerate(self._segments):
        #     c = seg.start_coord
        #     coords[i] = c
        # last_seg = self._segments[-1]
        # end = last_seg.start_coord + last_seg.length * last_seg.dir_v
        # coords[-1] = end

        coords = []
        coords_seg_indices = []
        if n == 1:
            coords = np.array([self._segments[0].start_coord,
                               self._segments[0].end_coord], dtype=np.float64)
            coords_seg_indices = np.array([0, 0], dtype=int)
            
        else:
            pass_forward = True if self._segments[0].trace_direction == TraceDirection.BACKWARD else False
            for i, seg in enumerate(self._segments):
                if seg.trace_direction == TraceDirection.BACKWARD:
                    if i==0:
                        coords.append(seg.start_coord)
                        coords_seg_indices.append(i)
                    coords.append(seg.end_coord)
                    coords_seg_indices.append(i)
                elif seg.trace_direction == TraceDirection.FORWARD:
                    if pass_forward and i!=n-1:
                        pass_forward = False
                        continue
                    coords.append(seg.start_coord)
                    coords_seg_indices.append(i)
                    if i==n-1:
                        coords.append(seg.end_coord)
                        coords_seg_indices.append(i)
                else:
                    if i==0:
                        coords.append(seg.start_coord)
                        coords_seg_indices.append(i)

            coords = np.array(coords, dtype=np.float64)
            coords_seg_indices = np.array(coords_seg_indices, dtype=int)

        if orientation == 'zyx':
            if len(coords) != 0:
                coords = coords[:, ::-1]

        if return_indices:
            return coords, coords_seg_indices
        
        return coords
    
    
    @property
    def path_length(self) -> float:
        coords = self.to_coords()
        if len(coords) <= 1:
            return 0
        return np.sum(np.linalg.norm(coords[1:] - coords[:-1], axis=1))

    def _check_chain_status(self, trace_mask: np.ndarray, 
                            side: Literal['head', 'tail', 'both'],
                            ) -> None:
        """
        check segment trace status on one sides from:
            1) voxel traced or not
            2) out of image bound
            3) seg score (need delete this segment)
            4) new seg forms loop
            5) new seg intensity extremely decreases (inspired by Sujun)
        """

        def _check_side(side_idx: int) -> None:
    
            if side_idx == 0:
                seg_idx = 0
                seg = self._segments[seg_idx]
                coord = seg.start_coord[::-1]
                coord_round = np.round(coord).astype(int)
            else:
                seg_idx = -1
                seg = self._segments[seg_idx]
                coord = seg.end_coord[::-1]
                coord_round = np.round(coord).astype(int)
            
            if len(self) >= 2:
                if np.any(coord < 0) or np.any((coord+1) > trace_mask.shape):
                    self._trace_status[side_idx] = TraceStatus.OUT_OF_BOUND
                elif trace_mask[coord_round[0], coord_round[1], coord_round[2]] == 1:
                    self._trace_status[side_idx] = TraceStatus.HIT_MARK
            
            if side!='both':  # 'both' is somehow an initial check for seed's seg.
                if seg.score < seg.get_norm_min_score(self._stop_seg_trace_score):
                    self._trace_status[side_idx] = TraceStatus.LOW_SCORE
                    self._remove_segment(seg_idx)  # delete this segment
                    # print(f'pop seg_idx={seg_idx} on side={side}, score={seg.score}')
                    return
                
                radius = seg.radius * sqrt(seg.scale)

                if radius > 25:
                    self._trace_status[side_idx] = TraceStatus.SEG_TOO_THICK
                    self._remove_segment(seg_idx)
                    return

                if radius < Defaults.MIN_SEG_RADIUS:
                    self._trace_status[side_idx] = TraceStatus.SEG_TOO_THIN
                    self._remove_segment(seg_idx)
                    return

                if len(self) >= 2:
                    # 
                    continuous_seg_idx = 1 if side_idx == 0 else -2
                    cur_seg = self._segments[seg_idx]
                    seg_prev = self._segments[continuous_seg_idx]
                    intensity_change = cur_seg.mean_intensity / seg_prev.mean_intensity
                    if intensity_change < 0.5:
                        self._trace_status[side_idx] = TraceStatus.SIGNAL_CHANGED
                        self._remove_segment(seg_idx)
                        # print(f'pop seg_idx={seg_idx} on side={side}', 'status=signal_changed')

                        return
                    
                    r1 = seg_prev.radius * sqrt(seg_prev.scale)
                    r2 = cur_seg.radius * sqrt(cur_seg.scale)
                    if r2 > 1.0:
                        ratio = r1 / r2
                        if ratio > 2.0 or ratio < 0.5:
                            self._trace_status[side_idx] = TraceStatus.RADIUS_CHANGED
                            self._remove_segment(seg_idx)
                            
                            return 

                    #
                    loop_flag = False
                    endpoint_flag = "start" if side_idx == 0 else "end"
                    opposite_endpoint_flag = "end" if side_idx == 0 else "start"
                    for tmpseg in self._segments[seg_idx+1:len(self)+seg_idx][::1 if side_idx==0 else -1]:
                        if tmpseg is seg_prev:
                            continue

                        loop_flag = test_seg_overlap(tmpseg, seg, seg2_coord_flag=endpoint_flag)
                        if loop_flag:
                            break
                        if test_seg_overlap(tmpseg, seg, seg2_coord_flag='center'):
                            if not test_seg_overlap(tmpseg, seg, seg2_coord_flag=opposite_endpoint_flag):
                                loop_flag = True
                                break
                            elif test_seg_turn(tmpseg, seg, max_angle=np.pi/2):
                                loop_flag = True
                                break

                    if loop_flag:
                        self._trace_status[side_idx] = TraceStatus.LOOP_FORMED
                        self._remove_segment(seg_idx)
                        
            return

        if side in ('tail', 'both'):
            _check_side(1)
        if side in ('head', 'both'):
            _check_side(0)
            
        return
    
    def _check_chain_init_status(self, trace_mask: np.ndarray) -> None:
        def _check_side(side_idx: int) -> None:
    
            if side_idx == 0:
                seg_idx = 0
                pos_ratio = 1/3
                seg = self._segments[seg_idx]
                coord_4_hit = np.round(seg.start_coord + seg.dir_v * seg.length * pos_ratio).astype(int)  # coord for hitting traced voxel detection
            else:
                seg_idx = -1
                pos_ratio = 2/3
                seg = self._segments[seg_idx]
                coord_4_hit = np.round(seg.start_coord + seg.dir_v * seg.length * pos_ratio).astype(int)

            coord_zyx = coord_4_hit[::-1]
            if np.any(coord_zyx < 0) or np.any(coord_zyx >= trace_mask.shape):
                self._trace_status[side_idx] = TraceStatus.OUT_OF_BOUND
            elif trace_mask[coord_zyx[0], coord_zyx[1], coord_zyx[2]] == 1:
                self._trace_status[side_idx] = TraceStatus.HIT_MARK

            return
        
        _check_side(1)
        _check_side(0)
        self._blocked_by_init_hit = (
            self._trace_status[0] == TraceStatus.HIT_MARK
            and self._trace_status[1] == TraceStatus.HIT_MARK
        )
            
        return


    def _init_next_seg(self, side: Literal['head', 'tail']) -> TracingSegment:
        side_idx = 0 if side=='head' else -1
        side_seg = self._segments[side_idx]
        new_seg = side_seg.copy()
        if side=='head':
            new_seg.flip_segment()
            new_seg.trace_direction = TraceDirection.BACKWARD
        else:
            new_seg.trace_direction = TraceDirection.FORWARD
        # map to [0, 2*PI] or [0, PI] for angles
        new_seg.psi = new_seg.psi % (2 * np.pi)
        new_seg.theta = new_seg.theta % np.pi
        new_coord = new_seg.start_coord + self._trace_step * new_seg.dir_v * (new_seg.length - 1.0)
        new_seg._set_coordinate(new_coord, 'start')
        
        return new_seg

    def _remove_overlap_sides(self) -> None:
        if len(self) >= 2:
            if test_seg_overlap(self._segments[1], self._segments[0], 'sides'):
                self._remove_segment(0)
        
        if len(self) >= 2:
            if test_seg_overlap(self._segments[-2], self._segments[-1], 'sides'):
                self._remove_segment(-1)

    def _remove_turn_sides(self) -> None:
        if len(self) >= 2:
            if test_seg_turn_2(self._segments[1], self._segments[0], max_angle=1.0):
                self._remove_segment(0)

        if len(self) >= 2:
            if test_seg_turn_2(self._segments[-2], self._segments[-1], max_angle=1.0):
                self._remove_segment(-1)

    def _refresh_endpoint_scores(self, signal_image: np.ndarray) -> None:
        if len(self) <= 1:
            return

        endpoint_indices = [0, -1]
        for idx in endpoint_indices:
            seg = self._segments[idx]
            seg.score, seg.mean_intensity = seg.score_segment(
                signal_image,
                [correlation_score, mean_intensity_score],
            )

    def generate_chain_trace(self, signal_image: np.ndarray, trace_mask: np.ndarray) -> None:
        """
        """
        self._check_chain_init_status(trace_mask)
        # forward trace
        while self._trace_status[1] == TraceStatus.NORMAL and len(self) < self._max_seg_num:
            new_seg = self._init_next_seg(side='tail')
            success = new_seg.fit_segment(signal_image)
            # print(len(self), self[-1],'\n',new_seg,'\n', self._trace_status[1])

            if success:
                self.append(new_seg)
            else:
                break

            self._check_chain_status(trace_mask, 'tail')


        # backward trace
        if self._trace_status[0] == TraceStatus.NORMAL:
            while self._trace_status[0] == TraceStatus.NORMAL and len(self) < self._max_seg_num:
                new_seg = self._init_next_seg(side='head')
                success = new_seg.fit_segment(signal_image)
                new_seg.flip_segment()
                # print(len(self), self[0],'\n',new_seg,'\n', self._trace_status[0])

                if success:
                    self.insert(0, new_seg)
                else:
                    break

                self._check_chain_status(trace_mask, 'head')
        
        
        if len(self) >= 2:
            if self._trace_status[1] != TraceStatus.HIT_MARK:
                self._segments[-1].length_search(signal_image)
            
            if self._trace_status[0] != TraceStatus.HIT_MARK:
                self._segments[0].flip_segment()
                self._segments[0].length_search(signal_image)
                self._segments[0].flip_segment()

        self._remove_overlap_sides()
        self._remove_turn_sides()
        # Endpoint geometry can change after tracing; refresh endpoint stats so
        # downstream chain screening uses up-to-date score/intensity values.
        self._refresh_endpoint_scores(signal_image)

        return
        
    def __len__(self):
        return len(self._segments)

    def __getitem__(self, idx):
        return self._segments[idx]

    def __iter__(self):
        return iter(self._segments)

    def __repr__(self):
        return f"<SegmentChain: {len(self)} segments>"
    

class SegmentChains:
    def __init__(
        self,
        chains: Union[List[SegmentChain], SegmentChain, None] = None,
        image_shape: Tuple[int, int, int] = (0, 0, 0),
    ):
        """
        Parameters
        ----------
        chains : list[SegmentChain] | SegmentChain
        """
        self._chains: List[SegmentChain] = []

        if isinstance(chains, SegmentChain):
            self._chains = [chains]
        elif isinstance(chains, list):
            self._chains = list(chains)

        self.trace_mask = np.zeros(image_shape, dtype=np.uint8)

        self._min_chain_score = Defaults.MIN_CHAIN_SCORE
        self._min_chain_length = Defaults.MIN_CHAIN_LENGTH

    def generate_neuron_trace(
        self,
        seeds,
        signal_image: np.ndarray,
        *,
        max_seeds: int | None = None,
        verbose: int = 1,
        check_timeout=None,
        progress_callback=None,
    ):
        seed_iterable = islice(seeds, max_seeds) if max_seeds is not None else seeds
        total = min(len(seeds), max_seeds) if max_seeds is not None else len(seeds)
        if progress_callback is not None:
            progress_callback("generate_neuron_trace", 0, total)
        for seed_idx, seed in enumerate(
            tqdm(seed_iterable, desc="Generating chains", disable=verbose < 1)
        ):
            if check_timeout is not None and seed_idx % 8 == 0:
                check_timeout("chain generation")
            chain = SegmentChain(seed.seg.copy())
            chain.generate_chain_trace(signal_image, self.trace_mask)
            keep_chain = (
                not chain._blocked_by_init_hit and (
                    chain.path_length >= self._min_chain_length
                    or chain._trace_status[0] == TraceStatus.HIT_MARK
                    or chain._trace_status[1] == TraceStatus.HIT_MARK
                )
            )
            if keep_chain:
                label_tracing_mask(chain, self.trace_mask, dilate=True)
                self.append(chain)
            if progress_callback is not None:
                progress_callback("generate_neuron_trace", seed_idx + 1, total)
        if verbose:
            print(f"Number of chains: {len(self)}")

    def generate_neuron_trace_lazy_seed_scoring(
        self,
        seeds,
        signal_image: np.ndarray,
        *,
        max_seeds: int | None = None,
        verbose: int = 1,
        check_timeout=None,
        progress_callback=None,
    ) -> list:
        seed_iterable = islice(seeds, max_seeds) if max_seeds is not None else seeds
        total = min(len(seeds), max_seeds) if max_seeds is not None else len(seeds)
        scored_seeds = []
        skipped_by_trace_mask = 0
        rejected_by_seed_filter = 0
        if progress_callback is not None:
            progress_callback("generate_neuron_trace", 0, total)
        for seed_idx, seed in enumerate(
            tqdm(seed_iterable, desc="Generating chains", disable=verbose < 1)
        ):
            if check_timeout is not None and seed_idx % 8 == 0:
                check_timeout("lazy seed scoring")
            if self.trace_mask[tuple(seed.coord[::-1])]:
                skipped_by_trace_mask += 1
                if progress_callback is not None:
                    progress_callback("generate_neuron_trace", seed_idx + 1, total)
                continue

            seed.score_seed(signal_image)
            if not (
                seed.score > seed.seg.get_norm_min_score(Defaults.MIN_SEED_SCORE)
                and seed.seg.radius <= Defaults.MAX_SEED_RADIUS
            ):
                rejected_by_seed_filter += 1
                if progress_callback is not None:
                    progress_callback("generate_neuron_trace", seed_idx + 1, total)
                continue

            scored_seeds.append(seed)
            chain = SegmentChain(seed.seg.copy())
            chain.generate_chain_trace(signal_image, self.trace_mask)
            keep_chain = (
                not chain._blocked_by_init_hit and (
                    chain.path_length >= self._min_chain_length
                    or chain._trace_status[0] == TraceStatus.HIT_MARK
                    or chain._trace_status[1] == TraceStatus.HIT_MARK
                )
            )
            if keep_chain:
                label_tracing_mask(chain, self.trace_mask, dilate=True)
                self.append(chain)
            if progress_callback is not None:
                progress_callback("generate_neuron_trace", seed_idx + 1, total)
        self.lazy_seed_scored_count = len(scored_seeds)
        self.lazy_seed_skipped_by_trace_mask = skipped_by_trace_mask
        self.lazy_seed_rejected_by_seed_filter = rejected_by_seed_filter
        if verbose:
            print(f"Number of chains: {len(self)}")
            print(
                "Lazy seed scoring: "
                f"scored={len(scored_seeds)}, "
                f"skipped_by_trace_mask={skipped_by_trace_mask}, "
                f"rejected_by_seed_filter={rejected_by_seed_filter}"
            )
        return scored_seeds

    def filter_chains(self, *, verbose: int = 1):
        if len(self) > 100:
            # mean_intensities = []
            mean_scores = []
            min_intensity = np.inf
            for chain in self:
                intensities = [seg.mean_intensity for seg in chain]
                scores = [seg.score for seg in chain]
                chain.mean_intensity = np.mean(intensities)
                chain.mean_score = np.mean(scores)
                if (chain.mean_score >= self._min_chain_score) and (chain.mean_intensity < min_intensity):
                    min_intensity = chain.mean_intensity

                # mean_intensities.append(chain.mean_intensity)
                mean_scores.append(chain.mean_score)

            # min_intensity = min(mean_intensities) if mean_intensities else np.inf
            self._chains = [
                chain for chain, score in zip(self._chains, mean_scores)
                if (score >= self._min_chain_score) or (chain.mean_intensity >= min_intensity)
            ]
            if verbose:
                print(f"Number of chains after filtering: {len(self)}")

        return

    def append(self, chain: SegmentChain):
        if not isinstance(chain, SegmentChain):
            raise TypeError("Can only add SegmentChain objects")
        self._chains.append(chain)

    def __len__(self):
        return len(self._chains)

    def __getitem__(self, idx):
        return self._chains[idx]

    def __iter__(self):
        return iter(self._chains)

    def __repr__(self):
        return f"<SegmentChains: {len(self)} chains>"


def test_seg_overlap(seg1: TracingSegment, seg2: TracingSegment, seg2_coord_flag: Literal['start', 'center', 'end', 'sides']) -> bool:
    """
    test if seg1 is overlapped with seg2
    """
    if seg2_coord_flag == 'start':
        seg2_coords = (seg2.start_coord,)
    elif seg2_coord_flag == 'center':
        seg2_coords = (seg2.center_coord,)
    elif seg2_coord_flag == 'end':
        seg2_coords = (seg2.end_coord,)
    elif seg2_coord_flag == 'sides':
        seg2_coords = (seg2.start_coord, seg2.end_coord)

    return all(point_in_seg(seg1, coord) for coord in seg2_coords)


def test_seg_turn(seg1: TracingSegment, seg2: TracingSegment, max_angle: float = 1) -> bool:
    """
    test if the angle between seg1 and seg2 is less than max_angle
    """
    seg1_dir_v = seg1.dir_v
    seg2_dir_v = seg2.dir_v
    # angle = np.arccos(np.dot(seg1_dir_v, seg2_dir_v) / (np.linalg.norm(seg1_dir_v) * np.linalg.norm(seg2_dir_v)))
    angle = np.arccos(np.clip(np.dot(seg1_dir_v, seg2_dir_v), -1.0, 1.0))

    return angle > max_angle

def test_seg_turn_2(seg1: TracingSegment, seg2: TracingSegment, max_angle: float) -> bool:
    """
    test if the angle between seg1 and seg2 is less than max_angle
    """
    seg1_dir_v = seg1.dir_v
    seg2_dir_v = seg2.dir_v

    cross = np.cross(seg1_dir_v, seg2_dir_v)
    dot = np.dot(seg1_dir_v, seg2_dir_v)
    angle = np.arctan2(np.linalg.norm(cross), dot)

    if angle > np.pi:
        angle = 2 * np.pi - angle

    return angle > max_angle
