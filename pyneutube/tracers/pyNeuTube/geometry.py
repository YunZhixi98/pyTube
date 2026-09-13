

from math import floor

import numpy as np

from pyneutube.core.processing.transform import rotate_by_theta_psi

from .tracing_base import BaseTracingSegment


def point_in_seg(seg: BaseTracingSegment, coord: np.ndarray, is_local=False):
    if is_local:
        coord_local = coord
    else:  
        coord_local = coord - seg.start_coord
        coord_local = rotate_by_theta_psi(coord_local, seg.theta, seg.psi, inverse=True)
    x, y, z = coord_local

    flag = False
    if z>=-0.5 and z<=seg.length-0.5:
        d2 = (x/seg.scale)**2 + y**2
        if d2 <= seg.radius**2:
            flag = True

    return flag

def point_in_chain_index(coord: np.ndarray, chain) -> int:
    """
    please note returned `i` is index + 1
    """
    for i, seg in enumerate(chain):
        if point_in_seg(seg, coord):
            return i+1
    return 0


def point_to_segment_distance(coord, lineseg_start, lineseg_end):
    """
    Compute the minimum distance from a point to a line segment,
    and the closest point on the segment.

    Parameters
    ----------
    point : array-like, shape (2,) or (3,)
        The point coordinates.
    seg_a : array-like
        The first endpoint of the segment.
    seg_b : array-like
        The second endpoint of the segment.

    Returns
    -------
    distance : float
        Minimum distance from the point to the segment.
    closest_point : np.ndarray
        Coordinates of the closest point on the segment.
    """

    p = np.array(coord, dtype=float)
    a = np.array(lineseg_start, dtype=float)
    b = np.array(lineseg_end, dtype=float)

    ab = b - a
    ap = p - a

    # Project point onto the line, expressed as a fraction of |AB|
    denom = np.dot(ab, ab)
    if denom == 0:  # segment is a single point
        closest = a
    else:
        t = np.dot(ap, ab) / denom
        t = max(0, min(1, t))  # clamp to [0,1] → stays within segment
        closest = a + t * ab

    distance = np.linalg.norm(p - closest)
    return distance, closest


def point_to_seg_surface(coord: np.ndarray, seg2: BaseTracingSegment) -> float:
    """
    distance between point and segment surface
    """
    radius = seg2.radius
    scale = seg2.scale
    
    def _xy_ellipse_dist(x, y):
        nx = x / (radius * scale)
        ny = y / radius
        r = np.hypot(nx, ny)
        return r

    top_z = seg2.length - 1

    coord_local = coord - seg2.start_coord
    coord_local = rotate_by_theta_psi(coord_local, seg2.theta, seg2.psi, inverse=True)
    
    intersection_point = None

    if point_in_seg(seg2, coord_local, is_local=True):
        dist = 0.0
        intersection_point = coord
    else:
        x, y, z = coord_local
        norm_r = _xy_ellipse_dist(x, y)

        if z < 0.0:
            z_s = 0.0
        elif z > top_z:
            z_s = top_z
        else:
            z_s = z

        if norm_r <= 1.0:
            # radial inside ellipse projection; closest is vertical projection to cap if outside z
            intersection_point = np.array([x, y, z_s])
        else:
            # outside radially: project onto ellipse boundary
            x_s = x / norm_r
            y_s = y / norm_r
            intersection_point = np.array([x_s, y_s, z_s])

        dist = np.linalg.norm(coord_local - intersection_point)

        intersection_point = (
            rotate_by_theta_psi(intersection_point, seg2.theta, seg2.psi) + seg2.start_coord
        )

    return dist, intersection_point

    
def seg_to_seg_surface(
    seg1: BaseTracingSegment,
    seg2: BaseTracingSegment,
    discrete_step: int = 1,
) -> float:
    """
    distance between discrete central axis of segment_1 and segment_2 surface
    """    

    seg1_coords = (
        seg1.start_coord
        + np.arange(0, floor(seg1.length), discrete_step).reshape(-1, 1) * seg1.dir_v
    )
    
    min_dist = np.inf
    intersection_point = None

    for coord in seg1_coords:
        dist, tmp_point = point_to_seg_surface(coord, seg2)

        if dist < min_dist:
            min_dist = dist
            intersection_point = tmp_point
            if min_dist == 0:
                break
    
    return min_dist, intersection_point


def segment_segment_distance(P0: np.ndarray, P1: np.ndarray,
                            Q0: np.ndarray, Q1: np.ndarray,
                            eps: float = 1e-12) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Return (min_distance, closest_point_on_seg1, closest_point_on_seg2)
    for two 3D segments P0-P1 and Q0-Q1.
    """

    def point_to_segment_closest(
        P: np.ndarray, A: np.ndarray, B: np.ndarray
    ) -> tuple[float, np.ndarray]:
        """Return (distance, closest_point_on_segment) from point P to segment AB."""
        AB = B - A
        AB_len2 = np.dot(AB, AB)
        if AB_len2 == 0.0:
            # A == B degenerate segment
            cp = A.copy()
            return float(np.linalg.norm(P - cp)), cp
        t = np.dot(P - A, AB) / AB_len2
        t_clamped = np.clip(t, 0.0, 1.0)
        cp = A + t_clamped * AB
        return float(np.linalg.norm(P - cp)), cp

    # convert to numpy arrays
    P0 = np.asarray(P0, dtype=float)
    P1 = np.asarray(P1, dtype=float)
    Q0 = np.asarray(Q0, dtype=float)
    Q1 = np.asarray(Q1, dtype=float)

    u = P1 - P0  # direction of segment P
    v = Q1 - Q0  # direction of segment Q
    w = P0 - Q0

    a = np.dot(u, u)  # |u|^2
    b = np.dot(u, v)
    c = np.dot(v, v)  # |v|^2
    d = np.dot(u, w)
    e = np.dot(v, w)

    den = a * c - b * b  # denominator

    # First, try to get closest points on infinite lines
    if den > eps:
        s = (b * e - c * d) / den
        t = (a * e - b * d) / den
    else:
        # lines nearly parallel: choose s = 0 and solve for t
        s = 0.0
        t = (e / c) if c > eps else 0.0

    # if s and t are both within [0,1], the closest points lie within the segments
    if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
        cpP = P0 + s * u
        cpQ = Q0 + t * v
        return float(np.linalg.norm(cpP - cpQ)), cpP, cpQ

    # Otherwise, the true closest pair involves at least one endpoint.
    # Check all endpoint-to-segment distances (4 checks) and pick the minimum.
    candidates = []

    d_ps, cp_ps = point_to_segment_closest(P0, Q0, Q1)
    candidates.append((d_ps, P0.copy(), cp_ps))
    d_pe, cp_pe = point_to_segment_closest(P1, Q0, Q1)
    candidates.append((d_pe, P1.copy(), cp_pe))
    d_qs, cp_qs = point_to_segment_closest(Q0, P0, P1)
    candidates.append((d_qs, cp_qs, Q0.copy()))
    d_qe, cp_qe = point_to_segment_closest(Q1, P0, P1)
    candidates.append((d_qe, cp_qe, Q1.copy()))

    # choose smallest
    candidates.sort(key=lambda x: x[0])
    mindist, closestP, closestQ = candidates[0]
    return float(mindist), closestP, closestQ
    
def seg_to_seg_dist(seg1: BaseTracingSegment, seg2: BaseTracingSegment) -> float:

    mindist, _, _ = segment_segment_distance(
        seg1.start_coord, seg1.end_coord, seg2.start_coord, seg2.end_coord
    )

    return mindist


    
def seg_chain_dist_upper_bound(chain, seg: BaseTracingSegment) -> float:
    min_dist = np.inf
    seg_center_coord = seg.center_coord

    for tmpseg in chain:
        min_dist = min(min_dist, np.linalg.norm(tmpseg.center_coord - seg_center_coord))

    return min_dist


def point_to_chain_surface(point: np.ndarray, chain) -> tuple[float, np.ndarray, float]:
    min_dist = np.inf
    intersection_point = None
    min_idx = 0

    for i, seg in enumerate(chain):
        dist, tmp_point = point_to_seg_surface(point, seg)
        if dist < min_dist:
            min_dist = dist
            intersection_point = tmp_point
            min_idx = i
            if min_dist == 0:
                break

    return min_dist, intersection_point, min_idx


def closest_circle(circle_list, n, coord):
    if n == 1:
        return 0
    
    min_index = 0
    min_dist = np.inf

    for i in range(n):
        tmp_dist = np.sum((coord - circle_list[i].center)**2)

        if tmp_dist < min_dist:
            min_dist = tmp_dist
            min_index = i

    return min_index


try:
    from .geometry_accel import (
        point_to_seg_surface as _point_to_seg_surface_accel,
        seg_to_seg_surface as _seg_to_seg_surface_accel,
        segment_segment_distance as _segment_segment_distance_accel,
        seg_to_seg_dist as _seg_to_seg_dist_accel,
        seg_chain_dist_upper_bound as _seg_chain_dist_upper_bound_accel,
        point_to_chain_surface as _point_to_chain_surface_accel,
    )
except ImportError:
    pass
else:
    point_to_seg_surface = _point_to_seg_surface_accel
    seg_to_seg_surface = _seg_to_seg_surface_accel
    segment_segment_distance = _segment_segment_distance_accel
    seg_to_seg_dist = _seg_to_seg_dist_accel
    seg_chain_dist_upper_bound = _seg_chain_dist_upper_bound_accel
    point_to_chain_surface = _point_to_chain_surface_accel
