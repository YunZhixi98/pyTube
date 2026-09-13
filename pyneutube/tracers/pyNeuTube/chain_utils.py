
from typing import Tuple, Callable, Literal, Optional

import numpy as np

from pyneutube.core.processing.sampling import sample_voxels

from .filters import MexicanHatFilter
from .tracing import SegmentChain
from .config import Defaults, TraceDirection
from .geometry import segment_segment_distance


def get_inner_chain_range(chain: SegmentChain, chain_idx: int, ref_coord: np.ndarray, 
                          dist_threshold: float = Defaults.SEG_LENGTH*2.5) -> Tuple[int, int]:
    """ tz_locseg_chain.c: static void locseg_chain_point_range(...) """
    n_chain = len(chain)
    if n_chain == 0:
        return 0, -1

    chain_idx = max(0, min(chain_idx, n_chain - 1))

    start_idx = chain_idx
    accum_dist = float(np.linalg.norm(chain[chain_idx].start_coord - ref_coord))
    probe_idx = chain_idx

    while accum_dist < dist_threshold:
        probe_idx -= 1
        if probe_idx < 0:
            break
        accum_dist += float(np.linalg.norm(chain[probe_idx].start_coord - ref_coord))
        start_idx = probe_idx

    end_idx = chain_idx
    accum_dist = float(np.linalg.norm(chain[chain_idx].end_coord - ref_coord))
    probe_idx = chain_idx

    while accum_dist < dist_threshold:
        probe_idx += 1
        if probe_idx >= n_chain:
            break
        accum_dist += float(np.linalg.norm(chain[probe_idx].end_coord - ref_coord))
        end_idx = probe_idx

    return start_idx, end_idx


def get_chain_min_seg_score(chain: SegmentChain, signal_image: np.ndarray, 
                            score_func: Callable[[np.ndarray, np.ndarray], float]) -> float:
    min_score = np.inf
    seg_filter = MexicanHatFilter()
    for seg in chain:
        coords_3d, _, weights_3d = seg_filter(seg)
        intensities = sample_voxels(signal_image, coords_3d)
        score = score_func(intensities, weights_3d)
        min_score = min(min_score, score)

    return min_score


def get_chain_side_bright_point(chain: SegmentChain, signal_image: np.ndarray, 
                                side: Literal['head', 'tail']) -> np.ndarray:
    if side == 'head':
        seg = chain[0]
    else:
        seg = chain[-1]

    sample_count = max(1, int(np.floor((seg.length - 1.0) / 2.0)) + 1)
    offsets = np.arange(sample_count, dtype=np.float64).reshape(-1, 1)

    if side == 'head':
        coords = seg.start_coord + seg.dir_v * offsets
    else:
        coords = seg.end_coord - seg.dir_v * offsets

    intensities = sample_voxels(signal_image, coords)
    valid_indices = np.flatnonzero(~np.isnan(intensities))
    if valid_indices.size == 0:
        return coords[0]

    idx = valid_indices[np.argmax(intensities[valid_indices])]
    return coords[idx]


def interpolate_chain(chain: SegmentChain, ref_point: np.ndarray, ort: Optional[np.ndarray] = None) -> Tuple[int, np.ndarray]:
    """
    linear interpolation of chain by given coord
    """
    coords, coords_seg_indices = chain.to_coords(return_indices=True)
    coords_num = len(coords)

    if coords_num == 1:
        raise ValueError("Chain has only one segment.")        
    
    min_index = None
    min_dist = np.inf

    closest1 = None
    index = None
    interp_lambda = 0.0

    if ort is None:
        raise NotImplementedError
    else:
        start = ref_point - ort * 5.0
        end = ref_point + ort * 5.0

        for i in range(coords_num-1):
            start_pos = coords[i]
            end_pos = coords[i+1]

            tmp_min_dist, tmp_closest1, _ = segment_segment_distance(start_pos, end_pos, start, end)
            
            if tmp_min_dist < min_dist:
                min_dist = tmp_min_dist
                min_index = i
                closest1 = tmp_closest1
                segment = end_pos - start_pos
                segment_len2 = float(np.dot(segment, segment))
                if segment_len2 > 0.0:
                    interp_lambda = float(np.dot(closest1 - start_pos, segment) / segment_len2)
                else:
                    interp_lambda = 0.0

    if interp_lambda > 0.0 and interp_lambda < 1.0:
        start_pos = coords[min_index]
        end_pos = coords[min_index+1]

        length = np.linalg.norm(end_pos - start_pos)
        
        if length * interp_lambda < 1.0 and interp_lambda <= 0.5:
            interp_lambda = 0.0
        elif length * (1.0 - interp_lambda) < 1.0 and interp_lambda >= 0.5:
            interp_lambda = 1.0
        else:
            if min_index == 0:
                if length * interp_lambda < 3.0:
                    interp_lambda = 0.0
            elif min_index == coords_num-2:
                if length * (1.0 - interp_lambda) < 3.0:
                    interp_lambda = 1.0

    if interp_lambda > 0.0 and interp_lambda < 1.0:
        if closest1 is None:
            closest1 = start_pos + (end_pos - start_pos) * interp_lambda

        start_seg_idx = coords_seg_indices[min_index]
        end_seg_idx = coords_seg_indices[min_index+1]
        prev_seg = chain[start_seg_idx]

        if start_seg_idx == end_seg_idx:
            if prev_seg.length > 1.0:
                interp_seg = prev_seg.copy()
                if interp_seg.trace_direction == TraceDirection.BOTH:
                    interp_seg.trace_direction = TraceDirection.BACKWARD
                prev_seg.length = (prev_seg.length - 1) * (1-interp_lambda) + 1
                prev_seg._set_coordinate(prev_seg.end_coord, 'end')  # nonstandard operation for a private method
                
                interp_seg.length = (interp_seg.length - 1) * interp_lambda + 1
                interp_seg._set_coordinate(interp_seg.start_coord, 'start')

                chain.insert(start_seg_idx, interp_seg)
                index = start_seg_idx
        else:
            # chain._invalidate_label_bbox()
            r1 = chain[start_seg_idx].radius
            r2 = chain[end_seg_idx].radius
            interp_seg = prev_seg.copy()
            if interp_seg.trace_direction == TraceDirection.BOTH:
                interp_seg.trace_direction = TraceDirection.FORWARD
            interp_seg.length = 1.0
            interp_seg._set_coordinate(closest1, 'start')
            if r1 != r2:
                interp_seg.radius = r1 + (r2 - r1) * interp_lambda
            
            chain.insert(end_seg_idx, interp_seg)
            index = end_seg_idx
        
        interp_pos = interp_seg.start_coord

    else:
        if interp_lambda == 0.0:
            interp_pos = coords[min_index]
        else:
            interp_pos = coords[min_index+1]

    return index, interp_pos


            
            
                
