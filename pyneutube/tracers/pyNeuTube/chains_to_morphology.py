# tracers/pyNeuTube/chains_to_morphology.py

"""
chains_to_morphology.py

Conversion of SegmentChains to a unified morphology (digitalized neuron tree).
"""

from math import exp
from typing import List

import numpy as np

from pyneutube.core.io.swc_parser import Neuron
from pyneutube.core.processing.swc_utils import (
    remove_zigzag, remove_spur, tune_branch, 
    merge_close_point, remove_overshoot,
    remove_subtrees_by_length, optimal_downsample
    )

from .geometry import (
    seg_to_seg_surface, seg_to_seg_dist,
    seg_chain_dist_upper_bound,
    point_to_chain_surface,
    point_in_chain_index, closest_circle
    )
from .config import ConnectorType, Defaults
from .stack_graph_utils import Graph
from .stack_graph import StackGraph, graph_edge_index, graph_update_edge_table, graph_expand_edge_table, GraphWorkspace
from .neuron_structures import Neurocomp_Conn, Circle
from .tracing import SegmentChain, SegmentChains
from .tracing_utils import label_tracing_mask
from .chain_utils import get_chain_side_bright_point, get_inner_chain_range, interpolate_chain


_CANDIDATE_FALLBACK_RATIO = 0.5


def postprocess_reconstruction(neuron: Neuron, *, verbose: int = 1, check_timeout=None) -> Neuron:
    import time

    def _vprint(message: str) -> None:
        if verbose:
            print(message)

    t0 = time.time()
    if check_timeout is not None:
        check_timeout("remove_zigzag")
    remove_zigzag(neuron)
    _vprint(f'--> remove_zigzag [length={neuron.length}]: {time.time() - t0:.5f}')

    t0 = time.time()
    if check_timeout is not None:
        check_timeout("tune_branch")
    tune_branch(neuron)
    _vprint(f'--> tune_branch [length={neuron.length}]: {time.time() - t0:.5f}')

    t0 = time.time()
    if check_timeout is not None:
        check_timeout("remove_spur")
    remove_spur(neuron)
    _vprint(f'--> remove_spur [length={neuron.length}]: {time.time() - t0:.5f}')

    t0 = time.time()
    if check_timeout is not None:
        check_timeout("merge_close_point")
    merge_close_point(neuron, 0.01)
    _vprint(f'--> merge close point [length={neuron.length}]: {time.time() - t0:.5f}')

    t0 = time.time()
    if check_timeout is not None:
        check_timeout("remove_overshoot")
    remove_overshoot(neuron)
    _vprint(f'--> remove overshoot [length={neuron.length}]: {time.time() - t0:.5f}')

    t0 = time.time()
    if check_timeout is not None:
        check_timeout("optimal_downsample")
    optimal_downsample(neuron)
    _vprint(f'--> downsample [length={neuron.length}]: {time.time() - t0:.5f}')

    t0 = time.time()
    if check_timeout is not None:
        check_timeout("remove_subtrees_by_length")
    remove_subtrees_by_length(neuron, verbose=verbose)
    _vprint(f'--> remove subtrees by length [length={neuron.length}]: {time.time() - t0:.5f}')

    return neuron


def _clone_neuron(neuron: Neuron) -> Neuron:
    return Neuron().initialize(np.asarray(neuron.swc, dtype=object).copy())


def _clone_connect_segment(seg):
    new_seg = seg.__class__.__new__(seg.__class__)
    new_seg.radius = seg.radius
    new_seg.length = seg.length
    new_seg.theta = seg.theta
    new_seg.psi = seg.psi
    new_seg.scale = seg.scale
    new_seg.start_coord = seg.start_coord.copy()
    new_seg.center_coord = seg.center_coord.copy()
    new_seg.end_coord = seg.end_coord.copy()
    new_seg.dir_v = seg.dir_v.copy()
    new_seg.trace_direction = seg.trace_direction
    new_seg.score = seg.score
    new_seg.mean_intensity = seg.mean_intensity
    new_seg.ball_radius = None
    return new_seg


def _bbox_distance(min1: np.ndarray, max1: np.ndarray, min2: np.ndarray, max2: np.ndarray) -> float:
    gap = np.maximum(0.0, np.maximum(min2 - max1, min1 - max2))
    return float(np.linalg.norm(gap))


def _connection_distance_limit(max_radius: float, dist_thresh: float) -> float:
    return float(2 * np.sqrt(max_radius**2 + ((Defaults.SEG_LENGTH - 1) / 2) ** 2) + dist_thresh)


def _chain_broadphase_stats(chains: SegmentChains):
    stats = []
    for chain in chains:
        if len(chain) == 0:
            stats.append(None)
            continue

        start_coords = np.asarray([seg.start_coord for seg in chain], dtype=np.float64)
        end_coords = np.asarray([seg.end_coord for seg in chain], dtype=np.float64)
        chain_coords = np.vstack((start_coords, end_coords))
        segment_bounds = [
            (
                np.minimum(start_coord, end_coord),
                np.maximum(start_coord, end_coord),
            )
            for start_coord, end_coord in zip(start_coords, end_coords)
        ]
        head_coords = np.vstack(
            (
                np.asarray(chain[0].start_coord, dtype=np.float64),
                np.asarray(chain[0].end_coord, dtype=np.float64),
            )
        )
        tail_coords = np.vstack(
            (
                np.asarray(chain[-1].start_coord, dtype=np.float64),
                np.asarray(chain[-1].end_coord, dtype=np.float64),
            )
        )
        stats.append(
            {
                "chain_min": chain_coords.min(axis=0),
                "chain_max": chain_coords.max(axis=0),
                "head_min": head_coords.min(axis=0),
                "head_max": head_coords.max(axis=0),
                "tail_min": tail_coords.min(axis=0),
                "tail_max": tail_coords.max(axis=0),
                "max_radius": max(float(seg.radius) for seg in chain),
                "segment_bounds": segment_bounds,
            }
        )

    return stats


def _can_skip_connect_test(chain1_stats, chain2_stats, dist_thresh: float) -> bool:
    # Broadphase safety (cylinders, isotropic geometry, no image term):
    # d_bbox <= min center distance used by connect_test's first reject check.
    # Therefore, if d_bbox > T then min_sdist > T also holds, so skipping here
    # cannot reject a pair that would pass the later geometric connection test.
    if chain1_stats is None or chain2_stats is None:
        return True

    min_bbox_dist = min(
        _bbox_distance(
            chain1_stats["head_min"],
            chain1_stats["head_max"],
            chain2_stats["chain_min"],
            chain2_stats["chain_max"],
        ),
        _bbox_distance(
            chain1_stats["tail_min"],
            chain1_stats["tail_max"],
            chain2_stats["chain_min"],
            chain2_stats["chain_max"],
        ),
    )
    max_radius = max(chain1_stats["max_radius"], chain2_stats["max_radius"])
    distance_limit = _connection_distance_limit(max_radius, dist_thresh)
    return min_bbox_dist > distance_limit


def _grid_index_bounds(
    min_corner: np.ndarray,
    max_corner: np.ndarray,
    cell_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    min_index = np.floor(min_corner / cell_size).astype(np.int32)
    max_index = np.floor(max_corner / cell_size).astype(np.int32)
    return min_index, max_index


def _iter_grid_keys(
    min_index: np.ndarray,
    max_index: np.ndarray,
):
    for ix in range(int(min_index[0]), int(max_index[0]) + 1):
        for iy in range(int(min_index[1]), int(max_index[1]) + 1):
            for iz in range(int(min_index[2]), int(max_index[2]) + 1):
                yield (ix, iy, iz)


def _build_segment_spatial_grid(
    broadphase_stats,
    cell_size: float,
) -> dict[tuple[int, int, int], list[int]]:
    spatial_grid: dict[tuple[int, int, int], list[int]] = {}
    for chain_idx, chain_stats in enumerate(broadphase_stats):
        if chain_stats is None:
            continue
        for segment_min, segment_max in chain_stats["segment_bounds"]:
            min_index, max_index = _grid_index_bounds(segment_min, segment_max, cell_size)
            for key in _iter_grid_keys(min_index, max_index):
                bucket = spatial_grid.setdefault(key, [])
                if not bucket or bucket[-1] != chain_idx:
                    bucket.append(chain_idx)
    return spatial_grid


def _candidate_indices_from_segment_grid(
    chain_stats,
    spatial_grid: dict[tuple[int, int, int], list[int]],
    distance_limit: float,
    cell_size: float,
    *,
    fallback_limit: int,
) -> set[int] | None:
    candidate_indices: set[int] = set()
    for box_min, box_max in (
        (chain_stats["head_min"], chain_stats["head_max"]),
        (chain_stats["tail_min"], chain_stats["tail_max"]),
    ):
        query_min_index, query_max_index = _grid_index_bounds(
            box_min - distance_limit,
            box_max + distance_limit,
            cell_size,
        )
        for key in _iter_grid_keys(query_min_index, query_max_index):
            candidate_indices.update(spatial_grid.get(key, ()))
            if len(candidate_indices) > fallback_limit:
                return None
    return candidate_indices


class ChainConnector:

    def __init__(
        self,
        verbose: int = 1,
        enable_crossover_test: bool = False,
    ):
        
        # 
        self.conn_list: List[Neurocomp_Conn] = []
        self.graph = Graph()

        self.gw = GraphWorkspace()

        # Connection_Test_Workspace
        # self.hook_spot = -1
        # self.dist = Defaults.SEG_LENGTH * 10.0
        self.dist_thresh = Defaults.SEG_LENGTH
        self.sp_test = True
        self.interpolate = True
        self.enable_crossover_test = bool(enable_crossover_test)

        self.verbose = verbose

    def _vprint(self, message: str) -> None:
        if self.verbose:
            print(message)

        # # Neurocomp_Conn
        # self.mode = ConnectorType.NEUROCOMP_CONN_HL
        # self.info = np.array([0, 0], dtype=np.uint8)
        # self.pos = np.empty(3, dtype=np.float64)  # point to surface min sdist intersection
        # self.ort = np.empty(3, dtype=np.float64)
        # self.cost = None
        # self.min_sdist = None  # distance between point and segment surface
        # self.min_pdist = None  # distance between two line segments

    def connect_test(
        self,
        chain1: SegmentChain,
        chain2: SegmentChain,
        conn: Neurocomp_Conn,
        *,
        chain2_max_radius: float | None = None,
    ):
        n_chain1 = len(chain1)
        n_chain2 = len(chain2)
        if n_chain1 == 0 or n_chain2 == 0:
            return False

        head = _clone_connect_segment(chain1[0])
        tail = _clone_connect_segment(chain1[-1])
        tail.flip_segment()
        if len(chain1) >= 1:
            head.length = 2.0
            tail.length = 2.0

        chain2_segments = chain2._segments
        min_center_dist = min(
            seg_chain_dist_upper_bound(chain2, head),
            seg_chain_dist_upper_bound(chain2, tail),
        )

        if chain2_max_radius is None:
            chain2_max_radius = max((seg.radius for seg in chain2_segments), default=0.0)
        max_radius = max(head.radius, tail.radius, chain2_max_radius)
        if min_center_dist > 2*np.sqrt(max_radius**2+((Defaults.SEG_LENGTH-1)/2)**2)+self.dist_thresh:
            conn.mode = ConnectorType.NEUROCOMP_CONN_NONE
            conn.cost = 10.0
            return False

        min_sdist = float("inf")
        conn.min_pdist = float("inf")
        head_ball_radius = head._set_ball_radius()
        tail_ball_radius = tail._set_ball_radius()
        head_center = head.center_coord
        tail_center = tail.center_coord
        norm = np.linalg.norm
        for i, seg in enumerate(chain2_segments):
            seg_ball_radius = seg.ball_radius
            if seg_ball_radius is None:
                seg_ball_radius = seg._set_ball_radius()

            seg_center = seg.center_coord
            if (norm(seg_center - head_center) - seg_ball_radius - head_ball_radius) < min_center_dist:
                surface_dist, tmp_intersection_p = seg_to_seg_surface(head, seg)
                update = False
                if surface_dist < min_sdist:
                    min_sdist = surface_dist
                    conn.min_pdist = seg_to_seg_dist(head, seg)
                    update = True
                elif surface_dist == min_sdist:
                    tmp_pdist = seg_to_seg_dist(head, seg)
                    if tmp_pdist < conn.min_pdist:
                        conn.min_pdist = tmp_pdist
                        update = True

                if update:
                    conn.info[0] = 0
                    conn.info[1] = i
                    conn.pos = tmp_intersection_p

            if (norm(seg_center - tail_center) - seg_ball_radius - tail_ball_radius) < min_center_dist:
                surface_dist, tmp_intersection_p = seg_to_seg_surface(tail, seg)
                update = False
                if surface_dist < min_sdist:
                    min_sdist = surface_dist
                    conn.min_pdist = seg_to_seg_dist(tail, seg)
                    update = True
                elif surface_dist == min_sdist:
                    tmp_pdist = seg_to_seg_dist(tail, seg)
                    if tmp_pdist < conn.min_pdist:
                        conn.min_pdist = tmp_pdist
                        update = True

                if update:
                    conn.info[0] = 1
                    conn.info[1] = i
                    conn.pos = tmp_intersection_p

        conn.min_sdist = min_sdist

        if conn.min_sdist>self.dist_thresh:
            conn.mode = ConnectorType.NEUROCOMP_CONN_NONE
            conn.cost = 10.0
            return False
        
        if conn.info[0] == 0:
            conn.ort = head.dir_v
        else:
            conn.ort = -tail.dir_v  # Previously, tail has been flipped

        return True

    
    def get_shortest_path(self, chain1: SegmentChain, chain2: SegmentChain, signal_image: np.ndarray, sgw: StackGraph) -> List[int]:
        """
        chain1: source chain
        chain2: target chain
        """        

        bright_point = get_chain_side_bright_point(chain1, signal_image, 'head')
        dist, intersection_point, min_idx = point_to_chain_surface(bright_point, chain2)
        chain1_seg = chain1[0]

        tmp_bright_point = get_chain_side_bright_point(chain1, signal_image, 'tail')
        tmpdist, tmp_intersection_point, tmp_min_idx = point_to_chain_surface(tmp_bright_point, chain2)

        if tmpdist < dist:
            bright_point = tmp_bright_point
            dist = tmpdist
            intersection_point = tmp_intersection_point
            min_idx = tmp_min_idx
            chain1_seg = chain1[-1]

        xyz_max = np.array(
            [signal_image.shape[2] - 1, signal_image.shape[1] - 1, signal_image.shape[0] - 1],
            dtype=np.float64,
        )
        if np.any(bright_point < 0) or np.any(bright_point > xyz_max):
            return []
        
        sgw.update_stack_graph_workspace_by_seg_chain(chain1_seg, chain2, signal_image)

        start_index, end_index = get_inner_chain_range(chain2, min_idx, intersection_point)

        return self.get_shortest_path_pt(bright_point, chain2, start_index, end_index, signal_image, sgw, chain1)

    def get_shortest_path_pt(self, pos, chain, start_index, end_index, signal_image, sgw: StackGraph, chain1) -> List[int]:

        seg_index = int((start_index + end_index) / 2)
        locseg = chain[seg_index]
        start_pos = np.rint(pos).astype(int)

        pos = locseg.center_coord

        end_pos = np.rint(pos).astype(int)

        if sgw.sp_option != 1:
            if sgw.group_mask is None:
                sgw.group_mask = np.zeros(signal_image.shape, dtype=np.uint8)
            else:
                sgw.group_mask.fill(0)
            label_tracing_mask(
                chain,
                sgw.group_mask,
                dilate=True,
                start=start_index,
                end=end_index,
            )
            bbox = chain.get_label_bbox(start_index, end_index)
            if bbox is None:
                sgw.set_range(int(start_pos[0]), int(start_pos[0]), int(start_pos[1]), int(start_pos[1]), int(start_pos[2]), int(start_pos[2]))
            else:
                sgw.set_range(int(start_pos[0]), int(bbox[0]), int(start_pos[1]), int(bbox[2]), int(start_pos[2]), int(bbox[4]))
                sgw.update_range(int(bbox[1]), int(bbox[3]), int(bbox[5]))
        else:
            sgw.set_range(int(start_pos[0]), int(end_pos[0]), int(start_pos[1]), int(end_pos[1]), int(start_pos[2]), int(end_pos[2]))
        
        sgw.expand_range(np.full(3, 10))
        sgw.validate_range(signal_image.shape[2], signal_image.shape[1], signal_image.shape[0])

        start_pos = np.clip(start_pos, [0, 0, 0], [signal_image.shape[2]-1, signal_image.shape[1]-1, signal_image.shape[0]-1])
        end_pos = np.clip(end_pos, [0, 0, 0], [signal_image.shape[2]-1, signal_image.shape[1]-1, signal_image.shape[0]-1])

        if np.sum(np.square(start_pos-end_pos)) < 1e+5:
            offset_path = sgw.stack_route(signal_image, start_pos, end_pos)
        else:
            self._vprint('too far')
            raise NotImplementedError  # tz_locseg_chain.c Locseg_Chain_Shortest_Path_Pt
        
        path = []
        nvoxels = signal_image.size

        for i in range(len(offset_path)):
            idx = offset_path[i]
            if idx>=0 and idx<nvoxels:
                path.append(idx)


        return path


    def validate_shortest_path(self, path: list, chain, signal_image: np.ndarray, sgw: StackGraph, conn: Neurocomp_Conn):

        path_length = len(path)
        if path_length == 0:
            conn.mode = ConnectorType.NEUROCOMP_CONN_NONE
            gdist = 0.0
            return gdist
        
        gdist = sgw.value  # (geodesic) shortest path distance (with weighted cost -- image intensity)

        dark_count = 0
        bright_count = 0
        hit_index = 0  # hit_index is: chain_seg_index + 1
        if path_length >= 5:
            
            for i in range(path_length):
                coord_zyx = np.array(np.unravel_index(path[i], signal_image.shape))
                coord_xyz = coord_zyx[::-1]
                if hit_index < 3:
                    if conn.info[0] == 0:
                        hit_index = point_in_chain_index(coord_xyz, chain)
                    else:
                        hit_index = point_in_chain_index(coord_xyz, chain[::-1])
                
                intensity = signal_image[coord_zyx[0], coord_zyx[1], coord_zyx[2]]
                if intensity == 0 or intensity < sgw.argv[3]-sgw.argv[4]:
                    dark_count+=1
                else:
                    bright_count+=1
            
            if (dark_count >= 2 and dark_count >= bright_count) or dark_count >= 5 or hit_index >= 3:
                conn.mode = ConnectorType.NEUROCOMP_CONN_NONE
            else:
                if dark_count+ bright_count >= 2:
                    prev_coord = np.array(np.unravel_index(path[path_length - 2], signal_image.shape))[::-1]
                    count = 0
                    conn.ort = np.zeros(3, dtype=np.float64)
                    for i in range(path_length - 3, -1, -1):
                        coord = np.array(np.unravel_index(path[i], signal_image.shape))[::-1]
                        conn.ort += prev_coord - coord
                        prev_coord = coord
                        count += 1
                        if count >= 5:
                            break
                    conn.ort = conn.ort / np.linalg.norm(conn.ort)


        return gdist

    def reconstruct(
        self,
        chains: SegmentChains,
        signal_image: np.ndarray,
        *,
        check_timeout=None,
        return_pre_postprocess_neuron: bool = False,
    ):
        self.prepare_chain_conn(chains, signal_image, check_timeout=check_timeout)
        self._vprint(
            f"prepare_chain_conn: graph vertices={self.graph.nvertex}, edges={self.graph.nedge}"
        )

        if check_timeout is not None:
            check_timeout("remove_redundant_edges")
        self.remove_redundant_edges()
        self._vprint(
            f"remove_redundant_edges: graph vertices={self.graph.nvertex}, edges={self.graph.nedge}"
        )

        if self.enable_crossover_test:
            if check_timeout is not None:
                check_timeout("crossover_test")
            self.crossover_test(chains)

        if check_timeout is not None:
            check_timeout("chains_to_circles")
        circle_list, circle_conn_list, circle_comp_list = self.chains_to_circles(chains)

        return self.circles_to_tree(
            circle_list,
            circle_conn_list,
            circle_comp_list,
            signal_image,
            check_timeout=check_timeout,
            return_pre_postprocess_neuron=return_pre_postprocess_neuron,
        )


    def prepare_chain_conn(self, chains: SegmentChains, signal_image: np.ndarray, *, check_timeout=None):
        """
        Construct a neuronal tree (in SWC format) for given SegmentChains
        """
        nchains = len(chains)
        self.graph.nvertex = nchains
        if nchains == 0:
            return None
        
        if nchains > 500:
            self.sp_test = False

        broadphase_stats = _chain_broadphase_stats(chains)
        global_max_radius = max(
            (stats["max_radius"] for stats in broadphase_stats if stats is not None),
            default=0.0,
        )
        global_distance_limit = _connection_distance_limit(global_max_radius, self.dist_thresh)
        cell_size = max(float(global_distance_limit), 1.0)
        spatial_grid = _build_segment_spatial_grid(broadphase_stats, cell_size)
        candidate_fallback_limit = max(1, int(_CANDIDATE_FALLBACK_RATIO * nchains))

        for i, chain1 in enumerate(chains):
            if check_timeout is not None and i % 4 == 0:
                check_timeout("chain connection")
            chain1_stats = broadphase_stats[i]
            if chain1_stats is None:
                continue

            candidate_indices = _candidate_indices_from_segment_grid(
                chain1_stats,
                spatial_grid,
                global_distance_limit,
                cell_size,
                fallback_limit=candidate_fallback_limit,
            )
            candidate_iter = range(nchains) if candidate_indices is None else sorted(candidate_indices)

            for j in candidate_iter:
                chain2 = chains[j]
                if i == j:
                    continue
                chain2_stats = broadphase_stats[j]
                if _can_skip_connect_test(chain1_stats, chain2_stats, self.dist_thresh):
                    continue
                conn = Neurocomp_Conn(
                    mode=ConnectorType.NEUROCOMP_CONN_HL,
                    info=np.array([0, 0], dtype=int),
                    pos=np.empty(3, dtype=np.float64),
                    ort=np.empty(3, dtype=np.float64),
                    cost=0.0,
                    min_pdist=-1.0,
                    min_sdist=-1.0
                )
                is_possible_connect = self.connect_test(
                    chain1,
                    chain2,
                    conn,
                    chain2_max_radius=None if chain2_stats is None else chain2_stats["max_radius"],
                )
                if is_possible_connect:
                    if conn.min_sdist > 2.0 and self.sp_test == True:
                        # initialize StackGraph
                        sgw = StackGraph(conn=26)
                        sgw.signal_mask = None
                        sgw.including_signal_border = True
                        path = self.get_shortest_path(chain1, chain2, signal_image, sgw)
                        gdist = self.validate_shortest_path(path, chain1, signal_image, sgw, conn)
                        conn.cost = 1.0 / (1.0 + exp(-(conn.min_sdist + gdist) / 100.0))
                    else:
                        conn.cost = 1.0 / (1.0 + exp(-conn.min_sdist / 100.0))
                    
                if conn.mode == ConnectorType.NEUROCOMP_CONN_NONE:
                    continue
                else:
                    if self.interpolate:
                        if conn.mode == ConnectorType.NEUROCOMP_CONN_HL:
                            index, conn.pos = interpolate_chain(chain2, conn.pos, conn.ort)
                            if index is not None:
                                conn.info[1] = index
                            else:
                                if conn.info[1] == 0:
                                    conn.info[1] = 0
                                    conn.mode = ConnectorType.NEUROCOMP_CONN_LINK
                                elif conn.info[1] == len(chain2)-1:
                                    conn.info[1] = 1
                                    conn.mode = ConnectorType.NEUROCOMP_CONN_LINK

                # go on
                if conn.info[1] <= 0:
                    conn.mode = ConnectorType.NEUROCOMP_CONN_LINK
                    conn.info[1] = 0
                elif conn.info[1] >= len(chain2) - 1:
                    conn.mode = ConnectorType.NEUROCOMP_CONN_LINK
                    conn.info[1] = 1

                conn_existed = False
                if i > j:
                    if self.graph.nedge > 0:
                        edge_idx = graph_edge_index(j, i, self.gw)
                        if edge_idx is not None:
                            if conn.mode == ConnectorType.NEUROCOMP_CONN_LINK:
                                if self.conn_list[edge_idx].info[0] == conn.info[1]:
                                    conn_existed = True
                            elif self.conn_list[edge_idx].mode == ConnectorType.NEUROCOMP_CONN_LINK:
                                if self.conn_list[edge_idx].info[1] == conn.info[0]:
                                    conn_existed = True
                            if conn_existed:
                                if self.conn_list[edge_idx].cost > conn.cost:
                                    self.conn_list[edge_idx] = conn
                                    self.graph.edges[edge_idx][0] = i
                                    self.graph.edges[edge_idx][1] = j
                                    graph_update_edge_table(self.graph, self.gw)
                if not conn_existed:
                    self.graph.add_edge(i, j)
                    self.conn_list.append(conn)
                    graph_expand_edge_table(i, j, self.graph.nedge - 1, self.gw)

                    
    def remove_redundant_edges(self):
        nedge = self.graph.nedge
        conn_list = self.conn_list
        edge_mask = np.zeros(nedge, dtype=np.uint8)
        edges = self.graph.edges

        # Set edge mask to 2 for link connection, 1 for hook-loop connection and 0 for no connection
        for i in range(nedge):
            if conn_list[i].mode == ConnectorType.NEUROCOMP_CONN_LINK:
                edge_mask[i] = 2
            elif conn_list[i].mode == ConnectorType.NEUROCOMP_CONN_HL:
                edge_mask[i] = 1
            else:
                edge_mask[i] = 0

        hook_loop_by_edge = {}
        for i in range(nedge):
            if edge_mask[i] == 1:
                hook_loop_by_edge.setdefault((edges[i][0], edges[i][1]), []).append(i)

        # Remove duplicated hook-loops by keeping the one with smaller cost.
        for i in range(nedge):
            if edge_mask[i] == 1:
                reverse_candidates = hook_loop_by_edge.get((edges[i][1], edges[i][0]), ())
                for j in reverse_candidates:
                    if j <= i or edge_mask[j] != 1:
                        continue
                    if conn_list[i].cost < conn_list[j].cost:
                        edge_mask[j] = 0
                    else:
                        edge_mask[i] = 0

        hook_loop_by_source = {}
        for i in range(nedge):
            if edge_mask[i] == 1:
                hook_loop_by_source.setdefault((edges[i][0], conn_list[i].info[0]), []).append(i)

        # Keep the smallest edge for multiple-loop connections.
        for i in range(nedge):
            if edge_mask[i] == 1:
                same_source_candidates = hook_loop_by_source.get((edges[i][0], conn_list[i].info[0]), ())
                for j in same_source_candidates:
                    if j <= i or edge_mask[j] != 1:
                        continue
                    if conn_list[i].cost < conn_list[j].cost:
                        edge_mask[j] = 0
                    else:
                        edge_mask[i] = 0

        # Move the edges into a compact array
        keep_indices = [i for i in range(nedge) if edge_mask[i] > 0]
        new_conn_list = [conn_list[i] for i in keep_indices]
        new_edges = [edges[i] for i in keep_indices]

        self.graph.nedge = len(new_conn_list)
        self.graph.edges = new_edges
        self.conn_list = new_conn_list
    
    def crossover_test(self, chains):
        nconn = self.graph.nedge
        ncomp = self.graph.nvertex

        status = np.zeros(nconn, dtype=np.uint8)
        
        dist_thresh = 20.0

        for i in range(ncomp):
            head_link_number = 0
            tail_link_number = 0

            head_link_index = []
            tail_link_index = []
            link_type = [0, 0]

            for j in range(nconn):
                if self.graph.edges[j][0] == i:
                    if self.conn_list[j].info[0] == 0:
                        if self.conn_list[j].mode == ConnectorType.NEUROCOMP_CONN_HL:
                            if self.conn_list[j].min_sdist < dist_thresh:
                                head_link_index.append(j)
                                head_link_number += 1
                        else:
                            head_link_index.append(j)
                            head_link_number += 1
                            if self.conn_angle(j, chains) < np.pi/4.0 and self.conn_list[j].min_sdist < dist_thresh:
                                link_type[0] = 1
                    else:
                        if self.conn_list[j].mode == ConnectorType.NEUROCOMP_CONN_HL:
                            if self.conn_list[j].min_sdist < dist_thresh:
                                tail_link_index.append(j)
                                tail_link_number += 1
                        else:
                            tail_link_index.append(j)
                            tail_link_number += 1
                            if self.conn_angle(j, chains) < np.pi/4.0 and self.conn_list[j].min_sdist < dist_thresh:
                                link_type[1] = 1
                
                if self.graph.edges[j][1] == i and self.conn_list[j].mode == ConnectorType.NEUROCOMP_CONN_LINK:
                    if self.conn_list[j].info[1] == 0:
                        head_link_index.append(j)
                        head_link_number += 1
                        if self.conn_angle(j, chains) < np.pi/4.0 and self.conn_list[j].min_sdist < dist_thresh:
                            link_type[0] = 1
                    else:
                        tail_link_index.append(j)
                        tail_link_number += 1
                        if self.conn_angle(j, chains) < np.pi/4.0 and self.conn_list[j].min_sdist < dist_thresh:
                            link_type[1] = 1
                
            if link_type[0] == 1:
                if head_link_number > 1:
                    for j in range(head_link_number):
                        if self.conn_list[head_link_index[j]].mode == ConnectorType.NEUROCOMP_CONN_HL:
                            status[head_link_index[j]] = 1
            elif link_type[1] == 1:
                if tail_link_number > 1:
                    for j in range(tail_link_number):
                        if self.conn_list[tail_link_index[j]].mode == ConnectorType.NEUROCOMP_CONN_HL:
                            status[tail_link_index[j]] = 1
        for i in range(nconn):
            if status[i] == 1:
                self.conn_list[i].cost += 1.0

        return

    def conn_angle(self, index, chains):
        if self.conn_list[index].mode == ConnectorType.NEUROCOMP_CONN_NONE:
            return -1.0
    
        chain1 = chains[self.graph.edges[index][0]]
        chain2 = chains[self.graph.edges[index][1]]

        if chain1 is None or chain2 is None:
            return -1.0

        mode = self.conn_list[index].mode

        if mode == ConnectorType.NEUROCOMP_CONN_HL:
            if self.conn_list[index].info[0] == 0:
                seg1 = chain1[0].copy()
                seg1.flip_segment()
            else:
                seg1 = chain1[-1].copy()

            seg2 = chain2[self.conn_list[index].info[1]].copy()

        elif mode == ConnectorType.NEUROCOMP_CONN_LINK:
            if self.conn_list[index].info[0] == 0:
                seg1 = chain1[0].copy()
                seg1.flip_segment()
            else:
                seg1 = chain1[-1].copy()

            if self.conn_list[index].info[1] == 0:
                seg2 = chain2[0].copy()
            else:
                seg2 = chain2[-1].copy()
                seg2.flip_segment()

        angle = np.arccos(np.clip(np.dot(seg1.dir_v, seg2.dir_v), -1.0, 1.0))

        return angle
    

    def chains_to_circles(self, chains):
        
        nchain = len(chains)
        nedge = self.graph.nedge

        start_id = np.empty(nchain+1, dtype=int)

        # initialize circle representation workspace
        ncomp = 0
        for i in range(nchain):
            start_id[i] = ncomp
            length = len(chains[i])
            assert length > 0, "Chain is empty"
            ncomp += length + 2
        ncomp *= 5

        circle_graph = Graph()
        circle_conn_list: List[Neurocomp_Conn] = []
        circle_comp_list: List[Circle] = []
        # ------------------------------------------

        def chain_to_circle(chain: SegmentChain):
            chain_coords, chain_indices = chain.to_coords(return_indices=True)
            n_chain_coords = len(chain_coords)

            coord_index = 0
            circle_list = []
            for i_seg, seg in enumerate(chain):
                # coord = chain_coords[coord_index]
                while coord_index < n_chain_coords:
                    coord_chain_idx = chain_indices[coord_index]
                    if coord_chain_idx == i_seg:
                        circle = Circle(chain_coords[coord_index], seg.radius*seg.scale, seg.theta, seg.psi)
                        circle_list.append(circle)

                        coord_index += 1
                    else:
                        break

            return circle_list
        
        for i_chain in range(nchain):
            chain = chains[i_chain]
            circle_list = chain_to_circle(chain)
            circle_comp_list.extend(circle_list)
            n_circle_list = len(circle_list)
            for i in range(n_circle_list-1):
                circle_conn_list.append(Neurocomp_Conn(mode=ConnectorType.NEUROCOMP_CONN_LINK, 
                                                       info=np.array([0, 0], dtype=int), 
                                                       cost=0.0,
                                                       pos=None, min_sdist=None, min_pdist=None, ort=None))
                circle_graph.add_edge(start_id[i_chain]+i, start_id[i_chain]+i+1)

            ncomp += n_circle_list
            start_id[i_chain + 1] = start_id[i_chain] + n_circle_list

        circle_graph.nvertex = ncomp

        id_ = [0, 0]
        for i in range(nedge):
            index1 = self.graph.edges[i][0]
            index2 = self.graph.edges[i][1]

            conn2 = self.conn_list[i]

            if conn2.mode == ConnectorType.NEUROCOMP_CONN_LINK or conn2.mode == ConnectorType.NEUROCOMP_CONN_HL:
                if conn2.info[0] == 0:
                    id_[0] = start_id[index1]  # chain head circle
                else:
                    id_[0] = start_id[index1 + 1] - 1  # chain tail circle

                if conn2.mode == ConnectorType.NEUROCOMP_CONN_LINK:
                    # Preserve LINK endpoint semantics: info[1] encodes target head (0) or tail (1).
                    if conn2.info[1] == 0:
                        id_[1] = start_id[index2]
                    else:
                        id_[1] = start_id[index2 + 1] - 1
                else:
                    id_[1] = closest_circle(
                        circle_comp_list[start_id[index2]: start_id[index2 + 1]],
                        start_id[index2 + 1] - start_id[index2],
                        conn2.pos,
                    ) + start_id[index2]
            
            conn = Neurocomp_Conn(mode=ConnectorType.NEUROCOMP_CONN_LINK, 
                                  info=np.array([0, 1], dtype=int), cost=conn2.cost,
                                  pos=None, min_sdist=None, min_pdist=None, ort=None)
            circle_graph.add_edge(id_[0], id_[1])
            circle_conn_list.append(conn)
        
        return circle_graph, circle_conn_list, circle_comp_list

    def circles_to_tree(
        self,
        circle_graph: Graph,
        circle_conn_list: List[Neurocomp_Conn],
        circle_comp_list: List[Circle],
        signal_image,
        *,
        check_timeout=None,
        return_pre_postprocess_neuron: bool = False,
    ):
        if not circle_comp_list:
            if return_pre_postprocess_neuron:
                return None, None
            return None
        
        circle_graph.weights = [circle_conn_list[i].cost for i in range(circle_graph.nedge)]

        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(len(circle_comp_list)))
        for i, (u, v) in enumerate(circle_graph.edges):
            G.add_edge(u, v, weight=circle_graph.weights[i])
        self._vprint(
            f"circles_to_tree: graph nodes={G.number_of_nodes()}, edges={G.number_of_edges()}"
        )
        tree = nx.minimum_spanning_tree(G, weight='weight')
        self._vprint(
            f"minimum_spanning_tree: nodes={tree.number_of_nodes()}, edges={tree.number_of_edges()}"
        )

        circle_graph.weights = None

        import time

        neuron = Neuron()
        t0 = time.time()
        if check_timeout is not None:
            check_timeout("from_graph")
        neuron.from_graph(tree, circle_comp_list)
        self._vprint(f'--> from_graph [length={neuron.length}]: {time.time() - t0:.5f}')
        pre_postprocess_neuron = _clone_neuron(neuron) if return_pre_postprocess_neuron else None
        neuron = postprocess_reconstruction(neuron, verbose=self.verbose, check_timeout=check_timeout)
        if return_pre_postprocess_neuron:
            return neuron, pre_postprocess_neuron
        return neuron
