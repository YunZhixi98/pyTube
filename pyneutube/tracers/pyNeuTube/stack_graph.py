from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from pyneutube.core.processing.filtering import rc_threshold

from .chain_utils import get_inner_chain_range
from .geometry import point_to_chain_surface
from .shortest_path_accel import graph_shortest_path_dijkstra_cy
from .stack_graph_utils import (
    Graph,
    GraphEdge,
    add_edges_cy,
    build_stack_graph_cy,
    graph_edge_neighbor_list_fast,
    process_voxel_neighbors_fast,
    stack_neighbor_dist_r_cy,
    stack_neighbor_offset_cy,
    stack_subindex_cy,
    stack_util_offset_cy,
    stack_voxel_weight_cy,
)
from .tracing_utils import label_tracing_mask


   
# Define the offset arrays as numpy arrays for better performance
X_OFFSET_2D = np.array([-1, 1, 0, 0, -1, 1, 1, -1], dtype=np.int32)
Y_OFFSET_2D = np.array([0, 0, -1, 1, -1, 1, -1, 1], dtype=np.int32)
Z_OFFSET_2D = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int32)

X_OFFSET_3D = np.array([
    -1, 1, 0, 0, 0, 0, -1, 1, 1, -1,
    -1, 1, 1, -1, 0, 0, 0, 0, -1, 1,
    1, -1, 1, -1, -1, 1
], dtype=np.int32)

Y_OFFSET_3D = np.array([
    0, 0, -1, 1, 0, 0, -1, 1, -1, 1,
    0, 0, 0, 0, -1, 1, 1, -1, -1, 1,
    1, -1, -1, 1, 1, -1
], dtype=np.int32)

Z_OFFSET_3D = np.array([
    0, 0, 0, 0, -1, 1, 0, 0, 0, 0,
    -1, 1, -1, 1, -1, 1, -1, 1, -1, 1,
    -1, 1, -1, 1, -1, 1
], dtype=np.int32)

def stack_neighbor_x_offset(conn: int) -> np.ndarray:
    """
    Get X offsets for neighbor connectivity pattern
    
    Args:
        conn: Connectivity (8 for 2D, 26 for 3D)
        
    Returns:
        Array of X offsets
    """
    return X_OFFSET_2D if conn == 8 else X_OFFSET_3D[:conn]

def stack_neighbor_y_offset(conn: int) -> np.ndarray:
    """
    Get Y offsets for neighbor connectivity pattern
    
    Args:
        conn: Connectivity (8 for 2D, 26 for 3D)
        
    Returns:
        Array of Y offsets
    """
    return Y_OFFSET_2D if conn == 8 else Y_OFFSET_3D[:conn]

def stack_neighbor_z_offset(conn: int) -> np.ndarray:
    """
    Get Z offsets for neighbor connectivity pattern
    
    Args:
        conn: Connectivity (8 for 2D, 26 for 3D)
        
    Returns:
        Array of Z offsets
    """
    return Z_OFFSET_2D if conn == 8 else Z_OFFSET_3D[:conn]


@lru_cache(maxsize=None)
def _scan_boundary_masks(conn: int) -> np.ndarray:
    """Precompute positive-neighbor masks for each boundary case."""
    x_offset = stack_neighbor_x_offset(conn)
    y_offset = stack_neighbor_y_offset(conn)
    z_offset = stack_neighbor_z_offset(conn)
    scan_mask = (np.zeros(conn, dtype=np.int32) >= 0).astype(np.uint8)
    neighbor = np.zeros(conn, dtype=np.int32)
    stack_neighbor_offset_cy(conn, 8, 8, neighbor)
    scan_mask[:] = neighbor > 0

    masks = np.empty((3, 3, 3, conn), dtype=np.uint8)
    axis_valid = (
        (
            x_offset >= 0,
            np.ones(conn, dtype=bool),
            x_offset <= 0,
        ),
        (
            y_offset >= 0,
            np.ones(conn, dtype=bool),
            y_offset <= 0,
        ),
        (
            z_offset >= 0,
            np.ones(conn, dtype=bool),
            z_offset <= 0,
        ),
    )

    for x_state in range(3):
        for y_state in range(3):
            for z_state in range(3):
                masks[x_state, y_state, z_state] = (
                    scan_mask
                    & axis_valid[0][x_state]
                    & axis_valid[1][y_state]
                    & axis_valid[2][z_state]
                )

    return masks


def _boundary_state(coord: int, max_coord: int) -> int:
    if coord == 0:
        return 0
    if coord == max_coord:
        return 2
    return 1


def number_of_neighbors(v: int, neighbors: list) -> int:
    return 0 if neighbors[v] is None else neighbors[v][0]

def neighbor_of(v: int, n: int, neighbors: list) -> int:
    return neighbors[v][n]

# deprecated: reimplemented in Cython
def graph_edge_neighbor_list(nvertex: int, edges: List[GraphEdge], nedge: int, neighbors, edge_idx):
    """
    Constructs an adjacency list representation of a graph
    
    Parameters:
        nvertex: Number of vertices in the graph
        edges: List of edges, where each edge is a tuple/list of two vertices
        nedge: Number of edges
        neighbors: Adjacency list. Inplace update
        edge_idx: Edge index table (if input edge_idx is not None), otherwise just neighbors. Inplace update
    """
    with_edge_idx = edge_idx is not None
    for i, (v1, v2) in enumerate(edges):
        neighbors[v1].append(v2)
        neighbors[v2].append(v1)
        if with_edge_idx:
            edge_idx[v1].append(i)
            edge_idx[v2].append(i)
    

def graph_edge_index(v1: int, v2: int, gw: GraphWorkspace) -> Optional[int]:
    if gw.edge_table is None:
        return None

    key = f"{v1}_{v2}"
    index = gw.edge_table.get(key, None)
    if index is not None:
        return gw.edge_map[index]
    return None


def graph_update_edge_table(graph: Graph, gw: GraphWorkspace) -> None:
    if gw.is_ready(GraphWorkspaceStatus.GRAPH_WORKSPACE_EDGE_TABLE):
        return

    gw.nedge = graph.nedge

    if gw.edge_table is not None:
        gw.edge_table = {}
        gw.edge_map = []

    for i in range(gw.nedge):
        key = f"{graph.edges[i][0]}_{graph.edges[i][1]}"
        gw.edge_table[key] = i
        gw.edge_map.append(i)

def graph_expand_edge_table(v1: int, v2: int, edge_idx: int, gw: GraphWorkspace) -> None:
    if graph_edge_index(v1, v2, gw) is None:
        if gw.edge_table is None:
            gw.edge_table = {}
            gw.edge_map = []

        gw.nedge += 1
        key = f"{v1}_{v2}"
        gw.edge_table[key] = len(gw.edge_map)
        gw.edge_map.append(edge_idx)

class GraphWorkspaceStatus(Enum):
    GRAPH_WORKSPACE_FIELD_BEGIN = 0
    GRAPH_WORKSPACE_CONNECTION = auto()
    GRAPH_WORKSPACE_PARENT = auto()
    GRAPH_WORKSPACE_CHILD = auto()
    GRAPH_WORKSPACE_WEIGHT = auto()
    GRAPH_WORKSPACE_MINDIST = auto()
    GRAPH_WORKSPACE_IDX = auto()
    GRAPH_WORKSPACE_DEGREE = auto()
    GRAPH_WORKSPACE_IN_DEGREE = auto()
    GRAPH_WORKSPACE_OUT_DEGREE = auto()
    GRAPH_WORKSPACE_VLIST = auto()
    GRAPH_WORKSPACE_ELIST = auto()
    GRAPH_WORKSPACE_DLIST = auto()
    GRAPH_WORKSPACE_EDGE_TABLE = auto()
    GRAPH_WORKSPACE_EDGE_MAP = auto()
    GRAPH_WORKSPACE_STATUS = auto()
    GRAPH_WORKSPACE_FIELD_END = auto()

@dataclass
class GraphWorkspace:
    """
    A workspace for graph algorithms that maintains various data structures
    and state information for graph processing.
    
    Attributes:
        allocated: Bitmask tracking allocated fields
        ready: Bitmask tracking initialized/ready fields
        nvertex: Number of vertices in the graph
        nedge: Number of edges in the graph
        connection: Adjacency list representation
        parent: Parent relationships in traversal trees
        child: Child relationships in traversal trees
        weight: Edge weights
        mindist: Minimum distances (for shortest path algorithms)
        idx: Edge indices
        degree: Vertex degrees
        in_degree: Vertex in-degrees (for directed graphs)
        out_degree: Vertex out-degrees (for directed graphs)
        vlist: Vertex list (for various algorithms)
        elist: Edge list
        dlist: Distance list
        edge_table: Hash table for edge lookups
        edge_map: Mapping between edge representations
        status: Vertex status flags
    """
    # Bitmask fields
    allocated: int = 0
    ready: int = 0
    
    # Graph dimensions
    nvertex: int = 0
    nedge: int = 0
    
    # Connection and relationship data
    connection: Optional[np.ndarray] = None
    connection_psize: int = 0
    parent: Optional[np.ndarray] = None
    parent_psize: int = 0
    child: Optional[np.ndarray] = None
    child_psize: int = 0
    
    # Weight and distance data
    weight: Optional[np.ndarray] = None
    weight_psize: int = 0
    mindist: Optional[np.ndarray] = None
    mindist_psize: int = 0
    
    # Indexing data
    idx: Optional[np.ndarray] = None
    idx_psize: int = 0
    
    # Degree information
    degree: Optional[np.ndarray] = None
    in_degree: Optional[np.ndarray] = None
    out_degree: Optional[np.ndarray] = None
    
    # Algorithm working data
    vlist: Optional[np.ndarray] = None
    elist: Optional[np.ndarray] = None
    dlist: Optional[np.ndarray] = None
    
    # Edge management
    edge_table: Optional[Dict] = None  # Using dict instead of custom hash table
    edge_map: Optional[List] = None    # Using list instead of Int_Arraylist
    status: Optional[np.ndarray] = None
    
    # Constants
    EDGE_ENTRY_SIZE: int = 50
    
    def __post_init__(self):
        """Initialize numpy arrays with appropriate dtypes"""
        if self.vlist is None:
            self.vlist = np.zeros(self.nvertex, dtype=np.int32)
        if self.dlist is None:
            self.dlist = np.zeros(self.nvertex, dtype=np.float64)
        if self.status is None:
            self.status = np.zeros(self.nvertex, dtype=np.uint8)
    
    def prepare(self, field_id: GraphWorkspaceStatus) -> None:
        """
        Prepare a specific field in the graph workspace by clearing existing data if needed
        
        Args:
            field_id: Identifier for the field to prepare (GRAPH_WORKSPACE_* constant)
        """
        if not self.is_ready(field_id) and not self.is_allocated(field_id):
            if field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_CONNECTION:
                self.connection = None
                self.connection_psize = 0
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_PARENT:
                self.parent = None
                self.parent_psize = 0
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_CHILD:
                self.child = None
                self.child_psize = 0
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_WEIGHT:
                self.weight = None
                self.weight_psize = 0
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_MINDIST:
                self.mindist = None
                self.mindist_psize = 0
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_IDX:
                self.idx = None
                self.idx_psize = 0
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_DEGREE:
                self.degree = None
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_IN_DEGREE:
                self.in_degree = None
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_OUT_DEGREE:
                self.out_degree = None
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_ELIST:
                self.elist = None
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_VLIST:
                self.vlist = None
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_DLIST:
                self.dlist = None
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_EDGE_TABLE:
                self.edge_table = None
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_EDGE_MAP:
                self.edge_map = None
            elif field_id == GraphWorkspaceStatus.GRAPH_WORKSPACE_STATUS:
                self.status = None
            else:
                raise ValueError(f"Invalid field_id: {field_id}")
        
    def is_ready(self, field_flag: GraphWorkspaceStatus) -> bool:
        """Check if a field is ready for use"""
        return (self.ready >> field_flag.value) & 1 == 1

    def is_allocated(self, field_flag: GraphWorkspaceStatus) -> bool:
        return (self.allocated >> field_flag.value) & 1 == 1

    def load(self, graph: Graph) -> None:
        self.nvertex = graph.nvertex
        self.nedge = graph.nedge

    def _ensure_workspace_size(self, graph: Graph) -> None:
        self.load(graph)
        if self.degree is None or self.degree.shape[0] != self.nvertex:
            self.degree = np.zeros(self.nvertex, dtype=np.int32)
        if self.vlist is None or self.vlist.shape[0] != self.nvertex:
            self.vlist = np.empty(self.nvertex, dtype=np.int32)
        if self.dlist is None or self.dlist.shape[0] != self.nvertex:
            self.dlist = np.empty(self.nvertex, dtype=np.float64)
        if self.status is None or self.status.shape[0] != self.nvertex:
            self.status = np.empty(self.nvertex, dtype=np.uint8)

    def _graph_edges_array(self, graph: Graph) -> np.ndarray:
        return graph.get_edges_array()

    def load_graph(self, graph: Graph) -> list:
        """Load graph data into the workspace"""
        self._ensure_workspace_size(graph)
        self.graph_neighbor_list(graph)

        return self.connection

    def get_degree(self, graph) -> np.ndarray:
        """
        Calculate vertex degrees for the graph
        
        Returns:
            numpy.ndarray: Array of vertex degrees
        """
        self._ensure_workspace_size(graph)
        self.degree.fill(0)

        edges = self._graph_edges_array(graph)
        if edges.size == 0:
            return self.degree

        np.add.at(self.degree, edges[:, 0], 1)
        np.add.at(self.degree, edges[:, 1], 1)

        return self.degree

    def graph_neighbor_list(self, graph: Graph) -> np.ndarray:
        """
        Generate neighbor list representation of the graph
        
        Args:
            graph: Input graph object
            
        Returns:
            Adjacency list representation of the graph
        """
        self._ensure_workspace_size(graph)
        self.connection = [[] for _ in range(graph.nvertex)]
        self.idx = [[] for _ in range(graph.nvertex)]
        self.connection_psize = graph.nvertex
        self.idx_psize = graph.nvertex

        graph_edge_neighbor_list_fast(
            graph.nvertex,
            graph.edges,
            graph.nedge,
            self.connection,
            self.idx,
        )

        return self.connection

    def graph_shortest_path_e(
        self,
        graph: Graph,
        start: int,
        end: int,
    ) -> Optional[np.ndarray]:
        """
        Find shortest path between two nodes using Dijkstra's algorithm
        
        Args:
            start: Starting node index
            end: Ending node index
            
        Returns:
            Array of node indices representing the shortest path, or None if no path exists
        """
        self._ensure_workspace_size(graph)
        edges = graph.get_edges_array()
        if graph.is_weighted():
            weights = graph.get_weights_array()
        else:
            weights = np.ones(graph.nedge, dtype=np.float64)

        predecessors, distances = graph_shortest_path_dijkstra_cy(
            edges,
            weights,
            graph.nvertex,
            start,
            end,
        )
        self.vlist[:] = predecessors
        self.dlist[:] = distances
        self.status[:] = np.isfinite(distances).astype(np.uint8)
        return self.vlist
        

    
    def clear(self) -> None:
        """Reset the workspace while maintaining allocated memory"""
        self.ready = 0
        if self.vlist is not None:
            self.vlist.fill(0)
        if self.dlist is not None:
            self.dlist.fill(0.0)
        if self.status is not None:
            self.status.fill(0)


class StackGraph:
    STACK_GRAPH_WORKSPACE_ARGC: int = 10 

    def __init__(
        self,
        conn: int = 26,
        range_g: Optional[NDArray[int]] = None, # avoid using range directly!
        resolution: Optional[np.ndarray] = None,
        wf: Optional = stack_voxel_weight_cy,  
        sp_option: int = 0,
        argv: Optional[np.ndarray] = None,
        gw: Optional = None, 
        group_mask: Optional[np.ndarray] = None,
        signal_mask: Optional[np.ndarray] = None,
        intensity: Optional[List[float]] = None,
        value: Optional[float] = None,
        virtualVertex: int = -1,
        including_signal_border: bool = False,
    ):
        self.conn = conn
        self.range = range_g
        self.resolution = (
            np.ones((3), dtype=float)
            if resolution is None
            else np.asarray(resolution, dtype=float).copy()
        )
        self.wf = wf    
        self.sp_option = sp_option
        self.argv = (
            np.full(self.STACK_GRAPH_WORKSPACE_ARGC, np.nan)
            if argv is None
            else np.asarray(argv, dtype=float).copy()
        )
        self.gw = gw
        self.group_mask = group_mask    
        self.signal_mask = signal_mask  
        self.intensity = intensity
        self.value = value
        self.virtualVertex = virtualVertex
        self.including_signal_border = including_signal_border

    def set_range(self, x0: int, x1: int, y0: int, y1: int, z0: int, z1: int) -> None:
        if self.range is None:
            self.range = np.zeros((6), dtype=int)
        
        self.range[:2] = [x0, x1] if x0 < x1 else [x1, x0]
        self.range[2:4] = [y0, y1] if y0 < y1 else [y1, y0]
        self.range[4:] = [z0, z1] if z0 < z1 else [z1, z0]

    def update_range(self, x: int, y: int, z: int) -> None:
        self.range[0:2] = min(x, self.range[0]), max(x, self.range[1])
        self.range[2:4] = min(y, self.range[2]), max(y, self.range[3])
        self.range[4:6] = min(z, self.range[4]), max(z, self.range[5])

    def expand_range(self, margin: np.ndarray) -> None:
        self.range[::2] -= margin
        self.range[1::2] += margin

    def validate_range(self, width: int, height: int, depth: int) -> None:
        np.clip(self.range, [0, 0, 0, 0, 0, 0], 
                [width-1, width-1, height-1, height-1, depth-1, depth-1], self.range)

    from .tracing import SegmentChain, TracingSegment
    def update_stack_graph_workspace_by_seg_chain(self, seg: TracingSegment, chain: SegmentChain, signal_image: np.ndarray):
        pos = seg.center_coord
        _, skel_pos, seg_index = point_to_chain_surface(pos, chain)

        start, end = get_inner_chain_range(chain, seg_index, skel_pos)

        if self.sp_option != 1:
            if self.group_mask is None:
                self.group_mask = np.zeros(signal_image.shape, dtype=np.uint8)
            else:
                self.group_mask.fill(0)
            label_tracing_mask(
                chain,
                self.group_mask,
                dilate=True,
                start=start,
                end=end,
            )
            bbox = chain.get_label_bbox(start, end)
            start_pos = np.rint(pos).astype(int)
            if bbox is None:
                self.set_range(int(start_pos[0]), int(start_pos[0]), int(start_pos[1]), int(start_pos[1]), int(start_pos[2]), int(start_pos[2]))
            else:
                self.set_range(int(start_pos[0]), int(bbox[0]), int(start_pos[1]), int(bbox[2]), int(start_pos[2]), int(bbox[4]))
                self.update_range(int(bbox[1]), int(bbox[3]), int(bbox[5]))
        else:
            end_pos = np.rint(chain[seg_index].center_coord).astype(int)
            self.set_range(int(pos[0]), int(end_pos[0]), int(pos[1]), int(end_pos[1]), int(pos[2]), int(end_pos[2]))
        self.validate_range(signal_image.shape[2], signal_image.shape[1], signal_image.shape[0])

        if np.isnan(self.argv[3]) or np.isnan(self.argv[4]):
            substack = signal_image[self.range[4]:self.range[5]+1, self.range[2]:self.range[3]+1, self.range[0]:self.range[1]+1]
            if self.signal_mask is not None:
                submask = self.signal_mask[self.range[4]:self.range[5]+1, self.range[2]:self.range[3]+1, self.range[0]:self.range[1]+1]
                threshold, c1, c2 = rc_threshold(substack[submask])
            else:
                threshold, c1, c2 = rc_threshold(substack)
            
            self.argv[3] = threshold
            c2_c1 = c2 - c1

            if c2_c1 < 1.0:
                c2_c1 = 1.0
            c2_c1 /= 9.2
            self.argv[4] = c2_c1



    def _stack_graph_w_python(self, stack: np.ndarray) -> Graph:
        """
        `stack` is the input image or signal, that is a 3D array.
        """
        offset = 0
        stack_range = np.zeros(6, dtype=np.int32)
        stack_depth, stack_height, stack_width = stack.shape
        stack_voxels = stack_depth * stack_height * stack_width
        
        # Set stack range
        if self.range is None:
            stack_range[0], stack_range[1] = 0, stack_width - 1
            stack_range[2], stack_range[3] = 0, stack_height - 1
            stack_range[4], stack_range[5] = 0, stack_depth - 1
        else:
            stack_range[0] = max(0, self.range[0])
            stack_range[1] = min(stack_width - 1, self.range[1])
            stack_range[2] = max(0, self.range[2])
            stack_range[3] = min(stack_height - 1, self.range[3])
            stack_range[4] = max(0, self.range[4])
            stack_range[5] = min(stack_depth - 1, self.range[5])
        
        # Calculate dimensions
        cdepth = stack_range[5] - stack_range[4]
        cheight = stack_range[3] - stack_range[2]
        cwidth = stack_range[1] - stack_range[0]
        nvertex = (cwidth + 1) * (cheight + 1) * (cdepth + 1)
        self.virtualVertex = nvertex
        
        # Initialize graph
        weighted = True
        if self.sp_option == 1:
            weighted = False
            self.intensity = np.zeros(nvertex + 1, dtype=float)
            self.intensity[nvertex] = float('inf')

        graph = Graph(weighted=weighted, initial_capacity=nvertex, use_python_lists=False)

        # Get neighbor offsets
        neighbor = np.zeros((26), dtype=np.int32)
        stack_neighbor_offset_cy(self.conn, cwidth + 1, cheight + 1, neighbor)
        org_neighbor = np.zeros((26), dtype=np.int32)
        stack_neighbor_offset_cy(self.conn, stack_width, stack_height, org_neighbor)
        org_neighbor_i64 = org_neighbor.astype(np.int64)
        
        # Get distances
        dist = np.zeros(26, dtype=float)
        stack_neighbor_dist_r_cy(self.conn, self.resolution, dist)
        x_offset = stack_neighbor_x_offset(self.conn)
        y_offset = stack_neighbor_y_offset(self.conn)
        z_offset = stack_neighbor_z_offset(self.conn)

        scan_boundary_masks = _scan_boundary_masks(self.conn)
        signal_mask = self.signal_mask

        # Process vertices
        group_vertex_map = np.zeros(256, dtype=int)
        swidth = cwidth + 1
        sarea = (cwidth + 1) * (cheight + 1)
        area = stack_width * stack_height
        empty_cond = np.zeros(self.conn, dtype=np.uint8)
        cond_buffer = np.empty(self.conn, dtype=np.uint8) if signal_mask is not None else None

        # Pre-allocate outside the loop
        unravel_offset2_tmp = np.empty(3, dtype=np.int32)
        offset2_neighbor_tmp = np.empty(26, dtype=np.int64)
        unravel_neighbor_tmp = np.empty((26, 3), dtype=np.int32)

        for z in range(cdepth + 1):
            z_state = _boundary_state(z, cdepth)
            for y in range(cheight + 1):
                y_state = _boundary_state(y, cheight)
                for x in range(cwidth + 1):
                    x_state = _boundary_state(x, cwidth)

                    offset2 = stack_subindex_cy(
                        offset, stack_range[0], stack_range[2], stack_range[4],
                        swidth, sarea, stack_width, area
                    )
                    # Process with pre-allocated arrays (fastest)
                    process_voxel_neighbors_fast(
                        offset2, org_neighbor_i64,
                        stack_depth, stack_height, stack_width, stack_voxels,
                        unravel_offset2_tmp, offset2_neighbor_tmp, unravel_neighbor_tmp
                    )

                    # Use the results from the tmp arrays
                    unravel_offset2_org_neighbor = unravel_neighbor_tmp
                    cond_mask = scan_boundary_masks[x_state, y_state, z_state]
                    if signal_mask is None:
                        cond = cond_mask
                    else:
                        center_active = signal_mask[
                            unravel_offset2_tmp[0],
                            unravel_offset2_tmp[1],
                            unravel_offset2_tmp[2],
                        ] > 0
                        if self.including_signal_border and center_active:
                            cond = cond_mask
                        elif (not self.including_signal_border) and (not center_active):
                            cond = empty_cond
                        else:
                            neighbor_active = signal_mask[
                                unravel_offset2_org_neighbor[:, 0],
                                unravel_offset2_org_neighbor[:, 1],
                                unravel_offset2_org_neighbor[:, 2],
                            ] > 0
                            cond_buffer[:] = neighbor_active
                            cond_buffer &= cond_mask
                            cond = cond_buffer

                    # add edges
                    add_edges_cy(graph, stack, cond, x_offset, y_offset, z_offset, neighbor,
                                 dist, x, y, z, offset, stack_range, self.argv, self.intensity)

                    if self.group_mask is not None:
                        group_id = self.group_mask[
                            unravel_offset2_tmp[0],
                            unravel_offset2_tmp[1],
                            unravel_offset2_tmp[2],
                        ]
                        if group_id > 0:
                            group_vertex = group_vertex_map[group_id]
                            if group_vertex <= 0:
                                group_vertex = nvertex
                                nvertex += 1
                                group_vertex_map[group_id] = group_vertex
                            graph.add_edge(group_vertex, offset, 0.0)
                    
                    offset += 1
        
        return graph

    def stack_graph_w(self, stack: np.ndarray) -> Graph:
        offset = 0
        stack_range = np.zeros(6, dtype=np.int32)
        stack_depth, stack_height, stack_width = stack.shape

        if self.range is None:
            stack_range[0], stack_range[1] = 0, stack_width - 1
            stack_range[2], stack_range[3] = 0, stack_height - 1
            stack_range[4], stack_range[5] = 0, stack_depth - 1
        else:
            stack_range[0] = max(0, self.range[0])
            stack_range[1] = min(stack_width - 1, self.range[1])
            stack_range[2] = max(0, self.range[2])
            stack_range[3] = min(stack_height - 1, self.range[3])
            stack_range[4] = max(0, self.range[4])
            stack_range[5] = min(stack_depth - 1, self.range[5])

        cdepth = stack_range[5] - stack_range[4]
        cheight = stack_range[3] - stack_range[2]
        cwidth = stack_range[1] - stack_range[0]
        nvertex = (cwidth + 1) * (cheight + 1) * (cdepth + 1)
        self.virtualVertex = nvertex

        weighted = self.sp_option != 1
        if not weighted:
            self.intensity = np.zeros(nvertex + 1, dtype=float)
            self.intensity[nvertex] = float("inf")

        graph = Graph(weighted=weighted, initial_capacity=nvertex, use_python_lists=False)

        neighbor = np.zeros((26), dtype=np.int32)
        stack_neighbor_offset_cy(self.conn, cwidth + 1, cheight + 1, neighbor)
        dist = np.zeros(26, dtype=float)
        stack_neighbor_dist_r_cy(self.conn, self.resolution, dist)
        x_offset = stack_neighbor_x_offset(self.conn)
        y_offset = stack_neighbor_y_offset(self.conn)
        z_offset = stack_neighbor_z_offset(self.conn)
        scan_boundary_masks = _scan_boundary_masks(self.conn)

        if self.intensity is None:
            self.intensity = np.empty(0, dtype=float)

        build_stack_graph_cy(
            graph,
            stack,
            self.conn,
            stack_range,
            x_offset,
            y_offset,
            z_offset,
            neighbor,
            dist,
            scan_boundary_masks,
            self.signal_mask,
            self.group_mask,
            self.including_signal_border,
            self.argv,
            self.intensity,
            nvertex,
        )

        return graph

    def parse_stack_shortest_path(self, path: np.ndarray, start: int, end: int,
                             width: int, height: int) -> List[int]:
        """
        Parse shortest path from a graph traversal result
        
        Args:
            path: Array containing path information (previous node indices)
            start: Starting node index
            end: Ending node index
            width: Original stack width
            height: Original stack height
            
        Returns:
            List of indices representing the shortest path
        """
        area = width * height
        swidth = self.range[1] - self.range[0] + 1
        sheight = self.range[3] - self.range[2] + 1
        sdepth = self.range[5] - self.range[4] + 1
        sarea = swidth * sheight
        svolume = sarea * sdepth
        
        path_indices = []
        
        while end >= 0:
            index = -1
            if end < svolume:
                index = stack_subindex_cy(end, self.range[0], self.range[2], self.range[4],
                                     swidth, sarea, width, area)
            path_indices.append(index)
            end = path[end]
        
        return path_indices[::-1]

    def stack_route(self, stack: np.ndarray, start: np.ndarray, end: np.ndarray) -> List:
        if self.gw is None:
            self.gw = GraphWorkspace()

        stack_depth, stack_height, stack_width = stack.shape

        if self.range is None:
            dist = np.linalg.norm(start - end)
            margin = np.round(dist - np.fabs(end - start + 1)).astype(int)
            margin[margin < 0] = 0
        
            self.set_range(start[0], end[0], start[1], end[1], start[2], end[2])
            self.expand_range(margin)
            self.validate_range(stack_width, stack_height, stack_depth)

        swidth = self.range[1] - self.range[0] + 1
        sheight = self.range[3] - self.range[2] + 1
        sdepth = self.range[5] - self.range[4] + 1

        start_index = stack_util_offset_cy(start[0] - self.range[0],
                                         start[1] - self.range[2],
                                         start[2] - self.range[4],
                                         swidth, sheight, sdepth)
        end_index = stack_util_offset_cy(end[0] - self.range[0],
                                      end[1] - self.range[2],
                                      end[2] - self.range[4],
                                      swidth, sheight, sdepth)

        if (start_index > end_index):
            start_index, end_index = end_index, start_index

        assert start_index >= 0, start_index
        assert end_index >= 0, end_index

        # initialize a StackGraph_W object
        graph = self.stack_graph_w(stack)

        path = self.gw.graph_shortest_path_e(
            graph,
            start_index,
            end_index,
        )

        self.value = self.gw.dlist[end_index]   # gw
        
        if np.isinf(self.value):
            return []

        offset_path = self.parse_stack_shortest_path(path, start_index, end_index, stack_width, stack_height) 

        org_start = stack_util_offset_cy(start[0], start[1], start[2], stack_width, stack_height, stack_depth)
        if org_start != offset_path[0]:
            offset_path = offset_path[::-1]

        org_end = stack_util_offset_cy(end[0], end[1], end[2], stack_width, stack_height, stack_depth)
        
        assert(org_start == offset_path[0])

        assert(org_end == offset_path[-1])
        
        return offset_path
