# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport exp, isnan, sqrt
from libc.stdlib cimport malloc, realloc, free
from libc.string cimport memcpy
from libc.stdint cimport int64_t
from cpython.mem cimport PyMem_Malloc, PyMem_Realloc, PyMem_Free

np.import_array()

# Define numpy array types
ctypedef np.float64_t DTYPE_f64
ctypedef np.int32_t DTYPE_i32
ctypedef np.uint8_t DTYPE_u8

# The signal volume is float32 whenever that is exact, so the graph builders are
# compiled for both widths. Every voxel is widened to `double` before it reaches
# the weight function, so both specialisations agree on identical values.
ctypedef fused STACK_t:
    np.float32_t
    np.float64_t


# Enum for GraphType
cpdef enum GraphType:
    GENERAL_GRAPH = 0
    TREE_GRAPH = 1
    COMPLETE_GRAPH = 2

# Pure C struct for edge (fastest)
cdef struct EdgeStruct:
    int v1
    int v2

# Cython extension type for GraphEdge
cdef class GraphEdge:
    """Fast graph edge implementation"""
    cdef public int v1
    cdef public int v2
    
    def __init__(self, int v1, int v2):
        self.v1 = v1
        self.v2 = v2
    
    def __getitem__(self, int idx):
        if idx == 0:
            return self.v1
        elif idx == 1:
            return self.v2
        else:
            raise IndexError("GraphEdge index out of range")
    
    def __setitem__(self, int idx, int value):
        if idx == 0:
            self.v1 = value
        elif idx == 1:
            self.v2 = value
        else:
            raise IndexError("GraphEdge index out of range")
    
    def __repr__(self):
        return f"GraphEdge({self.v1}, {self.v2})"
    
    @property
    def edge(self):
        """Return edge as numpy array for compatibility"""
        return np.array([self.v1, self.v2], dtype=np.int32)
    
    cdef tuple as_tuple(self):
        """Fast C method to get edge as tuple"""
        return (self.v1, self.v2)

# Optimized Graph class using C arrays
cdef class FastGraph:
    """
    High-performance graph implementation using C arrays
    """
    cdef:
        bint directed
        int gtype
        int nvertex
        int nedge
        int edge_capacity
        EdgeStruct* edges_array  # C array of edges
        double* weights_array     # C array of weights
        bint weighted
        
    def __cinit__(self, int initial_capacity=1000, bint weighted=False, bint directed=False):
        self.directed = directed
        self.gtype = GENERAL_GRAPH
        self.nvertex = 0
        self.nedge = 0
        self.edge_capacity = initial_capacity
        self.weighted = weighted
        
        # Allocate C arrays
        self.edges_array = <EdgeStruct*>PyMem_Malloc(initial_capacity * sizeof(EdgeStruct))
        if not self.edges_array:
            raise MemoryError("Failed to allocate edges array")
        
        if weighted:
            self.weights_array = <double*>PyMem_Malloc(initial_capacity * sizeof(double))
            if not self.weights_array:
                PyMem_Free(self.edges_array)
                raise MemoryError("Failed to allocate weights array")
        else:
            self.weights_array = NULL
    
    def __dealloc__(self):
        if self.edges_array:
            PyMem_Free(self.edges_array)
        if self.weights_array:
            PyMem_Free(self.weights_array)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cdef void _grow_capacity(self):
        """Grow array capacity when needed"""
        cdef int new_capacity = self.edge_capacity * 2
        cdef EdgeStruct* new_edges
        cdef double* new_weights
        
        # Reallocate edges array
        new_edges = <EdgeStruct*>PyMem_Realloc(self.edges_array, 
                                                new_capacity * sizeof(EdgeStruct))
        if not new_edges:
            raise MemoryError("Failed to grow edges array")
        self.edges_array = new_edges
        
        # Reallocate weights array if needed
        if self.weighted:
            new_weights = <double*>PyMem_Realloc(self.weights_array, 
                                                 new_capacity * sizeof(double))
            if not new_weights:
                raise MemoryError("Failed to grow weights array")
            self.weights_array = new_weights
        
        self.edge_capacity = new_capacity
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef void add_edge(self, int v1, int v2, double weight=0.0):
        """Add an edge to the graph"""
        # Grow capacity if needed
        if self.nedge >= self.edge_capacity:
            self._grow_capacity()
        
        # Add edge
        self.edges_array[self.nedge].v1 = v1
        self.edges_array[self.nedge].v2 = v2
        
        # Add weight if weighted
        if self.weighted:
            self.weights_array[self.nedge] = weight
        
        # Update vertex count
        if self.nvertex < v1 + 1:
            self.nvertex = v1 + 1
        if self.nvertex < v2 + 1:
            self.nvertex = v2 + 1
        
        self.nedge += 1
    
    @cython.boundscheck(False)
    cpdef void add_edge_batch(self, np.ndarray[DTYPE_i32, ndim=2] edges, 
                              const double[:] weights=None):
        """Add multiple edges at once for better performance"""
        cdef int n_edges = edges.shape[0]
        cdef int i
        cdef int new_capacity
        
        # Ensure capacity
        if self.nedge + n_edges > self.edge_capacity:
            new_capacity = max(self.edge_capacity * 2, self.nedge + n_edges)
            self.edge_capacity = new_capacity
            self._grow_capacity()
        
        # Add all edges
        for i in range(n_edges):
            self.edges_array[self.nedge + i].v1 = edges[i, 0]
            self.edges_array[self.nedge + i].v2 = edges[i, 1]
            
            # Update vertex count
            if self.nvertex < edges[i, 0] + 1:
                self.nvertex = edges[i, 0] + 1
            if self.nvertex < edges[i, 1] + 1:
                self.nvertex = edges[i, 1] + 1
            
            if self.weighted and weights is not None:
                self.weights_array[self.nedge + i] = weights[i]
        
        self.nedge += n_edges
    
    cpdef bint is_weighted(self):
        """Check if graph is weighted"""
        return self.weighted
    
    cpdef tuple get_edge(self, int idx):
        """Get edge at index as tuple"""
        if idx < 0 or idx >= self.nedge:
            raise IndexError(f"Edge index {idx} out of range")
        return (self.edges_array[idx].v1, self.edges_array[idx].v2)
    
    cpdef double get_weight(self, int idx):
        """Get weight at index"""
        if not self.weighted:
            raise ValueError("Graph is not weighted")
        if idx < 0 or idx >= self.nedge:
            raise IndexError(f"Weight index {idx} out of range")
        return self.weights_array[idx]
    
    cpdef np.ndarray get_edges_array(self):
        """Get all edges as numpy array"""
        cdef np.ndarray[DTYPE_i32, ndim=2] result = np.empty((self.nedge, 2), dtype=np.int32)
        cdef int i
        
        for i in range(self.nedge):
            result[i, 0] = self.edges_array[i].v1
            result[i, 1] = self.edges_array[i].v2
        
        return result
    
    cpdef np.ndarray get_weights_array(self):
        """Get all weights as numpy array"""
        if not self.weighted:
            return None
        
        cdef np.ndarray[DTYPE_f64, ndim=1] result = np.empty(self.nedge, dtype=np.float64)
        cdef int i
        
        for i in range(self.nedge):
            result[i] = self.weights_array[i]
        
        return result
    
    @property
    def edges(self):
        """Python-compatible edges property"""
        edges_list = []
        for i in range(self.nedge):
            edges_list.append(GraphEdge(self.edges_array[i].v1, 
                                       self.edges_array[i].v2))
        return edges_list
    
    @property
    def weights(self):
        """Python-compatible weights property"""
        if not self.weighted:
            return None
        return [self.weights_array[i] for i in range(self.nedge)]
    
    def __len__(self):
        return self.nedge
    
    def __repr__(self):
        return f"FastGraph(nvertex={self.nvertex}, nedge={self.nedge}, weighted={self.weighted})"


# Python-compatible wrapper that mimics original interface
cdef class Graph:
    """
    Python-compatible graph class that uses FastGraph internally
    """
    cdef FastGraph fast_graph
    cdef public bint directed
    cdef public int gtype
    cdef public int nvertex
    cdef public int nedge
    cdef public int edge_capacity
    cdef list _edges_list  # Store Python list separately if needed
    cdef list _weights_list  # Store Python weights list separately if needed
    cdef bint _use_python_lists  # Flag to use Python lists instead of FastGraph
    
    def __init__(self, bint directed=False, int gtype=0, bint weighted=False, 
                 int initial_capacity=1000, bint use_python_lists=True):
        self.directed = directed
        self.gtype = gtype
        self.nvertex = 0
        self.nedge = 0
        self.edge_capacity = initial_capacity
        self._use_python_lists = use_python_lists
        
        if use_python_lists:
            # Use Python lists for full compatibility
            self._edges_list = []
            self._weights_list = [] if weighted else None
            self.fast_graph = None
        else:
            # Use FastGraph for performance
            self.fast_graph = FastGraph(initial_capacity=initial_capacity, 
                                        weighted=weighted, 
                                        directed=directed)
            self._edges_list = None
            self._weights_list = None
    
    def is_weighted(self):
        if self._use_python_lists:
            return self._weights_list is not None
        else:
            return self.fast_graph.is_weighted()
    
    def add_edge(self, int v1, int v2, double weight=0.0):
        if self._use_python_lists:
            self._edges_list.append(GraphEdge(v1, v2))
            if self._weights_list is not None:
                self._weights_list.append(weight)
            self.nedge += 1
            if self.nvertex < v1 + 1:
                self.nvertex = v1 + 1
            if self.nvertex < v2 + 1:
                self.nvertex = v2 + 1
        else:
            self.fast_graph.add_edge(v1, v2, weight)
            self.nvertex = self.fast_graph.nvertex
            self.nedge = self.fast_graph.nedge
    
    @property
    def edges(self):
        """Return edges list"""
        if self._use_python_lists:
            return self._edges_list
        else:
            return self.fast_graph.edges
    
    @edges.setter
    def edges(self, value):
        """Allow setting edges for compatibility"""
        self._use_python_lists = True
        self._edges_list = value
        if value:
            self.nedge = len(value)
            # Update nvertex based on edges
            max_vertex = -1
            for edge in value:
                if isinstance(edge, GraphEdge):
                    max_vertex = max(max_vertex, edge.v1, edge.v2)
                elif hasattr(edge, '__getitem__'):
                    max_vertex = max(max_vertex, edge[0], edge[1])
            self.nvertex = max_vertex + 1 if max_vertex >= 0 else 0
    
    @property
    def weights(self):
        """Return weights list"""
        if self._use_python_lists:
            return self._weights_list
        else:
            return self.fast_graph.weights
    
    @weights.setter
    def weights(self, value):
        """Allow setting weights for compatibility"""
        self._use_python_lists = True
        self._weights_list = value
    
    def get_edges_array(self):
        """Get edges as numpy array for efficient processing"""
        if self._use_python_lists:
            if not self._edges_list:
                return np.empty((0, 2), dtype=np.int32)
            edges_array = np.empty((len(self._edges_list), 2), dtype=np.int32)
            for i, edge in enumerate(self._edges_list):
                if isinstance(edge, GraphEdge):
                    edges_array[i, 0] = edge.v1
                    edges_array[i, 1] = edge.v2
                else:
                    edges_array[i, 0] = edge[0]
                    edges_array[i, 1] = edge[1]
            return edges_array
        else:
            return self.fast_graph.get_edges_array()
    
    def get_weights_array(self):
        """Get weights as numpy array for efficient processing"""
        if self._use_python_lists:
            if self._weights_list:
                return np.array(self._weights_list, dtype=np.float64)
            return None
        else:
            return self.fast_graph.get_weights_array()


@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void _graph_add_edge_inline(Graph graph, int v1, int v2, double weight):
    if graph._use_python_lists:
        graph.add_edge(v1, v2, weight)
    else:
        graph.fast_graph.add_edge(v1, v2, weight)


# Inline weight function
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline double _stack_voxel_weight_cy(double d, double v1, double v2, 
                                         double thre, double scale) nogil:
    """Inline version of stack voxel weight for maximum speed"""
    if isnan(thre):
        thre = 60.0
    if isnan(scale):
        scale = 5.0
    
    cdef double exp1 = exp((v1 - thre) / scale)
    cdef double exp2 = exp((v2 - thre) / scale)
    
    return d * (1.0 / (1.0 + exp1) + 1.0 / (1.0 + exp2) + 0.00001)

cpdef double stack_voxel_weight_cy(double d, double v1, double v2,
                                     double thre, double scale):
    """
    Python-callable wrapper for the internal weight function.
    This acquires the GIL and then calls the internal function.
    """
    return _stack_voxel_weight_cy(d, v1, v2, thre, scale)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void add_edges_cy(
    Graph graph,
    const STACK_t[:, :, :] stack,  # 3D stack data (float32 or float64)
    const unsigned char[:] cond,  # Condition array
    const int[:] x_offset,
    const int[:] y_offset, 
    const int[:] z_offset,
    const int[:] neighbor,
    const double[:] dist,
    int x, int y, int z,
    int offset,
    const int[:] stack_range,
    double[:] argv,  # Buffer for weight function arguments
    double[:] intensity  # Intensity array
):
    """
    Cython optimized version of add_edges function
    """
    cdef int i, nx, ny, nz
    cdef double weight
    cdef int conn = cond.shape[0]
    
    for i in range(conn):
        if cond[i] != 0:
            nx = x + stack_range[0]
            ny = y + stack_range[2] 
            nz = z + stack_range[4]

            if graph.is_weighted():
                # Prepare arguments for weight function
                argv[0] = dist[i]
                argv[1] = stack[nz, ny, nx]
                argv[2] = stack[
                    nz + z_offset[i],
                    ny + y_offset[i], 
                    nx + x_offset[i],
                ]
                # Call Python weight function (this will acquire GIL)
                weight = _stack_voxel_weight_cy(argv[0], argv[1], argv[2], argv[3], argv[4])
                
                # Add weighted edge (this will acquire GIL)
                graph.add_edge(offset, offset + neighbor[i], weight)
            else:
                # Add unweighted edge (this will acquire GIL)
                graph.add_edge(offset, offset + neighbor[i])
                # Store intensity value
                intensity[offset] = stack[nz, ny, nx]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void build_stack_graph_cy(
    Graph graph,
    const STACK_t[:, :, :] stack,
    int conn,
    const int[:] stack_range,
    const int[:] x_offset,
    const int[:] y_offset,
    const int[:] z_offset,
    const int[:] neighbor,
    const double[:] dist,
    const unsigned char[:, :, :, :] scan_boundary_masks,
    object signal_mask_obj,
    object group_mask_obj,
    bint including_signal_border,
    double[:] argv,
    double[:] intensity,
    int initial_virtual_vertex,
):
    cdef:
        int stack_depth = stack.shape[0]
        int stack_height = stack.shape[1]
        int stack_width = stack.shape[2]
        int cdepth = stack_range[5] - stack_range[4]
        int cheight = stack_range[3] - stack_range[2]
        int cwidth = stack_range[1] - stack_range[0]
        int x_state, y_state, z_state
        int x, y, z, i
        int org_x, org_y, org_z
        int neighbor_x, neighbor_y, neighbor_z
        int offset = 0
        int current_virtual_vertex = initial_virtual_vertex
        int group_id
        int group_vertex
        int center_active
        int neighbor_active
        double center_value
        double weight
        bint weighted = graph.is_weighted()
        np.ndarray[DTYPE_i32, ndim=1] group_vertex_map_np = np.zeros(256, dtype=np.int32)
        DTYPE_i32[:] group_vertex_map = group_vertex_map_np

    cdef const DTYPE_u8[:, :, :] signal_mask
    cdef const DTYPE_u8[:, :, :] group_mask
    cdef bint has_signal_mask = signal_mask_obj is not None
    cdef bint has_group_mask = group_mask_obj is not None

    if has_signal_mask:
        signal_mask = signal_mask_obj
    if has_group_mask:
        group_mask = group_mask_obj

    for z in range(cdepth + 1):
        if z == 0:
            z_state = 0
        elif z == cdepth:
            z_state = 2
        else:
            z_state = 1

        org_z = z + stack_range[4]

        for y in range(cheight + 1):
            if y == 0:
                y_state = 0
            elif y == cheight:
                y_state = 2
            else:
                y_state = 1

            org_y = y + stack_range[2]

            for x in range(cwidth + 1):
                if x == 0:
                    x_state = 0
                elif x == cwidth:
                    x_state = 2
                else:
                    x_state = 1

                org_x = x + stack_range[0]
                center_value = stack[org_z, org_y, org_x]

                if has_signal_mask:
                    center_active = signal_mask[org_z, org_y, org_x] > 0
                else:
                    center_active = 1

                for i in range(conn):
                    if scan_boundary_masks[x_state, y_state, z_state, i] == 0:
                        continue

                    if has_signal_mask:
                        if including_signal_border and center_active:
                            neighbor_active = 1
                        else:
                            neighbor_x = org_x + x_offset[i]
                            neighbor_y = org_y + y_offset[i]
                            neighbor_z = org_z + z_offset[i]
                            neighbor_active = signal_mask[neighbor_z, neighbor_y, neighbor_x] > 0
                            if including_signal_border:
                                if (center_active == 0) and (neighbor_active == 0):
                                    continue
                            else:
                                if (center_active == 0) or (neighbor_active == 0):
                                    continue

                    neighbor_x = org_x + x_offset[i]
                    neighbor_y = org_y + y_offset[i]
                    neighbor_z = org_z + z_offset[i]

                    if weighted:
                        weight = _stack_voxel_weight_cy(
                            dist[i],
                            center_value,
                            stack[neighbor_z, neighbor_y, neighbor_x],
                            argv[3],
                            argv[4],
                        )
                        _graph_add_edge_inline(graph, offset, offset + neighbor[i], weight)
                    else:
                        _graph_add_edge_inline(graph, offset, offset + neighbor[i], 0.0)
                        intensity[offset] = center_value

                if has_group_mask:
                    group_id = group_mask[org_z, org_y, org_x]
                    if group_id > 0:
                        group_vertex = group_vertex_map[group_id]
                        if group_vertex <= 0:
                            group_vertex = current_virtual_vertex
                            current_virtual_vertex += 1
                            group_vertex_map[group_id] = group_vertex
                        _graph_add_edge_inline(graph, group_vertex, offset, 0.0)

                offset += 1

    if not graph._use_python_lists:
        graph.nvertex = graph.fast_graph.nvertex
        graph.nedge = graph.fast_graph.nedge


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void graph_edge_neighbor_list_fast(
    int nvertex,
    list edges,  # List of GraphEdge objects
    int nedge,
    list neighbors,  # List of lists
    list edge_idx):  # List of lists or None
    """
    Fast Cython version of graph_edge_neighbor_list
    
    Parameters:
    -----------
    nvertex : int
        Number of vertices
    edges : list
        List of GraphEdge objects
    nedge : int
        Number of edges
    neighbors : list
        List of lists for neighbors (modified in place)
    edge_idx : list or None
        List of lists for edge indices (modified in place)
    """
    cdef:
        int i, v1, v2
        GraphEdge edge
        bint with_edge_idx = edge_idx is not None
        list v1_neighbors, v2_neighbors
        list v1_edge_idx, v2_edge_idx
    
    for i in range(nedge):
        edge = edges[i]
        v1 = edge.v1
        v2 = edge.v2

        # Add to adjacency lists
        neighbors[v1].append(v2)
        neighbors[v2].append(v1)
        
        if with_edge_idx:
            edge_idx[v1].append(i)
            edge_idx[v2].append(i)

        
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void _update_degrees_inline(
    list edges,  
    int[:] degree,          
    int nedge              
):
    """Inline function to calculate vertex degrees"""
    cdef int i, v1, v2
    for i in range(nedge):
        v1 = edges[i][0]
        v2 = edges[i][1]
        degree[v1] += 1
        degree[v2] += 1

# Python wrapper
def update_degrees(edges, degree):
    """
    Python wrapper for the inline degree calculation function
    """
    cdef int nedge = len(edges)
    _update_degrees_inline(edges, degree, nedge)


### simple mathematical or logical operations
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void stack_neighbor_offset_cy(int n_nbr, int width, int height, np.ndarray[DTYPE_i32, ndim=1] neighbor) noexcept:
    """
    Calculate neighbor offsets for different connectivity patterns
    """
    cdef int area = width * height
    
    if n_nbr == 4:
        neighbor[0] = -1
        neighbor[1] = 1
        neighbor[2] = -width
        neighbor[3] = width
    elif n_nbr == 6:
        stack_neighbor_offset_cy(4, width, height, neighbor)
        neighbor[4] = -area
        neighbor[5] = area
    elif n_nbr == 8:
        stack_neighbor_offset_cy(4, width, height, neighbor)
        neighbor[4] = -width - 1
        neighbor[5] = width + 1
        neighbor[6] = -width + 1
        neighbor[7] = width - 1
    elif n_nbr == 10:
        stack_neighbor_offset_cy(6, width, height, neighbor)
        neighbor[6] = -width - 1
        neighbor[7] = width + 1
        neighbor[8] = -width + 1
        neighbor[9] = width - 1
    elif n_nbr == 18:
        stack_neighbor_offset_cy(6, width, height, neighbor)
        neighbor[6] = -width - 1
        neighbor[7] = width + 1
        neighbor[8] = -width + 1
        neighbor[9] = width - 1
        neighbor[10] = -area - 1
        neighbor[11] = area + 1
        neighbor[12] = -area + 1
        neighbor[13] = area - 1
        neighbor[14] = -area - width
        neighbor[15] = area + width
        neighbor[16] = -area + width
        neighbor[17] = area - width
    elif n_nbr == 26:
        stack_neighbor_offset_cy(18, width, height, neighbor)
        neighbor[18] = -area - width - 1
        neighbor[19] = area + width + 1
        neighbor[20] = -area + width + 1
        neighbor[21] = area - width - 1
        neighbor[22] = -area - width + 1
        neighbor[23] = area + width - 1
        neighbor[24] = -area + width - 1
        neighbor[25] = area - width + 1
    else:
        raise ValueError(f"Unsupported neighbor count: {n_nbr}")

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef void stack_neighbor_dist_r_cy(int conn, np.ndarray[DTYPE_f64, ndim=1] res, np.ndarray[DTYPE_f64, ndim=1] dist) noexcept:
    """
    Calculate neighbor distances based on connectivity and resolution
    """
    cdef double diag_dist
    
    # Handle invalid resolution case
    if res[0] <= 0 or res[1] <= 0 or res[2] <= 0:
        for i in range(conn):
            dist[i] = 1.0
        return
    
    # Calculate distances based on connectivity pattern
    if conn == 4:
        dist[0] = dist[1] = res[0]
        dist[2] = dist[3] = res[1]
    elif conn == 6:
        dist[0] = dist[1] = res[0]
        dist[2] = dist[3] = res[1]
        dist[4] = dist[5] = res[2]
    elif conn == 8:
        stack_neighbor_dist_r_cy(4, res, dist)
        diag_dist = sqrt(res[0]*res[0] + res[1]*res[1])
        dist[4] = dist[5] = dist[6] = dist[7] = diag_dist
    elif conn == 18:
        stack_neighbor_dist_r_cy(6, res, dist)
        # Face-diagonal distances
        diag_dist = sqrt(res[0]*res[0] + res[1]*res[1])
        dist[6] = dist[7] = dist[8] = dist[9] = diag_dist
        # Edge-diagonal distances
        diag_dist = sqrt(res[0]*res[0] + res[2]*res[2])
        dist[10] = dist[11] = dist[12] = dist[13] = diag_dist
        diag_dist = sqrt(res[1]*res[1] + res[2]*res[2])
        dist[14] = dist[15] = dist[16] = dist[17] = diag_dist
    elif conn == 26:
        stack_neighbor_dist_r_cy(18, res, dist)
        # Corner-diagonal distances
        diag_dist = sqrt(res[0]*res[0] + res[1]*res[1] + res[2]*res[2])
        for i in range(18, 26):
            dist[i] = diag_dist
    else:
        raise ValueError(f"Unsupported connectivity: {conn}")

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int stack_util_offset_cy(int x, int y, int z, int width, int height, int depth) noexcept:
    """
    Calculate the linear offset/index from 3D coordinates
    """
    if (x < 0) or (y < 0) or (z < 0) or (x >= width) or (y >= height) or (z >= depth):
        return -1

    return x + width * y + width * height * z

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int stack_util_offset2_cy(int x, int y, int z, int width, int area) noexcept:
    """
    Calculate linear offset with precomputed area
    """
    return x + width * y + area * z

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef tuple stack_util_coord_cy(int index, int width, int area) noexcept:
    """
    Convert linear index to 3D coordinates
    """
    cdef int z = index // area
    cdef int remainder = index % area
    cdef int y = remainder // width
    cdef int x = remainder % width
    return x, y, z

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int stack_subindex_cy(int sindex, int x0, int y0, int z0, 
                          int swidth, int sarea, int width, int area) noexcept:
    """
    Calculate subindex with offset
    """
    cdef int x, y, z
    x, y, z = stack_util_coord_cy(sindex, swidth, sarea)
    x += x0
    y += y0
    z += z0
    
    return stack_util_offset2_cy(x, y, z, width, area)

@cython.boundscheck(False)
@cython.wraparound(False)
def stack_neighbor_bound_test_s_cy(int n_nbr, int cwidth, int cheight, int cdepth,
                               int x, int y, int z):
    """
    Test if neighbors are within bounds for a given connectivity pattern
    
    Args:
        n_nbr: Number of neighbors (4, 6, 8, 10, 18, or 26)
        cwidth: Width of the volume
        cheight: Height of the volume
        cdepth: Depth of the volume
        x: Current x coordinate
        y: Current y coordinate
        z: Current z coordinate
        
    Returns:
        tuple: (number of neighbors in bounds, in_bound array)
    """
    cdef np.ndarray[np.int32_t, ndim=1] is_in_bound = np.ones(n_nbr, dtype=np.int32)
    cdef int n_in_bound = n_nbr
    cdef int i
    
    # Check if all neighbors are in bounds
    if n_nbr == 4:
        if (x > 0) and (x < cwidth) and (y > 0) and (y < cheight) and (z >= 0) and (z <= cdepth):
            return n_nbr, is_in_bound
        if (x < 0) or (x > cwidth) or (y < 0) or (y > cheight) or (z < 0) or (z > cdepth):
            return 0, np.zeros(n_nbr, dtype=np.int32)
            
        if x == 0:
            is_in_bound[0] = 0
            n_in_bound -= 1
        if x == cwidth:
            is_in_bound[1] = 0
            n_in_bound -= 1
        if y == 0:
            is_in_bound[2] = 0
            n_in_bound -= 1
        if y == cheight:
            is_in_bound[3] = 0
            n_in_bound -= 1
            
    elif n_nbr == 6:
        if (x > 0) and (x < cwidth) and (y > 0) and (y < cheight) and (z > 0) and (z < cdepth):
            return n_nbr, is_in_bound
        if (x < 0) or (x > cwidth) or (y < 0) or (y > cheight) or (z < 0) or (z > cdepth):
            return 0, np.zeros(n_nbr, dtype=np.int32)
            
        if x == 0:
            is_in_bound[0] = 0
            n_in_bound -= 1
        if x == cwidth:
            is_in_bound[1] = 0
            n_in_bound -= 1
        if y == 0:
            is_in_bound[2] = 0
            n_in_bound -= 1
        if y == cheight:
            is_in_bound[3] = 0
            n_in_bound -= 1
        if z == 0:
            is_in_bound[4] = 0
            n_in_bound -= 1
        if z == cdepth:
            is_in_bound[5] = 0
            n_in_bound -= 1
            
    elif n_nbr == 8:
        if (x > 0) and (x < cwidth) and (y > 0) and (y < cheight) and (z >= 0) and (z <= cdepth):
            return n_nbr, is_in_bound
        if (x < 0) or (x > cwidth) or (y < 0) or (y > cheight) or (z < 0) or (z > cdepth):
            return 0, np.zeros(n_nbr, dtype=np.int32)
            
        if x == 0:
            is_in_bound[0] = is_in_bound[4] = is_in_bound[7] = 0
            n_in_bound -= 3
        if x == cwidth:
            is_in_bound[1] = is_in_bound[5] = is_in_bound[6] = 0
            n_in_bound -= 3
        if y == 0:
            is_in_bound[2] = is_in_bound[6] = 0
            n_in_bound -= 2
        if y == cheight:
            is_in_bound[3] = is_in_bound[7] = 0
            n_in_bound -= 2
            
    elif n_nbr == 18:
        if (x > 0) and (x < cwidth) and (y > 0) and (y < cheight) and (z > 0) and (z < cdepth):
            return n_nbr, is_in_bound
        if (x < 0) or (x > cwidth) or (y < 0) or (y > cheight) or (z < 0) or (z > cdepth):
            return 0, np.zeros(n_nbr, dtype=np.int32)
            
        if x == 0:
            is_in_bound[0] = is_in_bound[6] = is_in_bound[9] = is_in_bound[10] = is_in_bound[13] = 0
            n_in_bound -= 5
        if x == cwidth:
            is_in_bound[1] = is_in_bound[7] = is_in_bound[8] = is_in_bound[11] = is_in_bound[12] = 0
            n_in_bound -= 5
        if y == 0:
            is_in_bound[2] = is_in_bound[6] = is_in_bound[8] = is_in_bound[14] = is_in_bound[17] = 0
            n_in_bound -= 5
        if y == cheight:
            is_in_bound[3] = is_in_bound[7] = is_in_bound[9] = is_in_bound[15] = is_in_bound[16] = 0
            n_in_bound -= 5
        if z == 0:
            is_in_bound[4] = is_in_bound[10] = is_in_bound[12] = is_in_bound[14] = is_in_bound[16] = 0
            n_in_bound -= 5
        if z == cdepth:
            is_in_bound[5] = is_in_bound[11] = is_in_bound[13] = is_in_bound[15] = is_in_bound[17] = 0
            n_in_bound -= 5
            
    elif n_nbr == 26:
        if (x > 0) and (x < cwidth) and (y > 0) and (y < cheight) and (z > 0) and (z < cdepth):
            return n_nbr, is_in_bound
        if (x < 0) or (x > cwidth) or (y < 0) or (y > cheight) or (z < 0) or (z > cdepth):
            return 0, np.zeros(n_nbr, dtype=np.int32)
            
        if x == 0:
            is_in_bound[0] = is_in_bound[6] = is_in_bound[9] = is_in_bound[10] = is_in_bound[13] = \
                is_in_bound[18] = is_in_bound[21] = is_in_bound[23] = is_in_bound[24] = 0
            n_in_bound -= 9
        if x == cwidth:
            is_in_bound[1] = is_in_bound[7] = is_in_bound[8] = is_in_bound[11] = is_in_bound[12] = \
                is_in_bound[19] = is_in_bound[20] = is_in_bound[22] = is_in_bound[25] = 0
            n_in_bound -= 9
        if y == 0:
            is_in_bound[2] = is_in_bound[6] = is_in_bound[8] = is_in_bound[14] = is_in_bound[17] = \
                is_in_bound[18] = is_in_bound[21] = is_in_bound[22] = is_in_bound[25] = 0
            n_in_bound -= 9
        if y == cheight:
            is_in_bound[3] = is_in_bound[7] = is_in_bound[9] = is_in_bound[15] = is_in_bound[16] = \
                is_in_bound[19] = is_in_bound[20] = is_in_bound[23] = is_in_bound[24] = 0
            n_in_bound -= 9
        if z == 0:
            is_in_bound[4] = is_in_bound[10] = is_in_bound[12] = is_in_bound[14] = is_in_bound[16] = \
                is_in_bound[18] = is_in_bound[20] = is_in_bound[22] = is_in_bound[24] = 0
            n_in_bound -= 9
        if z == cdepth:
            is_in_bound[5] = is_in_bound[11] = is_in_bound[13] = is_in_bound[15] = is_in_bound[17] = \
                is_in_bound[19] = is_in_bound[21] = is_in_bound[23] = is_in_bound[25] = 0
            n_in_bound -= 9
            
    else:
        raise ValueError(f"Unsupported neighbor count: {n_nbr}")
    
    return n_in_bound, is_in_bound


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline (int, int, int) fast_unravel_index_3d(int64_t idx, int depth, int height, int width) nogil:
    """Fast unravel_index for 3D arrays (depth, height, width order)"""
    cdef:
        int z, y, x
        int64_t height_width = height * width
        int64_t remainder
    
    z = idx // height_width
    remainder = idx % height_width
    y = remainder // width
    x = remainder % width
    
    return z, y, x

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline int64_t fast_clip(int64_t val, int64_t min_val, int64_t max_val) nogil:
    """Fast scalar clip function"""
    if val < min_val:
        return min_val
    elif val > max_val:
        return max_val
    return val

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline void fast_clip_array(int64_t[:] arr, int64_t min_val, int64_t max_val, int64_t[:] out) nogil:
    """Fast clip for array of values"""
    cdef int i
    for i in range(arr.shape[0]):
        if arr[i] < min_val:
            out[i] = min_val
        elif arr[i] > max_val:
            out[i] = max_val
        else:
            out[i] = arr[i]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef process_voxel_with_neighbors(int64_t offset2, 
                                   int64_t[:] org_neighbor,
                                   int stack_depth,
                                   int stack_height,
                                   int stack_width,
                                   int64_t stack_voxels):
    """
    Process a single voxel with its 26 neighbors
    
    Parameters:
    -----------
    offset2 : int64_t
        The offset index
    org_neighbor : array[26]
        Array of 26 neighbor offsets
    stack_depth, stack_height, stack_width : int
        Stack dimensions
    stack_voxels : int64_t
        Total number of voxels
        
    Returns:
    --------
    tuple: (unravel_offset2, offset2_org_neighbor_clipped, unravel_offset2_org_neighbor)
        - unravel_offset2: tuple (z, y, x)
        - offset2_org_neighbor_clipped: numpy array[26] of clipped neighbor indices
        - unravel_offset2_org_neighbor: numpy array[26, 3] of unraveled neighbor coordinates
    """
    cdef:
        int i, z, y, x
        int64_t stack_voxels_minus_1 = stack_voxels - 1
        np.ndarray[int64_t, ndim=1] offset2_org_neighbor = np.empty(26, dtype=np.int64)
        np.ndarray[int, ndim=2] unravel_neighbor = np.empty((26, 3), dtype=np.int32)
        int64_t neighbor_offset
    
    # Unravel offset2
    z, y, x = fast_unravel_index_3d(offset2, stack_depth, stack_height, stack_width)
    unravel_offset2 = (z, y, x)
    
    # Process all 26 neighbors
    for i in range(26):
        # Add neighbor offset and clip
        neighbor_offset = offset2 + org_neighbor[i]
        neighbor_offset = fast_clip(neighbor_offset, 0, stack_voxels_minus_1)
        offset2_org_neighbor[i] = neighbor_offset
        
        # Unravel the neighbor index
        z, y, x = fast_unravel_index_3d(neighbor_offset, stack_depth, stack_height, stack_width)
        unravel_neighbor[i, 0] = z
        unravel_neighbor[i, 1] = y
        unravel_neighbor[i, 2] = x
    
    return unravel_offset2, offset2_org_neighbor, unravel_neighbor

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void process_voxel_with_neighbors_nogil(int64_t offset2, 
                                             int64_t[:] org_neighbor,
                                             int stack_depth,
                                             int stack_height,
                                             int stack_width,
                                             int64_t stack_voxels,
                                             int[:] unravel_offset2_out,
                                             int64_t[:] offset2_neighbor_out,
                                             int[:, :] unravel_neighbor_out) nogil:
    """
    NoGIL version for maximum performance - outputs through passed arrays
    
    Parameters:
    -----------
    unravel_offset2_out : array[3] - output for unraveled offset2
    offset2_neighbor_out : array[26] - output for clipped neighbor offsets
    unravel_neighbor_out : array[26, 3] - output for unraveled neighbor coordinates
    """
    cdef:
        int i, z, y, x
        int64_t stack_voxels_minus_1 = stack_voxels - 1
        int64_t neighbor_offset
    
    # Unravel offset2
    z, y, x = fast_unravel_index_3d(offset2, stack_depth, stack_height, stack_width)
    unravel_offset2_out[0] = z
    unravel_offset2_out[1] = y
    unravel_offset2_out[2] = x
    
    # Process all 26 neighbors
    for i in range(26):
        # Add neighbor offset and clip
        neighbor_offset = offset2 + org_neighbor[i]
        neighbor_offset = fast_clip(neighbor_offset, 0, stack_voxels_minus_1)
        offset2_neighbor_out[i] = neighbor_offset
        
        # Unravel the neighbor index
        z, y, x = fast_unravel_index_3d(neighbor_offset, stack_depth, stack_height, stack_width)
        unravel_neighbor_out[i, 0] = z
        unravel_neighbor_out[i, 1] = y
        unravel_neighbor_out[i, 2] = x

# Wrapper for the nogil version
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef process_voxel_neighbors_fast(int64_t offset2,
                                   int64_t [:] org_neighbor,
                                   int stack_depth,
                                   int stack_height,
                                   int stack_width,
                                   int64_t stack_voxels,
                                   int[:] unravel_offset2_out,
                                   int64_t[:] offset2_neighbor_out,
                                   int[:, :] unravel_neighbor_out):
    """
    Fast processing with pre-allocated output arrays
    """
    process_voxel_with_neighbors_nogil(
        offset2, org_neighbor,
        stack_depth, stack_height, stack_width, stack_voxels,
        unravel_offset2_out, offset2_neighbor_out, unravel_neighbor_out
    )

# Optimized batch processing for multiple voxels
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef process_voxel_batch(int64_t[:] offset2_array,
                          int64_t [:, :] org_neighbor_array,  # Shape: [n_voxels, 26]
                          int stack_depth,
                          int stack_height,
                          int stack_width,
                          int64_t stack_voxels,
                          int[:, :] unravel_offset2_out,      # Shape: [n_voxels, 3]
                          int64_t[:, :] offset2_neighbor_out,    # Shape: [n_voxels, 26]
                          int[:, :, :] unravel_neighbor_out): # Shape: [n_voxels, 26, 3]
    """
    Process multiple voxels at once
    """
    cdef:
        int v, i, z, y, x
        int n_voxels = offset2_array.shape[0]
        int64_t stack_voxels_minus_1 = stack_voxels - 1
        int64_t offset2, neighbor_offset
    
    with nogil:
        for v in range(n_voxels):
            offset2 = offset2_array[v]
            
            # Unravel offset2
            z, y, x = fast_unravel_index_3d(offset2, stack_depth, stack_height, stack_width)
            unravel_offset2_out[v, 0] = z
            unravel_offset2_out[v, 1] = y
            unravel_offset2_out[v, 2] = x
            
            # Process all 26 neighbors for this voxel
            for i in range(26):
                neighbor_offset = offset2 + org_neighbor_array[v, i]
                neighbor_offset = fast_clip(neighbor_offset, 0, stack_voxels_minus_1)
                offset2_neighbor_out[v, i] = neighbor_offset
                
                z, y, x = fast_unravel_index_3d(neighbor_offset, stack_depth, stack_height, stack_width)
                unravel_neighbor_out[v, i, 0] = z
                unravel_neighbor_out[v, i, 1] = y
                unravel_neighbor_out[v, i, 2] = x

