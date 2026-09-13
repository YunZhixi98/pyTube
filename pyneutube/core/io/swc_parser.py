
import os
from typing import Dict, Union, Optional, List
import warnings

import numpy as np
import networkx as nx


class Neuron(object):
    def __init__(self):

        self.swc = None
        self.fn = "<memory>"
        self.source_path = None
        self.soma = None
        self.soma_idx = None
        self.somata = None
        self.bifur_indices = None
        self.tip_indices = None
        self.node_ids = None
        self.parent_ids = None
        self.node_types = None

        self.nidHash = None
        self.indexChildren = None

        self.dfs_nodes = None
        self.dfs_edges = None
        self.bfs_nodes = None
        self.bfs_edges = None

        # self.initialize()

    def __len__(self):
        return len(self.swc)

    def initialize(self, swc: Union[list, np.ndarray, str, os.PathLike] = None):
        if isinstance(swc, (str, os.PathLike)):
            self.source_path = os.fspath(swc)
            self.fn = os.path.basename(self.source_path)
            self.swc = self.read_swc(self.source_path)
        else:
            self.source_path = None
            self.fn = "<memory>"
            self.swc = np.asarray(swc, dtype=object)
        if self.swc.size == 0:
            raise ValueError("SWC input is empty.")
        self.length = len(self.swc)
        self.node_ids = self.swc[:, 0].astype(np.int64, copy=False)
        self.node_types = self.swc[:, 1].astype(np.int16, copy=False)
        self.parent_ids = self.swc[:, 6].astype(np.int64, copy=False)
        # self.soma = self.get_soma()

        self.nidHash = {node_id: i for i, node_id in enumerate(self.node_ids)}
        self.indexChildren = [[] for _ in range(self.length)]
        for i, pid in enumerate(self.parent_ids):
            idx = self.nidHash.get(pid)
            if idx is None:
                continue
            self.indexChildren[idx].append(i)

        self.somata = self.get_soma(allow_multiple=True)

        dfs_edges = []
        dfs_nodes = []
        for soma in self.somata:
            stack = list(self.indexChildren[self.nidHash[soma[0]]])
            while stack:
                cur_idx = stack.pop()
                dfs_edges.append((self.nidHash[self.parent_ids[cur_idx]], cur_idx))
                children = self.indexChildren[cur_idx]
                stack.extend(children)
        self.dfs_edges = dfs_edges

        # self.soma_idx = self.nidHash[self.soma[0]]
        valid_parent_ids = self.parent_ids[self.parent_ids != -1]
        pid_uni, pid_count = np.unique(valid_parent_ids, return_counts=True)
        
        self.bifur_indices = np.array([self.nidHash.get(x) for x in pid_uni[pid_count > 1]])
        self.tip_indices = np.flatnonzero(~np.isin(self.node_ids, self.parent_ids))

        return self

    def read_swc(self, path: str, mode: str = "t", scale: float = 1.0, eswc: bool = False):
        col_dtype = [np.int32, np.int32, np.float32, np.float32, np.float32, np.float32, np.int32]
        path = os.fspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"SWC file not found: {path}")
        if isinstance(scale, (int, float)):
            scale = np.array([scale, scale, scale])
        else:
            scale = np.asarray(scale, dtype=np.float32)
        if scale.shape != (3,):
            raise ValueError("`scale` must be a scalar or a length-3 vector.")
        swc_matrix = []
        with open(path, 'r') as f:
            while True:
                linelist = []
                line = f.readline()
                if not line:
                    break
                if line[0] == "#" or line[0] == 'i' or line[0] == "\n":
                    continue
                line = line.strip("\n").strip(" ")
                elem = line.split()
                if mode == "t":
                    pass
                elif mode == "a":
                    if elem[1] not in ['1', '2']:
                        continue
                elif mode.startswith("d"):
                    if mode == "d":
                        if elem[1] not in ['1', '3', '4']:
                            continue
                    elif mode == "da":
                        if elem[1] not in ['1', '4']:
                            continue
                    elif mode == "db":
                        if elem[1] not in ['1', '3']:
                            continue
                expected_cols = len(elem) if eswc else 7
                if len(elem) < expected_cols:
                    raise ValueError(f"Malformed SWC row in {path!r}: {line!r}")
                for i in range(expected_cols):
                    if i < 7:
                        linelist.append(col_dtype[i](elem[i]))
                    else:
                        linelist.append(float(elem[i]))
                swc_matrix.append(linelist)
        if not swc_matrix:
            raise ValueError(f"No SWC nodes were parsed from: {path}")
        swc_matrix = np.array(swc_matrix, dtype=object)
        # scaling neuron position and radius if needed
        swc_matrix[:, 2:5] *= scale
        swc_matrix[:, 5] *= np.prod(scale[0:2])

        return swc_matrix


    def get_soma(self, allow_multiple=False, strict=False):
        node_types = self.node_types
        parent_ids = self.parent_ids
        node_ids = self.node_ids
        somata = self.swc[(node_types == 1) & (parent_ids == -1)]
        if somata.size == 0:
            somata = self.swc[parent_ids == -1]
        if strict:
            if somata.size == 0:
                warnings.warn(f"{self.fn} no soma detected...", UserWarning)
            if not allow_multiple:
                warnings.warn(f"{self.fn} multi-soma detected, `allow_multiple` is {allow_multiple}, use the first one", UserWarning)
                return somata[0]
            return somata
        
        somata = self.swc[~np.isin(parent_ids, node_ids)]

        if somata.size == 0:
            warnings.warn(f"{self.fn} no soma detected...", UserWarning)
            return None
        
        if somata.shape[0] > 1:
            if not allow_multiple:
                warnings.warn(f"{self.fn} multi-soma detected, `allow_multiple` is {allow_multiple}, use the first one", UserWarning)
                return somata[0]
        
        return somata
    

    def save_swc(self, output_file_path: str, *, verbose: int = 1):
        """Save the SWC data to a specified file in SWC format."""

        def format_value(value):
            """Format the value to remove trailing zeros while keeping up to three decimal places."""
            return f"{value:.3f}".rstrip('0').rstrip('.')

        output_dir = os.path.dirname(os.fspath(output_file_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        with open(output_file_path, 'w') as f:
            f.write("#id type x y z radius pid\n")
            for row in self.swc:
                # Prepare the formatted values
                formatted_values = [format_value(value) for value in row[2:6]]  # X, Y, Z, Radius
                line = "{} {} {} {} {} {} {}\n".format(
                    int(row[0]),  # ID
                    int(row[1]),  # Type
                    *formatted_values,  # Unpack formatted X, Y, Z, Radius
                    int(row[6]),  # Parent
                )
                f.write(line)
        if verbose:
            print(f"SWC data saved to {output_file_path}")

        return
    
    def from_graph(self, graph: nx.Graph, circle_comp_list: list):
        
        swc = []
        for i, c in enumerate(nx.connected_components(graph)):
            subgraph = graph.subgraph(c)
            if subgraph.number_of_nodes() == 0:
                continue
            root = next(iter(subgraph.edges()))[1]
            swc.append([root, i+2] + list(circle_comp_list[root].center) + [circle_comp_list[root].radius, -1])
            for u, v in nx.bfs_edges(subgraph, source=root):
                circle = circle_comp_list[v]
                swc.append([v, i+2] + list(circle.center) + [circle.radius, u])

        self.initialize(swc)
        return self
        
    
    def get_path_length(self, start_id: Optional[int] = None):
        somata_ids = self.somata[:, 0].ravel()
        somata_idxs = [self.nidHash[x] for x in somata_ids]
        if start_id == None:
            start_id = somata_ids[0]
        start_idx =  self.nidHash[start_id]
        somata_idxs_set = set(somata_idxs) - {start_idx}

        path_length = 0.0
        start_compute = False
        for u, v in self.dfs_edges:
            if u == start_idx:
                start_compute = True
                
            if start_compute:
                path_length += np.linalg.norm(self.swc[v][2:5] - self.swc[u][2:5])
            
            if start_compute and u in somata_idxs_set:
                break
            
        return path_length
    

    # def split_subtrees(self,) -> List[np.ndarray]:
        
    #     subtrees = []
    #     for soma in self.somata:
    #         subtree = []
    #         stack = list(self.indexChildren[self.nidHash[soma[0]]])

    #         while stack:
    #             cur_node = 

    #     return subtrees

    @property
    def coords(self):
        return self.swc[:, 2:5]
    
