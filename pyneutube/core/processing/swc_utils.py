
from collections import deque
import numpy as np

from pyneutube.core.io.swc_parser import Neuron


def _filtered_swc(swc: np.ndarray, nodes_to_remove) -> np.ndarray:
    mask = np.ones(len(swc), dtype=bool)
    mask[np.fromiter(nodes_to_remove, dtype=int, count=len(nodes_to_remove))] = False
    return swc[mask]


def _squared_distance(point1, point2):
    diff = point1 - point2
    return np.dot(diff, diff)


def is_sharp_turn(pos1, pos2, pos3):
    """
    Check if three continuous 3D positions form a sharp turn (pos1->pos2->pos3).
    Returns True if angle >= 90 degrees (dot product <= 0).
    """
    dir_v1 = pos2 - pos1
    dir_v2 = pos3 - pos2
    
    return np.dot(dir_v1, dir_v2) <= 0

def remove_zigzag(neuron: Neuron):
    """
    Processes nodes level by level which can be more appropriate for some tree structures.
    """
    # Cache frequently accessed attributes  
    nid_hash = neuron.nidHash
    children_map = neuron.indexChildren
    swc = neuron.swc
    positions = swc[:, 2:5]
    
    nodes_to_remove = set()
    
    # Initialize DFS with all soma children
    stack = deque()
    for soma in neuron.somata:
        soma_idx = nid_hash[soma[0]]
        stack.extend(children_map[soma_idx])
    
        # DFS traversal
        while stack:
            cur_idx = stack.pop()
            cur_node = swc[cur_idx]
            
            # Build parent chain
            parent1_idx = nid_hash.get(cur_node[6])
            if parent1_idx is None:
                stack.extend(children_map[cur_idx])
                continue
                
            parent1_node = swc[parent1_idx]
            parent2_idx = nid_hash.get(parent1_node[6])
            if parent2_idx is None:
                stack.extend(children_map[cur_idx])
                continue
                
            parent2_node = swc[parent2_idx]
            parent3_idx = nid_hash.get(parent2_node[6])
            if parent3_idx is None:
                stack.extend(children_map[cur_idx])
                continue
                
            # Check zigzag pattern at parent2
            if (
                is_sharp_turn(positions[parent3_idx], positions[parent2_idx], positions[parent1_idx])
                and is_sharp_turn(positions[parent2_idx], positions[parent1_idx], positions[cur_idx])
            ):
                
                # Reconnect all children of parent2 to parent3
                parent3_nid = swc[parent3_idx][0]
                for child_idx in children_map[parent2_idx]:
                    swc[child_idx][6] = parent3_nid
                
                nodes_to_remove.add(parent2_idx)
                children_map[parent3_idx] = children_map[parent2_idx] + children_map[parent3_idx]
            
            # Add children to queue
            stack.extend(children_map[cur_idx])
    
    # Apply deletions
    if nodes_to_remove:
        swc = _filtered_swc(swc, nodes_to_remove)
        neuron.initialize(swc)

def tune_branch(neuron: Neuron):
    """
    Optimize branch connections by bypassing intermediate nodes when beneficial.
    Reconnects nodes to grandparents if it reduces distance and maintains sharp turns.
    """
    # Cache frequently accessed attributes
    nid_hash = neuron.nidHash
    children_map = neuron.indexChildren
    swc = neuron.swc
    
    # Pre-compute positions for faster distance calculations
    positions = swc[:, 2:5]
    
    # Track modifications
    modified = False
    
    # Get soma indices more efficiently
    soma_indices = {nid_hash[soma[0]] for soma in neuron.somata}
    
    # Process each soma's subtree
    for soma_idx in soma_indices:
        stack = deque(children_map[soma_idx])
        
        while stack:
            cur_idx = stack.pop()
            cur_node = swc[cur_idx]
            
            # Get parent node
            parent_id = cur_node[6]
            if parent_id == -1:
                continue
                
            parent_idx = nid_hash.get(parent_id)
            if parent_idx is None:
                continue
            
            if len(children_map[parent_idx]) > 1:
                stack.extend(children_map[cur_idx])
                continue

            parent_node = swc[parent_idx]
            
            # Get grandparent node
            gparent_id = parent_node[6]
            if gparent_id == -1:
                continue
                
            gparent_idx = nid_hash.get(gparent_id)
            if gparent_idx is None:
                continue
            
            # Skip if grandparent has only one child (no branching point)
            if len(children_map[gparent_idx]) == 1:
                stack.extend(children_map[cur_idx])
                continue
            
            # Get great-grandparent node
            ggparent_id = swc[gparent_idx][6]
            if ggparent_id == -1:
                stack.extend(children_map[cur_idx])
                continue
                
            ggparent_idx = nid_hash.get(ggparent_id)
            if ggparent_idx is None:
                stack.extend(children_map[cur_idx])
                continue
            
            # Check turn conditions
            min_dist = np.inf
            new_pid = None
            if not is_sharp_turn(positions[gparent_idx], positions[parent_idx], positions[cur_idx]):
                min_dist = min(min_dist, _squared_distance(positions[parent_idx], positions[gparent_idx]))
            if not is_sharp_turn(positions[ggparent_idx], positions[parent_idx], positions[cur_idx]):
                tmp_dist = _squared_distance(positions[parent_idx], positions[ggparent_idx])
                if tmp_dist < min_dist:
                    min_dist = tmp_dist
                    new_pid = ggparent_id
                    modified = True

            gparent_children = children_map[gparent_idx]
            for child_idx in gparent_children[::-1]:  # child_idx is previous sibling in DFS
                if parent_idx == child_idx:
                    break
                if is_sharp_turn(positions[child_idx], positions[parent_idx], positions[cur_idx]):
                    tmp_dist = _squared_distance(positions[parent_idx], positions[child_idx])
                    if tmp_dist < min_dist:
                        min_dist = tmp_dist
                        new_pid = swc[child_idx][0]
                        modified = True

            # modify the children_map of gparent node
            if new_pid is not None:
                swc[parent_idx][6] = new_pid
                children_map[nid_hash[new_pid]].insert(0, parent_idx)
                children_map[gparent_idx].remove(parent_idx)

            stack.extend(children_map[cur_idx])
    
    # Only reinitialize if modifications were made
    if modified:
        neuron.initialize(swc)


def remove_spur(neuron: Neuron):
    """
    inplaced remove one node branch
    """

    nid_hash = neuron.nidHash
    swc = neuron.swc

    bifur_indices = set(neuron.bifur_indices)

    nodes_to_remove = set()
    for tip_idx in neuron.tip_indices:
        p_idx = nid_hash.get(swc[tip_idx][6])
        if p_idx in bifur_indices or p_idx is None:
            nodes_to_remove.add(tip_idx)

    if nodes_to_remove:
        swc = _filtered_swc(swc, nodes_to_remove)
        neuron.initialize(swc)

    return


def merge_close_point(neuron: Neuron, threshold=0.01):

    nid_hash = neuron.nidHash
    children_map = neuron.indexChildren
    swc = neuron.swc

    nodes_to_remove = set()

    stack = deque()
    threshold2 = threshold * threshold
    
    for soma in neuron.somata:
        soma_idx = nid_hash[soma[0]]
        stack.extend(children_map[soma_idx])

        while stack:
            cur_idx = stack.pop()
            cur_node = swc[cur_idx]  # for updating the parent

            parent1_idx = nid_hash[cur_node[6]]
            parent1_node = swc[parent1_idx]  # main node to calculate

            parent2_idx = nid_hash.get(parent1_node[6])
            if parent2_idx is None:
                stack.extend(children_map[cur_idx])
                continue

            parent2_node = swc[parent2_idx]
            parent1_children = list(children_map[parent1_idx])  # need a copy or it will be modified

            if _squared_distance(parent1_node[2:5], parent2_node[2:5]) < threshold2:
                
                parent2_nid = parent2_node[0]
                for child_idx in children_map[parent1_idx]:
                    swc[child_idx][6] = parent2_nid

                nodes_to_remove.add(parent1_idx)
        
            elif len(parent1_children) > 1 and cur_idx == parent1_children[0]:  # the last child in DFS
                parent1_children = deque(parent1_children)
                child1_idx = parent1_children.pop()
                while parent1_children:
                    child2_idx = parent1_children.pop()

                    child1_node = swc[child1_idx]
                    child2_node = swc[child2_idx]

                    if _squared_distance(child1_node[2:5], child2_node[2:5]) < threshold2:
                        child1_nid = child1_node[0]
                        for child_idx in children_map[child2_idx]:
                            swc[child_idx][6] = child1_nid

                        nodes_to_remove.add(child2_idx)
                    else:
                        child1_idx = child2_idx

    if nodes_to_remove:
        swc = _filtered_swc(swc, nodes_to_remove)
        neuron.initialize(swc)

    return


def remove_overshoot(neuron: Neuron):
    """
    Remove overshoot nodes around bifurcation points.
    Removes a node if:
      1) The node is a turn (sharp angle)
      2) One neighbor is a branch point and the other is not
    """
    nid_hash = neuron.nidHash
    children_map = neuron.indexChildren
    swc = neuron.swc
    positions = swc[:, 2:5]
    parent_ids = swc[:, 6].astype(int).ravel()
    
    remove_set = set()
    
    # Get bifurcation points (excluding soma)
    soma_indices = {nid_hash[soma[0]] for soma in neuron.somata}
    bifur_indices = set(neuron.bifur_indices) - soma_indices
    
    for bifur_idx in bifur_indices:
        bifur_pos = positions[bifur_idx]
        
        # Check parent side
        parent_id = parent_ids[bifur_idx]
        if parent_id != -1:
            parent_idx = nid_hash.get(parent_id)
            if parent_idx is not None and len(children_map[parent_idx]) == 1:
                parent_pos = positions[parent_idx]
                
                gparent_id = parent_ids[parent_idx]
                if gparent_id != -1:
                    gparent_idx = nid_hash.get(gparent_id)
                    if gparent_idx is not None and len(children_map[gparent_idx]) == 1:
                        gparent_pos = positions[gparent_idx]
                        
                        # Check angle
                        if is_sharp_turn(gparent_pos, parent_pos, bifur_pos):
                            remove_set.add(parent_idx)
                            swc[bifur_idx][6] = swc[gparent_idx][0]
        
        # Check children side
        for child_idx in children_map[bifur_idx]:
            if child_idx in remove_set:
                continue
                
            # Child must have exactly one child to be an overshoot candidate
            if len(children_map[child_idx]) == 1:
                child_pos = positions[child_idx]
                gchild_idx = children_map[child_idx][0]
                
                # Grandchild should not be a branch point
                if len(children_map[gchild_idx]) <= 1:
                    gchild_pos = positions[gchild_idx]
                    
                    # Check angle
                    if is_sharp_turn(bifur_pos, child_pos, gchild_pos):
                        remove_set.add(child_idx)
                        swc[gchild_idx][6] = swc[bifur_idx][0]
    
    # Apply removals
    if remove_set:
        swc = _filtered_swc(swc, remove_set)
        neuron.initialize(swc)


def optimal_downsample(neuron: Neuron):

    def _is_node1_within_node2(node1, node2):
        radius_margin = node2[5] - node1[5]
        if radius_margin < 0:
            return False
        return _squared_distance(node1[2:5], node2[2:5]) <= radius_margin * radius_margin
    def _is_node_overlap(node1, node2):
        radius_sum = node1[5] + node2[5]
        return _squared_distance(node1[2:5], node2[2:5]) < radius_sum * radius_sum
    def _interpolate_node(node1, node2, ref_node):
        d1 = np.linalg.norm(node1[2:5] - ref_node[2:5])
        d2 = np.linalg.norm(node2[2:5] - ref_node[2:5])
        t = d1 / (d1 + d2)
        new_pos = (1 - t) * node1[2:5] + t * node2[2:5]
        new_radius = (1 - t) * node1[5] + t * node2[5]

        interp_node = ref_node.copy()
        interp_node[2:5] = new_pos
        interp_node[5] = new_radius

        return interp_node
    def _is_inter_redundant(node1, node2, ref_node):
        """
        Check if a intermediate node (`ref_node`) can be represented by interpolation.
        node1 is parent, node2 is child, compared to ref_node.
        """
        interp_node = _interpolate_node(node1, node2, ref_node)
        size_scale = 1.2
        # Match NeuTu: the interpolated proxy must stay within half a radius.
        if _squared_distance(ref_node[2:5], interp_node[2:5]) * 4.0 < interp_node[5] * interp_node[5]:
            if ref_node[5] / size_scale < interp_node[5] < ref_node[5] * size_scale:
                return True

        return False
    
    while True:
        nid_hash = neuron.nidHash
        children_map = neuron.indexChildren
        swc = neuron.swc
        nodes_to_remove = set()

        stack = deque()
        for soma in neuron.somata:
            soma_idx = nid_hash[soma[0]]
            stack.extend(children_map[soma_idx])

            while stack:
                cur_idx = stack.pop()
                cur_node = swc[cur_idx]

                parent1_idx = nid_hash[cur_node[6]]
                parent1_node = swc[parent1_idx]

                parent2_idx = nid_hash.get(parent1_node[6])
                if parent2_idx is None:
                    stack.extend(children_map[cur_idx])
                    continue
            
                redundant = False
                parent2_node = swc[parent2_idx]
                
                if _is_node1_within_node2(parent1_node, parent2_node) or _is_node1_within_node2(parent1_node, cur_node):
                    redundant = True

                if not redundant:
                    if _is_node_overlap(parent1_node, parent2_node):
                        redundant = _is_inter_redundant(parent2_node, cur_node, parent1_node)

                if redundant:
                    parent2_nid = swc[parent2_idx][0]
                    for child_idx in children_map[parent1_idx]:
                        swc[child_idx][6] = parent2_nid
                    nodes_to_remove.add(parent1_idx)
                    remaining_children = [iidx for iidx in children_map[parent2_idx] if iidx != parent1_idx]
                    children_map[parent2_idx] = remaining_children + children_map[parent1_idx]
                    children_map[parent1_idx] = []

                stack.extend(children_map[cur_idx])

        if nodes_to_remove:
            swc = _filtered_swc(swc, nodes_to_remove)
            neuron.initialize(swc)
        else:
            break

    # optimizeCriticalParent
    nid_hash = neuron.nidHash
    children_map = neuron.indexChildren
    swc = neuron.swc
    bifur_indices = set(neuron.bifur_indices) - set([nid_hash[x] for x in neuron.somata[:, 0]])
    tip_indices = set(neuron.tip_indices)

    nodes_to_remove = set()

    stack = deque()
    for soma in neuron.somata:
        soma_idx = nid_hash[soma[0]]
        stack.extend(children_map[soma_idx])

        while stack:
            cur_idx = stack.pop()
            cur_node = swc[cur_idx]

            parent1_idx = nid_hash[cur_node[6]]
            parent1_node = swc[parent1_idx]

            parent2_idx = nid_hash.get(parent1_node[6])
            if parent2_idx is None:
                stack.extend(children_map[cur_idx])
                continue

            parent2_node = swc[parent2_idx]

            if cur_idx in bifur_indices or cur_idx in tip_indices:
                redundant = False
                if _is_node1_within_node2(parent1_node, cur_node) or _is_node1_within_node2(cur_node, parent1_node):
                    redundant = True

                if not redundant:
                    if _is_node_overlap(parent1_node, cur_node):
                        redundant = _is_inter_redundant(parent2_node, cur_node, parent1_node)

                if redundant:
                    parent2_nid = swc[parent2_idx][0]
                    for child_idx in children_map[parent1_idx]:
                        swc[child_idx][6] = parent2_nid
                    nodes_to_remove.add(parent1_idx)
                    remaining_children = [iidx for iidx in children_map[parent2_idx] if iidx != parent1_idx]
                    children_map[parent2_idx] = remaining_children + children_map[parent1_idx]
                    children_map[parent1_idx] = []
            
            stack.extend(children_map[cur_idx])

    if nodes_to_remove:
        swc = _filtered_swc(swc, nodes_to_remove)
        neuron.initialize(swc)


def remove_subtrees_by_length(neuron: Neuron, *, verbose: int = 1):

    if len(neuron.somata) < 10:
        return 

    nid_hash = neuron.nidHash
    children_map = neuron.indexChildren
    swc = neuron.swc
    parent_indices = np.full(len(swc), -1, dtype=int)
    for idx, node in enumerate(swc):
        parent_indices[idx] = nid_hash.get(node[6], -1)

    def _subtree_indices(root_idx: int):
        indices = [root_idx]
        stack = deque(children_map[root_idx])
        while stack:
            cur_idx = stack.pop()
            indices.append(cur_idx)
            stack.extend(children_map[cur_idx])
        return indices

    def _distance_to_root_map(root_idx: int, subtree_indices: list[int]):
        dist_to_root = {root_idx: 0.0}
        stack = deque(children_map[root_idx])
        while stack:
            cur_idx = stack.pop()
            parent_idx = parent_indices[cur_idx]
            dist_to_root[cur_idx] = dist_to_root[parent_idx] + np.linalg.norm(
                swc[cur_idx][2:5] - swc[parent_idx][2:5]
            )
            stack.extend(children_map[cur_idx])
        return dist_to_root

    def _common_ancestor_index(idx1: int, idx2: int) -> int:
        ancestors = set()
        cur_idx = idx1
        while cur_idx >= 0:
            ancestors.add(cur_idx)
            cur_idx = parent_indices[cur_idx]

        cur_idx = idx2
        while cur_idx not in ancestors:
            cur_idx = parent_indices[cur_idx]

        return cur_idx

    def _main_trunk_length(root_idx: int, subtree_indices: list[int]) -> float:
        leaves = [idx for idx in subtree_indices if len(children_map[idx]) == 0]
        candidates = [root_idx] + leaves
        if len(candidates) < 2:
            return 0.0

        dist_to_root = _distance_to_root_map(root_idx, subtree_indices)
        max_score = -1.0
        max_length = 0.0
        for i, idx1 in enumerate(candidates[:-1]):
            for idx2 in candidates[i + 1:]:
                ancestor_idx = _common_ancestor_index(idx1, idx2)
                geodesic_length = (
                    dist_to_root[idx1]
                    + dist_to_root[idx2]
                    - 2.0 * dist_to_root[ancestor_idx]
                )
                euclidean_length = np.linalg.norm(swc[idx1][2:5] - swc[idx2][2:5])
                score = geodesic_length * 0.2 + euclidean_length * 0.8
                if score > max_score:
                    max_score = score
                    max_length = geodesic_length

        return max_length

    root_indices = [nid_hash[soma[0]] for soma in neuron.somata]
    subtree_indices = [_subtree_indices(root_idx) for root_idx in root_indices]
    subtree_lengths = [_main_trunk_length(root_idx, indices) for root_idx, indices in zip(root_indices, subtree_indices)]

    mean_length = np.mean(subtree_lengths)
    keep_mask = np.zeros(len(swc), dtype=bool)
    for indices, trunk_length in zip(subtree_indices, subtree_lengths):
        if trunk_length >= mean_length:
            keep_mask[indices] = True

    neuron.initialize(swc[keep_mask])

    if verbose:
        print(
            "Total "
            f"{len(subtree_lengths)} subtrees, Removed "
            f"{len(subtree_lengths) - np.sum(np.asarray(subtree_lengths) >= mean_length)} subtrees, "
            f"Remove mean length: {mean_length}"
        )

    return
