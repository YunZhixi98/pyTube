# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
from pyneutube.core import neighbors
from libc.stdlib cimport malloc, free
from libc.stdint cimport int64_t
from cython cimport boundscheck, wraparound


cdef int[:, ::1] neighbors_18 = np.ascontiguousarray(neighbors.neighbors_18, dtype=np.int32)
cdef int[:, ::1] neighbors_26_forward = np.ascontiguousarray(neighbors.neighbors_26_forward, dtype=np.int32)

ctypedef np.uint8_t UINT8_t
np.import_array()

ctypedef np.float64_t FLOAT64_t
ctypedef np.float32_t FLOAT32_t

ctypedef fused IMAGE_t:
    FLOAT32_t
    FLOAT64_t


@boundscheck(False)
@wraparound(False)
def Stack_Locmax_Region(np.ndarray[IMAGE_t, ndim=3] image,
                        np.ndarray[UINT8_t, ndim=3] loc_max_mask):
    """
    Clear every entry of `loc_max_mask` whose voxel is not a regional maximum.

    Neighbors outside the volume count as background (value 0, mask 0), which
    reproduces the earlier zero-padded implementation without allocating a
    padded copy. The work queue stores flat indices instead of (z, y, x)
    triples, so it costs 4 rather than 12 bytes per voxel, and it is left
    uninitialised so untouched pages are never faulted in.

    Accepts float32 or float64; comparisons are always in double precision, so
    both specialisations return the same mask for the same values.

    `loc_max_mask` is modified in place and returned.
    """

    cdef:
        IMAGE_t[:, :, ::1] img = image
        UINT8_t[:, :, ::1] mask = loc_max_mask
        Py_ssize_t depth = img.shape[0]
        Py_ssize_t height = img.shape[1]
        Py_ssize_t width = img.shape[2]
        Py_ssize_t plane = height * width
        Py_ssize_t total = depth * plane
        Py_ssize_t z, y, x, nz, ny, nx, n, rest
        Py_ssize_t head = 0, tail = 0
        Py_ssize_t n_neighbors = neighbors_18.shape[0]
        int dz, dy, dx
        int on_border
        double c, n_val
        int* queue

    if total > 2147483647:
        raise MemoryError(
            f"Stack_Locmax_Region supports at most 2**31-1 voxels, got {total}."
        )

    queue = <int*> malloc(total * sizeof(int))
    if queue == NULL:
        raise MemoryError("Unable to allocate the regional-maximum work queue.")

    try:
        # Step 1: a voxel with a strictly greater neighbor is not a maximum.
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    c = img[z, y, x]
                    if c == 0:
                        continue
                    on_border = (z == 0 or z == depth - 1 or
                                 y == 0 or y == height - 1 or
                                 x == 0 or x == width - 1)
                    for n in range(n_neighbors):
                        dz = neighbors_18[n, 0]
                        dy = neighbors_18[n, 1]
                        dx = neighbors_18[n, 2]
                        nz = z + dz
                        ny = y + dy
                        nx = x + dx
                        if on_border:
                            if (nz < 0 or nz >= depth or
                                    ny < 0 or ny >= height or
                                    nx < 0 or nx >= width):
                                n_val = 0
                            else:
                                n_val = img[nz, ny, nx]
                        else:
                            n_val = img[nz, ny, nx]
                        if n_val > c:
                            mask[z, y, x] = 0
                            queue[tail] = <int> (z * plane + y * width + x)
                            tail += 1
                            break

        # Step 2: propagate "not a maximum" to equal-or-lower neighbors.
        while head < tail:
            rest = queue[head]
            head += 1
            z = rest / plane
            rest = rest - z * plane
            y = rest / width
            x = rest - y * width
            c = img[z, y, x]
            for n in range(n_neighbors):
                dz = neighbors_18[n, 0]
                dy = neighbors_18[n, 1]
                dx = neighbors_18[n, 2]
                nz = z + dz
                ny = y + dy
                nx = x + dx
                if (nz < 0 or nz >= depth or
                        ny < 0 or ny >= height or
                        nx < 0 or nx >= width):
                    continue
                if mask[nz, ny, nx]:
                    n_val = img[nz, ny, nx]
                    if n_val <= c:
                        mask[nz, ny, nx] = 0
                        queue[tail] = <int> (nz * plane + ny * width + nx)
                        tail += 1
    finally:
        free(queue)

    return loc_max_mask


# please note that this function can be replaced by introducing scipy.ndimage.label and other minor operations.
# cpdef tuple Stack_Label_Objects_Ns(np.ndarray[UINT8_t, ndim=3] image_3d,
#                                    UINT8_t target_value,
#                                    UINT8_t new_label,
#                                    UINT8_t seed_label):
#     """
#     3D connected-component labeling for voxels == target_value.
#     Marks each new region with new_label, except the seed with seed_label.

#     Returns (labeled_array, object_count)
#     """
#     cdef int depth = image_3d.shape[0]
#     cdef int height = image_3d.shape[1]
#     cdef int width  = image_3d.shape[2]

#     # make a copy for output
#     cdef np.ndarray[UINT8_t, ndim=3] labeled = image_3d.copy()
#     # get a C-level view
#     cdef UINT8_t[:, :, :] lab = labeled

#     queue = deque()
#     cdef int object_count = 0

#     cdef int z, y, x, cz, cy, cx
#     cdef int dz, dy, dx, nz, ny, nx

#     cdef int n_neighbors = neighbors_18.shape[0]

#     # loop over every voxel
#     for z in range(depth):
#         for y in range(height):
#             for x in range(width):
#                 if lab[z, y, x] == target_value:
#                     # mark the seed
#                     lab[z, y, x] = seed_label
#                     queue.append((z, y, x))
#                     object_count += 1

#                     # BFS flood-fill
#                     while queue:
#                         cz, cy, cx = queue.popleft()
#                         # examine each of the 18 neighbors
#                         for n in range(n_neighbors):
#                             dz = neighbors_18[n][0]
#                             dy = neighbors_18[n][1]
#                             dx = neighbors_18[n][2]
#                             nz = cz + dz
#                             ny = cy + dy
#                             nx = cx + dx
#                             # check bounds
#                             if 0 <= nz < depth and 0 <= ny < height and 0 <= nx < width:
#                                 if lab[nz, ny, nx] == target_value:
#                                     lab[nz, ny, nx] = new_label
#                                     queue.append((nz, ny, nx))

#     return labeled, object_count


def Stack_Local_Max(np.ndarray[IMAGE_t, ndim=3] image):
    """
    Cythonized maximum filter mask:
    - 13 forward neighbors only
    - center<neighbor -> zero center
    - center>=neighbor -> kill neighbor

    Accepts float32 or float64; comparisons are always in double precision, so
    both specialisations return the same mask for the same values.
    """
    cdef:
        int depth = image.shape[0]
        int height = image.shape[1]
        int width = image.shape[2]
        np.ndarray[UINT8_t, ndim=3] out = np.ones_like(image, dtype=np.uint8)
        IMAGE_t[:,:,:] img = image
        UINT8_t[:,:,:] res = out
        int z, y, x, i, j
        int dz, dy, dx, nz, ny, nx
        double c, n


    # process boundaries first
    boundaries = neighbors.get_boundary_indices((depth, height, width))
    cdef int[:, :] bview = boundaries   # Cython memoryview
    cdef int n_boundaries = bview.shape[0]
    cdef int n_neighbors = neighbors_26_forward.shape[0]

    for i in range(n_boundaries):
        z = bview[i,0]
        y = bview[i,1]
        x = bview[i,2]
        c = img[z,y,x]
        if c == 0:
            res[z,y,x] = 0
            continue
        for j in range(n_neighbors):
            dz = neighbors_26_forward[j,0]
            dy = neighbors_26_forward[j,1]
            dx = neighbors_26_forward[j,2]
            nz = z + dz
            ny = y + dy
            nx = x + dx
            # bounds check
            if (0 <= nz < depth and 0 <= ny < height and 0 <= nx < width):
                n = img[nz,ny,nx]
                if n > 0:
                    if c < n:
                        # center is smaller: kill center
                        res[z,y,x] = 0
                    else:
                        # center >= neighbor: kill neighbor
                        res[nz,ny,nx] = 0
                else:
                    res[nz,ny,nx] = 0


    # internal voxels
    for z in range(1, depth-1):
        for y in range(1, height-1):
            for x in range(1, width-1):
                c = img[z,y,x]
                if c == 0:
                    res[z,y,x] = 0
                    continue
                for i in range(n_neighbors):
                    dz = neighbors_26_forward[i,0]
                    dy = neighbors_26_forward[i,1]
                    dx = neighbors_26_forward[i,2]
                    nz = z + dz
                    ny = y + dy
                    nx = x + dx
                    # bounds check (no need)
                    # if (0 <= nz < depth and 0 <= ny < height and 0 <= nx < width):
                    n = img[nz,ny,nx]
                    if n > 0:
                        if c < n:
                            # center is smaller: kill center
                            res[z,y,x] = 0
                        else:
                            # center >= neighbor: kill neighbor
                            res[nz,ny,nx] = 0
                    else:
                        res[nz,ny,nx] = 0

    return out



# # erode_3x3x3_cy_fixed2.pyx
# import numpy as np
# cimport numpy as np
# cimport cython

# # dtypes
# ctypedef np.int32_t I32
# ctypedef np.int64_t I64

# @cython.boundscheck(False)
# @cython.wraparound(False)
# def erode_3x3x3_cy(np.ndarray[np.uint8_t, ndim=3] in_arr):
#     """
#     Cython-implemented erosion for se = ones((3,3,3)) with nogil loops.
#     Accepts a contiguous np.int32 3D array (depth, height, width), values 0/1.
#     Returns np.int32 array.
#     """
#     if in_arr is None:
#         return None
#     if in_arr.ndim != 3:
#         raise ValueError("in_arr must be a 3D array (depth, height, width)")

#     # Ensure we have a contiguous int32 array and binary values
#     cdef np.ndarray[I32, ndim=3] arr = np.ascontiguousarray(in_arr, dtype=np.int32)
#     arr = (arr != 0).astype(np.int32, copy=False)

#     cdef int D = arr.shape[0]
#     cdef int H = arr.shape[1]
#     cdef int W = arr.shape[2]

#     # integral image as int64, contiguous
#     cdef np.ndarray[I64, ndim=3] ii = arr.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2).astype(np.int64, copy=False)

#     # output copy
#     cdef np.ndarray[I32, ndim=3] out = arr.copy()

#     # typed memoryviews for nogil region
#     cdef I32[:, :, :] a = arr
#     cdef I32[:, :, :] o = out
#     cdef I64[:, :, :] ii_mv = ii

#     cdef int z, y, x
#     cdef int x0, y0, z0, x1, y1, z1
#     cdef int x0c, y0c, z0c, x1c, y1c, z1c
#     cdef int Wm1 = W - 1
#     cdef int Hm1 = H - 1
#     cdef int Dm1 = D - 1
#     cdef I64 s
#     cdef int keep
#     cdef int nx, ny, nz
#     cdef int nnx, nny, nnz

#     print("bound:", 27)
#     print("thre:", 27)

#     with nogil:
#         for z in range(D):
#             for y in range(H):
#                 for x in range(W):
#                     if a[z, y, x] == 1:
#                         # box coords
#                         x0 = x - 1
#                         y0 = y - 1
#                         z0 = z - 1
#                         x1 = x + 1
#                         y1 = y + 1
#                         z1 = z + 1

#                         # If box is completely outside -> sum = 0
#                         if (x1 < 0) or (y1 < 0) or (z1 < 0) or (x0 > Wm1) or (y0 > Hm1) or (z0 > Dm1):
#                             s = 0
#                         else:
#                             # clamp coordinates to image bounds
#                             x0c = x0 if x0 >= 0 else 0
#                             y0c = y0 if y0 >= 0 else 0
#                             z0c = z0 if z0 >= 0 else 0
#                             x1c = x1 if x1 <= Wm1 else Wm1
#                             y1c = y1 if y1 <= Hm1 else Hm1
#                             z1c = z1 if z1 <= Dm1 else Dm1

#                             # inclusion-exclusion on clamped coords
#                             s = ii_mv[z1c, y1c, x1c]

#                             if (z0c - 1) >= 0:
#                                 s -= ii_mv[z0c - 1, y1c, x1c]
#                             if (y0c - 1) >= 0:
#                                 s -= ii_mv[z1c, y0c - 1, x1c]
#                             if (x0c - 1) >= 0:
#                                 s -= ii_mv[z1c, y1c, x0c - 1]

#                             if (z0c - 1) >= 0 and (y0c - 1) >= 0:
#                                 s += ii_mv[z0c - 1, y0c - 1, x1c]
#                             if (z0c - 1) >= 0 and (x0c - 1) >= 0:
#                                 s += ii_mv[z0c - 1, y1c, x0c - 1]
#                             if (y0c - 1) >= 0 and (x0c - 1) >= 0:
#                                 s += ii_mv[z1c, y0c - 1, x0c - 1]
#                             if (z0c - 1) >= 0 and (y0c - 1) >= 0 and (x0c - 1) >= 0:
#                                 s -= ii_mv[z0c - 1, y0c - 1, x0c - 1]

#                         # if full box -> keep
#                         if s == 27:
#                             continue

#                         # interior quick reject
#                         if (x != 0) and (x != Wm1) and (y != 0) and (y != Hm1) and (z != 0) and (z != Dm1):
#                             o[z, y, x] = 0
#                             continue

#                         # boundary: check only in-image offsets
#                         keep = 1
#                         nz = -1
#                         while nz <= 1 and keep:
#                             ny = -1
#                             while ny <= 1 and keep:
#                                 nx = -1
#                                 while nx <= 1:
#                                     nnx = x + nx
#                                     nny = y + ny
#                                     nnz = z + nz
#                                     if (0 <= nnx <= Wm1) and (0 <= nny <= Hm1) and (0 <= nnz <= Dm1):
#                                         if a[nnz, nny, nnx] != 1:
#                                             keep = 0
#                                             break
#                                     nx += 1
#                                 ny += 1
#                             nz += 1

#                         if keep == 0:
#                             o[z, y, x] = 0
#                         else:
#                             o[z, y, x] = 1
#                     # else center==0: nothing to do (out already copy)

#     return out


