# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

from libc.math cimport sin, cos, sqrt, fabs, floor, ceil, isfinite
cimport numpy as cnp


def label_segment(seg, cnp.uint8_t[:, :, :] mask, bint dilate):
    """Label voxel centers inside a segment, returning its occupied XYZ bounds.

    Uses NeuTube's axial interval [-0.5, h - 0.5] and elliptical cross section.
    The mask may be strided; its existing labels outside the segment are kept.
    """
    cdef double ry = max(float(seg.radius), 1e-8)
    cdef double rx = max(float(seg.radius * seg.scale), 1e-8)
    cdef double h = float(seg.length)
    cdef double sx = float(seg.start_coord[0])
    cdef double sy = float(seg.start_coord[1])
    cdef double sz = float(seg.start_coord[2])
    cdef double theta = float(seg.theta), psi = float(seg.psi)
    if not (isfinite(rx) and isfinite(ry) and isfinite(h)
            and isfinite(sx) and isfinite(sy) and isfinite(sz)
            and isfinite(theta) and isfinite(psi)):
        raise ValueError("Segment geometry must be finite.")
    if h <= 0 or mask.shape[0] == 0 or mask.shape[1] == 0 or mask.shape[2] == 0:
        return None
    if dilate:
        rx = min(rx * 1.5, rx + 3.0)
        ry = min(ry * 1.5, ry + 3.0)

    cdef double ct = cos(theta), st = sin(theta)
    cdef double cp = cos(psi), sp = sin(psi)
    # Columns of Rz(psi) Rx(theta): local X, Y and segment axis in world space.
    cdef double ux = cp, uy = sp
    cdef double vx = -sp * ct, vy = cp * ct, vz = st
    cdef double wx = sp * st, wy = -cp * st, wz = ct
    cdef double half = h / 2.0, center_z = (h - 1.0) / 2.0
    cdef double cx = sx + wx * center_z
    cdef double cy = sy + wy * center_z
    cdef double cz = sz + wz * center_z
    cdef double ex = sqrt((rx * ux) ** 2 + (ry * vx) ** 2) + fabs(wx) * half
    cdef double ey = sqrt((rx * uy) ** 2 + (ry * vy) ** 2) + fabs(wy) * half
    cdef double ez = fabs(ry * vz) + fabs(wz) * half
    # Clip in floating point before conversion, including fully external segments.
    cdef double lx = max(0.0, floor(cx - ex)), hx = min(mask.shape[2] - 1.0, ceil(cx + ex))
    cdef double ly = max(0.0, floor(cy - ey)), hy = min(mask.shape[1] - 1.0, ceil(cy + ey))
    cdef double lz = max(0.0, floor(cz - ez)), hz = min(mask.shape[0] - 1.0, ceil(cz + ez))
    if lx > hx or ly > hy or lz > hz:
        return None
    cdef Py_ssize_t x0 = <Py_ssize_t>lx, x1 = <Py_ssize_t>hx
    cdef Py_ssize_t y0 = <Py_ssize_t>ly, y1 = <Py_ssize_t>hy
    cdef Py_ssize_t z0 = <Py_ssize_t>lz, z1 = <Py_ssize_t>hz
    cdef Py_ssize_t xmin = x1 + 1, xmax = -1
    cdef Py_ssize_t ymin = y1 + 1, ymax = -1
    cdef Py_ssize_t zmin = z1 + 1, zmax = -1
    cdef Py_ssize_t x, y, z
    cdef double dx, dy, dz, local_x, local_y, local_z
    cdef double top = h - 0.5
    cdef double inv_rx = 1.0 / rx, inv_ry = 1.0 / ry
    with nogil:
        for z in range(z0, z1 + 1):
            dz = z - sz
            for y in range(y0, y1 + 1):
                dy = y - sy
                for x in range(x0, x1 + 1):
                    dx = x - sx
                    local_z = wx * dx + wy * dy + wz * dz
                    if local_z < -0.5 or local_z > top:
                        continue
                    local_x = (ux * dx + uy * dy) * inv_rx
                    local_y = (vx * dx + vy * dy + vz * dz) * inv_ry
                    if local_x * local_x + local_y * local_y <= 1.0:
                        mask[z, y, x] = 1
                        # Include geometric coverage even if the voxel was already set.
                        if x < xmin: xmin = x
                        if x > xmax: xmax = x
                        if y < ymin: ymin = y
                        if y > ymax: ymax = y
                        if z < zmin: zmin = z
                        if z > zmax: zmax = z
    if xmax < 0:
        return None
    return [xmin, xmax, ymin, ymax, zmin, zmax]
