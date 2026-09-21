import numpy as np

from .tracing_mask_accel import label_segment


def label_tracing_mask(
    seg,
    trace_mask: np.ndarray,
    dilate: bool,
    *,
    start: int | None = None,
    end: int | None = None,
):
    """Label segment interiors at voxel centers in a uint8 tracing mask."""
    if hasattr(seg, "_segments") and hasattr(seg, "label_bboxes"):
        if start is None:
            start = 0
        if end is None:
            end = len(seg) - 1

        if seg.label_bboxes is None or seg.label_bboxes.shape != (len(seg), 6):
            seg.label_bboxes = np.full((len(seg), 6), -1, dtype=np.intp)
        for idx in range(start, end + 1):
            bbox = label_segment(seg[idx], trace_mask, dilate)
            seg.label_bboxes[idx] = -1 if bbox is None else bbox
        return trace_mask

    if start is not None or end is not None:
        raise ValueError("Segment subrange is only supported for SegmentChain inputs.")

    label_segment(seg, trace_mask, dilate)
    return trace_mask
