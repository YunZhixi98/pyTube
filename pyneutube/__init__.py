"""Public package interface for PyNeuTube."""

from __future__ import annotations

from ._version import __version__
from .io import ImageParser, Neuron
from .processing import (
    connectivity_filter,
    estimate_background_level,
    local_max_filter,
    refine_local_max_threshold,
    subtract_background,
    threshold_filter,
    triangle_threshold,
)
from .tracing import (
    PreprocessedVolume,
    SUPPORTED_IMAGE_SUFFIXES,
    connect_trace_chains,
    extract_trace_seeds,
    generate_trace_chains,
    load_trace_stage,
    preprocess_volume,
    save_trace_stage,
    trace_file,
    trace_volume,
    trace_files,
    trace_directory,
)
from .visualization import save_chain_overlay_figure, save_overlay_figure, save_seed_overlay_figure

__all__ = [
    "__version__",
    "trace_file",
    "trace_volume",
    "trace_files",
    "trace_directory",
    "save_overlay_figure",
    "save_seed_overlay_figure",
    "save_chain_overlay_figure",
    "ImageParser",
    "Neuron",
    "PreprocessedVolume",
    "preprocess_volume",
    "extract_trace_seeds",
    "generate_trace_chains",
    "connect_trace_chains",
    "save_trace_stage",
    "load_trace_stage",
    "SUPPORTED_IMAGE_SUFFIXES",
    "subtract_background",
    "estimate_background_level",
    "threshold_filter",
    "triangle_threshold",
    "refine_local_max_threshold",
    "local_max_filter",
    "connectivity_filter",
]
