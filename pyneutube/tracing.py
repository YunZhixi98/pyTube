"""Public tracing API and high-level pipeline helpers."""

from __future__ import annotations

import json
import os
import pickle
import threading
import traceback
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import import_module
from multiprocessing import Manager, get_context
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from queue import Empty
from time import perf_counter
from types import ModuleType
from typing import Any, Callable

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from tqdm import tqdm

from pyneutube.core.io.image_parser import ImageParser
from pyneutube.core.io.swc_parser import Neuron
from pyneutube.core.processing.filtering import (
    connectivity_filter,
    estimate_background_level,
    local_max_filter,
    refine_local_max_threshold,
    threshold_filter,
    triangle_threshold,
)
from pyneutube.tracers.pyNeuTube.chains_to_morphology import ChainConnector
from pyneutube.tracers.pyNeuTube.config import Defaults as TraceDefaults
from pyneutube.tracers.pyNeuTube.config import Optimization as TraceOptimization
from pyneutube.tracers.pyNeuTube.seeds import Seed, Seeds
from pyneutube.tracers.pyNeuTube.tracing import SegmentChain, SegmentChains, TracingSegment
from pyneutube.visualization import (
    _project_overlay_image,
    save_chain_overlay_figure,
    save_overlay_figure,
    save_seed_overlay_figure,
)

SUPPORTED_IMAGE_SUFFIXES = (
    ".tif",
    ".tiff",
    ".v3draw",
    ".raw",
    ".v3dpbd",
    ".nii",
    ".nii.gz",
    ".nrrd",
    ".nhdr",
)

_TRACE_PROGRESS_STAGES = (
    "load_image",
    "preprocess_volume",
    "generate_tracing_seeds",
    "generate_neuron_trace",
    "reconstruct",
)
_TRACE_PROGRESS_STAGE_INDEX = {stage: index + 1 for index, stage in enumerate(_TRACE_PROGRESS_STAGES)}
_TRACE_PROGRESS_STAGE_LABELS = {
    "load_image": "load image",
    "preprocess_volume": "preprocess volume",
    "generate_tracing_seeds": "generate tracing seeds",
    "generate_neuron_trace": "generate trace chains",
    "reconstruct": "reconstruct morphology",
}
_TRACE_PROGRESS_REFRESH_EVERY = 100
_TRACE_PROGRESS_MIN_INTERVAL = 0.2
_MORPHOLOGY_STRUCTURE = np.ones((3, 3, 3), dtype=bool)


@dataclass
class TracingResult:
    image_path: Path | None
    threshold: float
    seeds: Seeds
    chains: SegmentChains
    neuron: Neuron | None = None
    pre_postprocess_neuron: Neuron | None = None
    output_swc: Path | None = None
    output_visualization: Path | None = None
    output_seed_visualization: Path | None = None
    output_chain_visualization: Path | None = None
    output_pre_postprocess_visualization: Path | None = None
    signal_image: np.ndarray | None = None
    binary_image: np.ndarray | None = None
    skipped: bool = False
    skip_reason: str | None = None


@dataclass
class PreprocessedVolume:
    threshold: float


class TraceTimeoutError(TimeoutError):
    def __init__(self, timeout_seconds: float, stage: str | None = None) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self.stage = stage
        if stage:
            message = f"Tracing timed out after {self.timeout_seconds:g} s during {stage}."
        else:
            message = f"Tracing timed out after {self.timeout_seconds:g} s."
        super().__init__(message)


def _vprint(verbose: int, message: str) -> None:
    if verbose:
        print(message)


def _time_step(verbose: int, label: str, started_at: float) -> None:
    if verbose:
        print(f"{label}: {perf_counter() - started_at:.3f}s")


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    if n_jobs <= 0:
        raise ValueError("`n_jobs` must be a positive integer or -1.")
    return n_jobs


def _resolve_seed_strategy(seed_strategy: str | None) -> str:
    strategy = "eager" if seed_strategy is None else seed_strategy
    strategy = str(strategy).lower()
    if strategy not in {"eager", "lazy"}:
        raise ValueError("`seed_strategy` must be one of {'eager', 'lazy'}.")
    return strategy


def _resolve_chain_seed_strategy(seed_strategy: str | None, seeds: Seeds) -> str:
    strategy = "auto" if seed_strategy is None else str(seed_strategy).lower()
    if strategy == "auto":
        return "lazy" if not getattr(seeds, "scored", True) else "eager"
    return _resolve_seed_strategy(strategy)


def _make_timeout_checker(
    timeout: float | None,
) -> tuple[float | None, Callable[[str | None], None]]:
    if timeout is None:
        return None, (lambda stage=None: None)

    timeout_seconds = float(timeout)
    if timeout_seconds <= 0:
        raise ValueError("`timeout` must be a positive number of seconds or None.")

    deadline = perf_counter() + timeout_seconds

    def _check_timeout(stage: str | None = None) -> None:
        if perf_counter() > deadline:
            raise TraceTimeoutError(timeout_seconds, stage)

    return timeout_seconds, _check_timeout


def _iso_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _overwrite_error(path: Path, *, mode: str) -> FileExistsError:
    if mode == "single":
        return FileExistsError(f"Output already exists: {path}. Pass overwrite=True to replace it.")
    return FileExistsError(
        f"Output already exists: {path}. Pass overwrite=True or --overwrite to replace it."
    )


def _resolve_on_exists(on_exists: str | None, *, default: str) -> str:
    policy = default if on_exists is None else str(on_exists).lower()
    if policy not in {"error", "skip"}:
        raise ValueError("`on_exists` must be one of {'error', 'skip'}.")
    return policy


def _skipped_trace_result(
    image_path: Path,
    *,
    output_swc: Path | None = None,
    reason: str,
) -> TracingResult:
    return TracingResult(
        image_path=image_path,
        threshold=float("nan"),
        seeds=Seeds(),
        chains=SegmentChains(),
        neuron=None,
        output_swc=output_swc,
        skipped=True,
        skip_reason=reason,
    )


def _matches_suffix(path: Path, suffixes: Sequence[str]) -> bool:
    lower_name = path.name.lower()
    return any(lower_name.endswith(suffix.lower()) for suffix in suffixes)


def _visualization_output_path(
    image_path: Path,
    visualization_dir: str | Path | None,
    kind: str = "result",
) -> Path | None:
    if visualization_dir is None:
        return None
    return Path(visualization_dir) / kind / f"{image_path.name}.png"


def _load_image_volume(image: np.ndarray | str | Path) -> np.ndarray:
    if isinstance(image, (str, Path)):
        return ImageParser(image).load()
    return np.asarray(image)


def _emit_trace_progress(
    progress_callback: Callable[[str, int | None, int | None], None] | None,
    stage: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if progress_callback is not None:
        progress_callback(stage, current, total)


def _safe_tqdm_close(bar) -> None:
    if bar is None:
        return
    try:
        bar.close()
    except AttributeError:
        pass


def _batch_pool_context(*, show_progress: bool):
    if os.name != "posix":
        return None
    # Forking a multi-threaded parent can deadlock workers when verbose mode
    # has already started tqdm and progress-drain threads.
    start_method = "spawn" if show_progress else "fork"
    return get_context(start_method)


class _QueuedBatchProgressReporter:
    def __init__(
        self,
        input_path: str,
        progress_queue,
        *,
        refresh_every: int = _TRACE_PROGRESS_REFRESH_EVERY,
        min_interval: float = _TRACE_PROGRESS_MIN_INTERVAL,
        timer: Callable[[], float] = perf_counter,
    ) -> None:
        self._input_path = str(input_path)
        self._progress_queue = progress_queue
        self._refresh_every = max(1, int(refresh_every))
        self._min_interval = max(0.0, float(min_interval))
        self._timer = timer
        self._last_stage: str | None = None
        self._last_current: int | None = None
        self._last_total: int | None = None
        self._last_emit_at: float | None = None

    def emit(self, stage: str, current: int | None = None, total: int | None = None) -> None:
        stage_name = str(stage)
        current_value = None if current is None else int(current)
        total_value = None if total is None else int(total)
        now = self._timer()

        should_emit = (
            self._last_stage != stage_name
            or self._last_total != total_value
            or current_value is None
            or total_value is None
        )

        if current_value is not None and total_value is not None:
            if current_value <= 0 or current_value >= total_value:
                should_emit = True
            elif self._last_current is None or current_value < self._last_current:
                should_emit = True
            elif current_value - self._last_current >= self._refresh_every:
                should_emit = True
            elif self._last_emit_at is None or now - self._last_emit_at >= self._min_interval:
                should_emit = True

        if not should_emit:
            return

        self._progress_queue.put((self._input_path, stage_name, current_value, total_value))
        self._last_stage = stage_name
        self._last_current = current_value
        self._last_total = total_value
        self._last_emit_at = now


def _signal_dtype(source: np.ndarray) -> type[np.floating]:
    """
    Pick the narrowest float that stores the background-subtracted volume exactly.

    8/16-bit integers stay within [-32768, 65535], so every value is an integer
    below 2**24 that float32 holds exactly, and the kernels reading the volume
    widen each sample to double anyway. Wider integers can exceed 2**24 and a
    float input would be rounded, so both keep double precision.
    """
    dtype = source.dtype
    if dtype.kind in "iu" and dtype.itemsize <= 2:
        return np.float32
    return np.float64


def _prepare_signal_image(image: np.ndarray, *, verbose: int = 1) -> np.ndarray:
    t0 = perf_counter()
    source = np.asarray(image)
    # Histogram the source in its native dtype, then subtract in place;
    # `subtract_background` would allocate a second full-volume temporary.
    background_level = estimate_background_level(source, verbose=max(verbose - 1, 0))
    signal_image = source.astype(_signal_dtype(source), order="C")
    if background_level:
        signal_image -= background_level
    # `subtract_background` only clips when the level exceeds the image minimum;
    # clipping unconditionally is the same, since otherwise nothing went negative.
    np.maximum(signal_image, 0.0, out=signal_image)
    _time_step(verbose, "subtract_background", t0)
    return signal_image


def _build_binary_image(
    signal_image: np.ndarray,
    threshold: float,
    *,
    verbose: int = 1,
) -> np.ndarray:
    t0 = perf_counter()
    binary_image = threshold_filter(signal_image, float(threshold))
    binary_image = connectivity_filter(binary_image, 4, n_neighbors=26)
    binary_image = binary_dilation(
        binary_image.astype(bool, copy=False),
        structure=_MORPHOLOGY_STRUCTURE,
        border_value=0,
    )
    binary_image = binary_erosion(binary_image, structure=_MORPHOLOGY_STRUCTURE, border_value=1)
    _time_step(verbose, "binary_mask", t0)
    return np.ascontiguousarray(binary_image, dtype=np.uint8)


def _estimate_threshold(signal_image: np.ndarray, *, verbose: int = 1) -> float:
    t0 = perf_counter()
    local_max_mask = local_max_filter(signal_image)
    _time_step(verbose, "local_max_filter", t0)
    local_max_values = signal_image[local_max_mask.view(bool)]
    if local_max_values.size == 0:
        raise ValueError("No local maxima were found in the input volume.")

    if np.ptp(local_max_values) == 0:
        threshold = max(float(local_max_values[0]) - 1.0, 0.0)
        _vprint(
            verbose,
            f"triangle_threshold skipped; using constant-value fallback {threshold:.3f}",
        )
        return threshold

    t0 = perf_counter()
    threshold = float(
        triangle_threshold(
            local_max_values,
            max_height_value=int(np.max(local_max_values)) - 1,
        )
    )
    _time_step(verbose, f"triangle_threshold={threshold:.3f}", t0)

    t0 = perf_counter()
    threshold = float(
        refine_local_max_threshold(
            signal_image,
            threshold,
            threshold_source=local_max_values,
        )
    )
    _time_step(verbose, f"refine_local_max_threshold={threshold:.3f}", t0)
    return threshold


def _resolve_config_module(config: str | ModuleType | None) -> ModuleType | None:
    if config is None:
        return None
    if isinstance(config, ModuleType):
        return config
    if config == "default":
        return None
    return import_module(config)


def _iter_config_attributes(source: type[Any]) -> list[tuple[str, Any]]:
    return [
        (name, value)
        for name, value in vars(source).items()
        if not name.startswith("_") and not callable(value)
    ]


@contextmanager
def _temporary_trace_config(config: str | ModuleType | None):
    module = _resolve_config_module(config)
    if module is None:
        yield
        return

    overrides: list[tuple[type[Any], str, Any]] = []
    for source_name, target in (
        ("Defaults", TraceDefaults),
        ("Optimization", TraceOptimization),
    ):
        source = getattr(module, source_name, None)
        if source is None:
            continue
        for attr_name, attr_value in _iter_config_attributes(source):
            if hasattr(target, attr_name):
                overrides.append((target, attr_name, getattr(target, attr_name)))
                setattr(target, attr_name, attr_value)

    try:
        yield
    finally:
        for target, attr_name, attr_value in reversed(overrides):
            setattr(target, attr_name, attr_value)


@contextmanager
def _shared_array(image: np.ndarray):
    array = np.ascontiguousarray(np.asarray(image))
    shm = SharedMemory(create=True, size=array.nbytes)
    shared = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared[...] = array
    spec = (shm.name, array.shape, array.dtype.str)
    try:
        yield shared, spec
    finally:
        shm.close()
        shm.unlink()


def save_trace_stage(stage_data: object, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(stage_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path


def load_trace_stage(input_path: str | Path) -> object:
    with Path(input_path).open("rb") as handle:
        return pickle.load(handle)


def preprocess_volume(
    image: np.ndarray,
    *,
    verbose: int = 1,
    output_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    signal_image = _prepare_signal_image(image, verbose=verbose)
    threshold = _estimate_threshold(signal_image, verbose=verbose)
    binary_image = _build_binary_image(signal_image, threshold, verbose=verbose)
    if output_path is not None:
        save_trace_stage(PreprocessedVolume(threshold=threshold), output_path)

    return signal_image, binary_image, threshold


def trace_volume(
    image: np.ndarray,
    *,
    n_jobs: int = 1,
    timeout: float | None = None,
    verbose: int = 1,
    return_intermediates: bool = False,
    config: str | ModuleType | None = None,
    seed_strategy: str | None = None,
) -> TracingResult:
    return _trace_volume_internal(
        image,
        n_jobs=n_jobs,
        timeout=timeout,
        verbose=verbose,
        return_intermediates=return_intermediates,
        config=config,
        seed_strategy=seed_strategy,
    )


def _trace_volume_internal(
    image: np.ndarray,
    *,
    n_jobs: int = 1,
    timeout: float | None = None,
    verbose: int = 1,
    max_seeds: int | None = None,
    connect_chains: bool = True,
    filter_chains: bool = True,
    return_intermediates: bool = False,
    config: str | ModuleType | None = None,
    seed_strategy: str | None = None,
    progress_callback: Callable[[str, int | None, int | None], None] | None = None,
) -> TracingResult:
    with _temporary_trace_config(config):
        resolved_n_jobs = _resolve_n_jobs(n_jobs)
        resolved_seed_strategy = _resolve_seed_strategy(seed_strategy)
        _, check_timeout = _make_timeout_checker(timeout)

        check_timeout("preprocess_volume")
        _emit_trace_progress(progress_callback, "preprocess_volume")
        signal_image, binary_image, threshold = preprocess_volume(image, verbose=verbose)
        check_timeout("preprocess_volume")

        def _run_trace(
            active_signal_image: np.ndarray,
            active_binary_image: np.ndarray,
            shared_image_spec=None,
        ) -> TracingResult:
            t0 = perf_counter()
            seeds = Seeds()
            _emit_trace_progress(progress_callback, "generate_tracing_seeds")
            if resolved_seed_strategy == "lazy":
                seeds.generate_seed_candidates(
                    active_binary_image,
                    n_jobs=resolved_n_jobs,
                    verbose=verbose,
                    check_timeout=check_timeout,
                )
            else:
                seeds.generate_tracing_seeds(
                    active_signal_image,
                    active_binary_image,
                    n_jobs=resolved_n_jobs,
                    verbose=verbose,
                    check_timeout=check_timeout,
                    shared_image=shared_image_spec,
                    progress_callback=progress_callback,
                )
            _time_step(verbose, "generate_tracing_seeds", t0)
            check_timeout("generate_tracing_seeds")

            binary_image_result = active_binary_image if return_intermediates else None
            if not return_intermediates:
                active_binary_image = None

            t0 = perf_counter()
            chains = SegmentChains(image_shape=active_signal_image.shape)
            _emit_trace_progress(progress_callback, "generate_neuron_trace")
            if resolved_seed_strategy == "lazy":
                seeds = Seeds(
                    chains.generate_neuron_trace_lazy_seed_scoring(
                        seeds,
                        active_signal_image,
                        max_seeds=max_seeds,
                        verbose=verbose,
                        check_timeout=check_timeout,
                        progress_callback=progress_callback,
                    ),
                    seed_strategy="lazy",
                    scored=True,
                )
            else:
                chains.generate_neuron_trace(
                    seeds,
                    active_signal_image,
                    max_seeds=max_seeds,
                    verbose=verbose,
                    check_timeout=check_timeout,
                    progress_callback=progress_callback,
                )
            if filter_chains:
                check_timeout("filter_chains")
                chains.filter_chains(verbose=verbose)
            _time_step(verbose, "generate_neuron_trace", t0)
            check_timeout("generate_neuron_trace")

            neuron = None
            pre_postprocess_neuron = None
            if connect_chains:
                t0 = perf_counter()
                connector = ChainConnector(
                    verbose=verbose,
                    enable_crossover_test=getattr(TraceDefaults, "CROSSOVER_TEST", False),
                )
                _emit_trace_progress(progress_callback, "reconstruct")
                reconstruct_result = connector.reconstruct(
                    chains,
                    active_signal_image,
                    check_timeout=check_timeout,
                    return_pre_postprocess_neuron=True,
                )
                if isinstance(reconstruct_result, tuple):
                    neuron, pre_postprocess_neuron = reconstruct_result
                else:
                    neuron = reconstruct_result
                _time_step(verbose, "reconstruct", t0)
                check_timeout("reconstruct")

            signal_image_result = (
                np.asarray(active_signal_image).copy() if return_intermediates else None
            )
            return TracingResult(
                image_path=None,
                threshold=threshold,
                seeds=seeds,
                chains=chains,
                neuron=neuron,
                pre_postprocess_neuron=pre_postprocess_neuron,
                signal_image=signal_image_result,
                binary_image=binary_image_result,
            )

        if resolved_n_jobs > 1 and resolved_seed_strategy == "eager":
            with _shared_array(signal_image) as (shared_signal_image, shared_image_spec):
                del signal_image
                return _run_trace(shared_signal_image, binary_image, shared_image_spec)

        return _run_trace(signal_image, binary_image)


def trace_file(
    image_path: str | Path,
    *,
    output_swc: str | Path | None = None,
    visualization_dir: str | Path | None = None,
    n_jobs: int = 1,
    timeout: float | None = None,
    verbose: int = 1,
    overwrite: bool = False,
    on_exists: str | None = None,
    return_intermediates: bool = False,
    config: str | ModuleType | None = None,
    seed_strategy: str | None = None,
) -> TracingResult:
    return _trace_file_internal(
        image_path,
        output_swc=output_swc,
        visualization_dir=visualization_dir,
        n_jobs=n_jobs,
        timeout=timeout,
        verbose=verbose,
        overwrite=overwrite,
        on_exists=on_exists,
        return_intermediates=return_intermediates,
        config=config,
        seed_strategy=seed_strategy,
    )


def _trace_file_internal(
    image_path: str | Path,
    *,
    output_swc: str | Path | None = None,
    visualization_dir: str | Path | None = None,
    n_jobs: int = 1,
    timeout: float | None = None,
    verbose: int = 1,
    overwrite: bool = False,
    on_exists: str | None = None,
    max_seeds: int | None = None,
    connect_chains: bool = True,
    filter_chains: bool = True,
    return_intermediates: bool = False,
    config: str | ModuleType | None = None,
    seed_strategy: str | None = None,
    progress_callback: Callable[[str, int | None, int | None], None] | None = None,
) -> TracingResult:
    _, check_timeout = _make_timeout_checker(timeout)
    image_path = Path(image_path)
    output_swc_path = Path(output_swc) if output_swc is not None else None
    output_visualization_path = _visualization_output_path(image_path, visualization_dir, "result")
    output_seed_visualization_path = _visualization_output_path(image_path, visualization_dir, "seeds")
    output_chain_visualization_path = _visualization_output_path(image_path, visualization_dir, "chains")
    output_pre_postprocess_visualization_path = _visualization_output_path(
        image_path,
        visualization_dir,
        "pre_postprocess",
    )
    existing_output = None
    if output_swc_path is not None and output_swc_path.exists():
        existing_output = output_swc_path
    elif output_visualization_path is not None and output_visualization_path.exists():
        existing_output = output_visualization_path
    elif output_seed_visualization_path is not None and output_seed_visualization_path.exists():
        existing_output = output_seed_visualization_path
    elif output_chain_visualization_path is not None and output_chain_visualization_path.exists():
        existing_output = output_chain_visualization_path
    elif (
        output_pre_postprocess_visualization_path is not None
        and output_pre_postprocess_visualization_path.exists()
    ):
        existing_output = output_pre_postprocess_visualization_path

    if existing_output is not None and not overwrite:
        exists_policy = _resolve_on_exists(on_exists, default="error")
        if exists_policy == "error":
            raise _overwrite_error(existing_output, mode="single")
        _vprint(
            verbose,
            f"Skipped {image_path.name} (exists; use overwrite=True to replace)",
        )
        return _skipped_trace_result(
            image_path,
            output_swc=output_swc_path,
            reason="exists",
        )

    parser = ImageParser(image_path, verbose=verbose)

    t0 = perf_counter()
    check_timeout("load_image")
    _emit_trace_progress(progress_callback, "load_image")
    image = parser.load()
    _time_step(verbose, "load_image", t0)
    check_timeout("load_image")

    result = _trace_volume_internal(
        image,
        n_jobs=n_jobs,
        timeout=timeout,
        verbose=verbose,
        max_seeds=max_seeds,
        connect_chains=connect_chains,
        filter_chains=filter_chains,
        return_intermediates=return_intermediates,
        config=config,
        seed_strategy=seed_strategy,
        progress_callback=progress_callback,
    )
    result.image_path = image_path

    if output_swc is not None:
        if result.neuron is None:
            raise ValueError("No neuron reconstruction is available for SWC export.")
        result.output_swc = output_swc_path
        result.neuron.save_swc(result.output_swc, verbose=verbose)

    projected_image = None
    if (
        output_seed_visualization_path is not None
        or output_chain_visualization_path is not None
        or output_visualization_path is not None
        or output_pre_postprocess_visualization_path is not None
    ):
        projected_image = _project_overlay_image(image, log_transform=True)

    if output_seed_visualization_path is not None and len(result.seeds) > 0:
        result.output_seed_visualization = save_seed_overlay_figure(
            image,
            result.seeds,
            output_seed_visualization_path,
            title=f"{image_path.name} seeds",
            projected_image=projected_image,
        )

    if output_chain_visualization_path is not None and len(result.chains) > 0:
        result.output_chain_visualization = save_chain_overlay_figure(
            image,
            result.chains,
            output_chain_visualization_path,
            title=f"{image_path.name} chains",
            projected_image=projected_image,
        )

    if (
        output_pre_postprocess_visualization_path is not None
        and result.pre_postprocess_neuron is not None
    ):
        result.output_pre_postprocess_visualization = save_overlay_figure(
            image,
            result.pre_postprocess_neuron,
            output_pre_postprocess_visualization_path,
            title=f"{image_path.name} pre-postprocess",
            projected_image=projected_image,
        )

    if output_visualization_path is not None:
        if result.neuron is not None:
            result.output_visualization = save_overlay_figure(
                image,
                result.neuron,
                output_visualization_path,
                title=image_path.name,
                projected_image=projected_image,
            )
        else:
            chain_coords = [chain.to_coords() for chain in result.chains if len(chain) > 0]
            if chain_coords:
                coords = np.concatenate(chain_coords, axis=0)
                result.output_visualization = save_overlay_figure(
                    image,
                    coords,
                    output_visualization_path,
                    title=image_path.name,
                    projected_image=projected_image,
                )
            else:
                raise ValueError("No coordinates are available for visualization export.")

    return result


def _trace_file_record(
    input_path: str,
    output_swc: str,
    visualization_dir: str | None,
    n_jobs: int,
    timeout: float | None,
    verbose: int,
    overwrite: bool,
    config: str | None,
    seed_strategy: str | None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, object]:
    started_at = perf_counter()
    try:
        if progress_callback is None:
            result = trace_file(
                input_path,
                output_swc=output_swc,
                visualization_dir=visualization_dir,
                n_jobs=n_jobs,
                timeout=timeout,
                verbose=verbose,
                overwrite=overwrite,
                config=config,
                seed_strategy=seed_strategy,
            )
        else:
            result = _trace_file_internal(
                input_path,
                output_swc=output_swc,
                visualization_dir=visualization_dir,
                n_jobs=n_jobs,
                timeout=timeout,
                verbose=verbose,
                overwrite=overwrite,
                config=config,
                seed_strategy=seed_strategy,
                progress_callback=progress_callback,
            )
    except Exception as exc:
        timed_out = isinstance(exc, TraceTimeoutError)
        return {
            "timestamp_utc": _iso_timestamp(),
            "status": "timed_out" if timed_out else "failed",
            "input_path": input_path,
            "output_swc": output_swc,
            "elapsed_seconds": perf_counter() - started_at,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "timeout_seconds": exc.timeout_seconds if timed_out else None,
        }

    return {
        "timestamp_utc": _iso_timestamp(),
        "status": "completed",
        "input_path": input_path,
        "output_swc": str(result.output_swc) if result.output_swc is not None else None,
        "elapsed_seconds": perf_counter() - started_at,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "timeout_seconds": None,
    }


def _trace_file_worker(
    payload: tuple[str, str, str | None, int, float | None, int, bool, str | None, str | None, object | None],
) -> dict[str, object]:
    input_path, output_swc, visualization_dir, n_jobs, timeout, verbose, overwrite, config, seed_strategy, progress_queue = payload

    progress_callback = None
    if progress_queue is not None:
        reporter = _QueuedBatchProgressReporter(input_path, progress_queue)

        def progress_callback(stage: str, current: int | None = None, total: int | None = None) -> None:
            reporter.emit(stage, current, total)

    return _trace_file_record(
        input_path,
        output_swc,
        visualization_dir,
        n_jobs,
        timeout,
        verbose,
        overwrite,
        config,
        seed_strategy,
        progress_callback=progress_callback,
    )


def _append_manifest_record(
    manifest_path: Path | None,
    record: dict[str, object],
) -> None:
    if manifest_path is None:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _skip_existing_record(input_path: Path, output_swc: Path) -> dict[str, object]:
    return {
        "timestamp_utc": _iso_timestamp(),
        "status": "skipped",
        "input_path": str(input_path),
        "output_swc": str(output_swc),
        "elapsed_seconds": 0.0,
        "error_type": None,
        "error_message": None,
        "traceback": None,
        "reason": "exists",
        "timeout_seconds": None,
    }


def trace_files(
    input_paths: Sequence[str | Path],
    output_dir: str | Path,
    *,
    visualization_dir: str | Path | None = None,
    batch_n_jobs: int = 1,
    trace_n_jobs: int = 1,
    trace_timeout: float | None = None,
    verbose: int = 1,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
    on_exists: str | None = None,
    config: str | None = None,
    seed_strategy: str | None = None,
) -> list[Path]:
    image_paths = [Path(path) for path in input_paths]
    if not image_paths:
        raise FileNotFoundError("No input image files were provided.")

    missing_paths = [path for path in image_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Input image file does not exist: {missing_paths[0]}")

    unsupported_paths = [path for path in image_paths if not _matches_suffix(path, SUPPORTED_IMAGE_SUFFIXES)]
    if unsupported_paths:
        raise ValueError(f"Unsupported image file suffix: {unsupported_paths[0]}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_batch_n_jobs = _resolve_n_jobs(batch_n_jobs)
    resolved_trace_n_jobs = _resolve_n_jobs(trace_n_jobs)
    resolved_seed_strategy = _resolve_seed_strategy(seed_strategy)
    exists_policy = _resolve_on_exists(on_exists, default="skip")
    manifest = Path(manifest_path) if manifest_path is not None else None

    _vprint(
        verbose,
        (
            f"Tracing {len(image_paths)} image(s) "
            f"with batch_n_jobs={resolved_batch_n_jobs}, trace_n_jobs={resolved_trace_n_jobs}"
        ),
    )

    show_progress = verbose >= 1
    completed_outputs: list[Path] = []
    jobs: list[tuple[str, str, str | None, int, float | None, int, bool, str | None, str | None]] = []
    progress = tqdm(
        total=len(image_paths),
        desc="Tracing files",
        unit="file",
        disable=not show_progress,
    )
    try:
        for image_path in image_paths:
            output_swc = output_dir / f"{image_path.name}.swc"
            output_visualization = _visualization_output_path(image_path, visualization_dir, "result")
            output_seed_visualization = _visualization_output_path(image_path, visualization_dir, "seeds")
            output_chain_visualization = _visualization_output_path(image_path, visualization_dir, "chains")
            output_pre_postprocess_visualization = _visualization_output_path(
                image_path,
                visualization_dir,
                "pre_postprocess",
            )
            has_existing_output = output_swc.exists() or any(
                path is not None and path.exists()
                for path in (
                    output_visualization,
                    output_seed_visualization,
                    output_chain_visualization,
                    output_pre_postprocess_visualization,
                )
            )
            if has_existing_output and not overwrite:
                existing_path = next(
                    path
                    for path in (
                        output_swc,
                        output_visualization,
                        output_seed_visualization,
                        output_chain_visualization,
                        output_pre_postprocess_visualization,
                    )
                    if path is not None and path.exists()
                )
                if exists_policy == "error":
                    raise _overwrite_error(existing_path, mode="batch")
                completed_outputs.append(output_swc)
                _append_manifest_record(manifest, _skip_existing_record(image_path, output_swc))
                if not show_progress:
                    _vprint(
                        verbose,
                        f"Skipped {image_path.name} -> {output_swc.name} (exists; use overwrite=True to replace)",
                    )
                progress.update(1)
                continue
            jobs.append(
                (
                    str(image_path),
                    str(output_swc),
                    None if visualization_dir is None else str(visualization_dir),
                    resolved_trace_n_jobs,
                    trace_timeout,
                    0,
                    overwrite,
                    config,
                    resolved_seed_strategy,
                )
            )

        def emit_batch_message(level: int, message: str) -> None:
            if show_progress:
                progress.write(message)
            else:
                _vprint(level, message)

        def handle_record(record: dict[str, object]) -> None:
            status = str(record["status"])
            input_path = Path(str(record["input_path"]))
            output_swc_value = record["output_swc"]
            output_swc = Path(str(output_swc_value)) if isinstance(output_swc_value, str) else None
            _append_manifest_record(manifest, record)
            if status == "completed":
                if output_swc is not None:
                    completed_outputs.append(output_swc)
                    if not show_progress:
                        _vprint(verbose, f"Completed {input_path.name} -> {output_swc.name}")
                elif not show_progress:
                    _vprint(verbose, f"Completed {input_path.name}")
                return
            if status == "skipped":
                if not show_progress:
                    if output_swc is not None:
                        _vprint(verbose, f"Skipped {input_path.name} -> {output_swc.name}")
                    else:
                        _vprint(verbose, f"Skipped {input_path.name}")
                return
            if status == "timed_out":
                timeout_seconds = record.get("timeout_seconds")
                timeout_suffix = (
                    f" (timeout={float(timeout_seconds):g} s)"
                    if isinstance(timeout_seconds, (int, float))
                    else ""
                )
                emit_batch_message(
                    max(verbose, 1),
                    f"Timed out {input_path.name}: {record['error_type']}: {record['error_message']}{timeout_suffix}",
                )
                return

            emit_batch_message(
                max(verbose, 1),
                f"Failed {input_path.name}: {record['error_type']}: {record['error_message']}",
            )

        if not jobs:
            return sorted(set(completed_outputs))

        if resolved_batch_n_jobs == 1:
            for job in jobs:
                stage_progress = None
                detail_progress = None
                stage_state = {"index": 0}
                detail_state = {"stage": None, "current": 0, "total": 0}
                if show_progress:
                    input_path = Path(job[0])
                    stage_progress = tqdm(
                        total=len(_TRACE_PROGRESS_STAGES),
                        desc=input_path.name,
                        unit="stage",
                        disable=False,
                        leave=False,
                    )

                    def progress_callback(
                        stage: str,
                        current: int | None = None,
                        total: int | None = None,
                        *,
                        bar=stage_progress,
                        state=stage_state,
                        input_name=input_path.name,
                    ) -> None:
                        nonlocal detail_progress
                        stage_index = _TRACE_PROGRESS_STAGE_INDEX.get(stage)
                        if stage_index is None:
                            return
                        delta = stage_index - state["index"]
                        if delta > 0:
                            bar.update(delta)
                            state["index"] = stage_index
                        bar.set_postfix_str(_TRACE_PROGRESS_STAGE_LABELS.get(stage, stage))

                        detail_label = _TRACE_PROGRESS_STAGE_LABELS.get(stage, stage)
                        if total is None:
                            if detail_progress is not None and detail_state["stage"] != stage:
                                _safe_tqdm_close(detail_progress)
                                detail_progress = None
                                detail_state["stage"] = None
                                detail_state["current"] = 0
                                detail_state["total"] = 0
                            return

                        if detail_progress is None or detail_state["stage"] != stage or detail_state["total"] != total:
                            if detail_progress is not None:
                                _safe_tqdm_close(detail_progress)
                            detail_progress = tqdm(
                                total=total,
                                desc=f"{input_name}: {detail_label}",
                                unit="item",
                                disable=False,
                                leave=False,
                            )
                            detail_state["stage"] = stage
                            detail_state["current"] = 0
                            detail_state["total"] = total

                        current_value = max(0, min(int(current or 0), int(total)))
                        delta_items = current_value - int(detail_state["current"])
                        if delta_items >= _TRACE_PROGRESS_REFRESH_EVERY or current_value >= int(total):
                            detail_progress.update(delta_items)
                            detail_state["current"] = current_value

                        if current_value >= int(total):
                            _safe_tqdm_close(detail_progress)
                            detail_progress = None
                            detail_state["stage"] = None
                            detail_state["current"] = 0
                            detail_state["total"] = 0

                try:
                    if show_progress:
                        record = _trace_file_record(*job, progress_callback=progress_callback)
                    else:
                        record = _trace_file_worker(job + (None,))
                    handle_record(record)
                finally:
                    _safe_tqdm_close(detail_progress)
                    _safe_tqdm_close(stage_progress)
                progress.update(1)
            return sorted(set(completed_outputs))

        progress_queue = None
        manager = None
        progress_thread = None
        stop_progress = threading.Event()
        mp_context = _batch_pool_context(show_progress=show_progress)
        if show_progress:
            manager = Manager()
            progress_queue = manager.Queue()

            def drain_progress_queue():
                slot_count = min(resolved_batch_n_jobs, len(jobs))
                slot_bars = [
                    tqdm(
                        total=1,
                        desc="",
                        unit="",
                        disable=False,
                        leave=False,
                        position=slot_index + 1,
                        bar_format="{desc}",
                        mininterval=_TRACE_PROGRESS_MIN_INTERVAL,
                    )
                    for slot_index in range(slot_count)
                ]
                path_to_slot: dict[str, int] = {}
                free_slots = list(range(slot_count))
                slot_state: dict[int, tuple[str, int, int] | None] = {slot: None for slot in range(slot_count)}
                slot_dirty: dict[int, bool] = {slot: False for slot in range(slot_count)}
                slot_last_refresh: dict[int, float] = {slot: 0.0 for slot in range(slot_count)}

                def refresh_slot(slot: int, *, force: bool = False) -> None:
                    if not slot_dirty[slot]:
                        return
                    now = perf_counter()
                    if not force and now - slot_last_refresh[slot] < _TRACE_PROGRESS_MIN_INTERVAL:
                        return
                    slot_bars[slot].refresh()
                    slot_dirty[slot] = False
                    slot_last_refresh[slot] = now

                def set_slot_text(slot: int, text: str) -> None:
                    slot_bars[slot].set_description_str(text, refresh=False)
                    slot_dirty[slot] = True
                    refresh_slot(slot)

                def release_slot(input_path_str: str) -> None:
                    slot = path_to_slot.pop(input_path_str, None)
                    if slot is None:
                        return
                    set_slot_text(slot, " ")
                    refresh_slot(slot, force=True)
                    slot_state[slot] = None
                    free_slots.append(slot)

                while not stop_progress.is_set():
                    try:
                        message = progress_queue.get(timeout=0.1)
                    except Empty:
                        for slot in range(slot_count):
                            refresh_slot(slot)
                        continue
                    if message is None:
                        break
                    input_path, stage, current, total = message
                    input_path_str = str(input_path)
                    input_name = Path(input_path_str).name
                    if str(stage) == "__done__":
                        release_slot(input_path_str)
                        continue
                    label = _TRACE_PROGRESS_STAGE_LABELS.get(str(stage), str(stage))
                    slot = path_to_slot.get(input_path_str)
                    if slot is None:
                        if not free_slots:
                            continue
                        slot = min(free_slots)
                        free_slots.remove(slot)
                        path_to_slot[input_path_str] = slot

                    if isinstance(total, int) and total > 0:
                        current_value = 0 if current is None else int(current)
                        previous = slot_state.get(slot)
                        should_refresh = (
                            previous is None
                            or previous[0] != str(stage)
                            or previous[2] != int(total)
                            or current_value >= int(total)
                            or current_value == 0
                            or current_value - previous[1] >= _TRACE_PROGRESS_REFRESH_EVERY
                        )
                        if not should_refresh:
                            continue
                        text = f"{input_name} | {label} {current_value}/{total}"
                        slot_state[slot] = (str(stage), current_value, int(total))
                    else:
                        text = f"{input_name} | {label}"
                        slot_state[slot] = None

                    set_slot_text(slot, text)

                    if isinstance(total, int) and total > 0 and int(current or 0) >= total:
                        release_slot(input_path_str)

                for input_path_str in list(path_to_slot):
                    release_slot(input_path_str)
                for slot, bar in enumerate(slot_bars):
                    refresh_slot(slot, force=True)
                    _safe_tqdm_close(bar)

            progress_thread = threading.Thread(target=drain_progress_queue, daemon=True)
            progress_thread.start()

        executor_kwargs: dict[str, object] = {
            "max_workers": min(resolved_batch_n_jobs, len(jobs)),
        }
        if mp_context is not None:
            executor_kwargs["mp_context"] = mp_context

        try:
            with ProcessPoolExecutor(**executor_kwargs) as executor:
                futures = {
                    executor.submit(_trace_file_worker, job + (progress_queue,)): Path(job[0]) for job in jobs
                }
                for future in as_completed(futures):
                    try:
                        record = future.result()
                    except Exception as exc:  # pragma: no cover - defensive guard
                        input_path = futures[future]
                        record = {
                            "timestamp_utc": _iso_timestamp(),
                            "status": "failed",
                            "input_path": str(input_path),
                            "output_swc": str(output_dir / f"{input_path.name}.swc"),
                            "elapsed_seconds": None,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                            "traceback": traceback.format_exc(),
                            "timeout_seconds": None,
                        }

                    handle_record(record)
                    progress.update(1)
                    if progress_queue is not None:
                        progress_queue.put((str(record["input_path"]), "__done__", 1, 1))
        finally:
            if progress_queue is not None:
                stop_progress.set()
                progress_queue.put(None)
            if progress_thread is not None:
                progress_thread.join(timeout=1.0)
            if manager is not None:
                manager.shutdown()

        return sorted(set(completed_outputs))
    finally:
        _safe_tqdm_close(progress)


def trace_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    suffixes: Sequence[str] = SUPPORTED_IMAGE_SUFFIXES,
    visualization_dir: str | Path | None = None,
    batch_n_jobs: int = 1,
    trace_n_jobs: int = 1,
    trace_timeout: float | None = None,
    verbose: int = 1,
    manifest_path: str | Path | None = None,
    overwrite: bool = False,
    on_exists: str | None = None,
    config: str | None = None,
    seed_strategy: str | None = None,
) -> list[Path]:
    input_dir = Path(input_dir)
    image_paths = sorted(
        path for path in input_dir.iterdir() if path.is_file() and _matches_suffix(path, suffixes)
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported image files were found in {input_dir}.")

    return trace_files(
        image_paths,
        output_dir,
        visualization_dir=visualization_dir,
        batch_n_jobs=batch_n_jobs,
        trace_n_jobs=trace_n_jobs,
        trace_timeout=trace_timeout,
        verbose=verbose,
        manifest_path=manifest_path,
        overwrite=overwrite,
        on_exists=on_exists,
        config=config,
        seed_strategy=seed_strategy,
    )


def extract_trace_seeds(
    image: np.ndarray | str | Path,
    *,
    threshold: float | None = None,
    binary_image: np.ndarray | None = None,
    n_jobs: int = 1,
    timeout: float | None = None,
    verbose: int = 1,
    output_path: str | Path | None = None,
    visualization_path: str | Path | None = None,
    config: str | ModuleType | None = None,
    seed_strategy: str | None = None,
    check_timeout: Callable[[str | None], None] | None = None,
) -> Seeds:
    with _temporary_trace_config(config):
        resolved_seed_strategy = _resolve_seed_strategy(seed_strategy)
        if check_timeout is None:
            _, check_timeout = _make_timeout_checker(timeout)
        volume = _load_image_volume(image)
        signal_image = _prepare_signal_image(volume, verbose=verbose)
        if binary_image is None:
            if threshold is None:
                threshold = _estimate_threshold(signal_image, verbose=verbose)
                binary_image = _build_binary_image(signal_image, threshold, verbose=verbose)
            else:
                binary_image = _build_binary_image(signal_image, threshold, verbose=verbose)
        else:
            binary_image = np.ascontiguousarray(np.asarray(binary_image), dtype=np.uint8)
        seeds = Seeds()
        resolved_n_jobs = _resolve_n_jobs(n_jobs)
        if resolved_seed_strategy == "lazy":
            seeds.generate_seed_candidates(
                binary_image,
                n_jobs=resolved_n_jobs,
                verbose=verbose,
                check_timeout=check_timeout,
            )
            seeds.seed_strategy = "lazy"
            seeds.scored = False
        elif resolved_n_jobs > 1:
            with _shared_array(signal_image) as (shared_signal_image, shared_image_spec):
                seeds.generate_tracing_seeds(
                    shared_signal_image,
                    binary_image,
                    n_jobs=resolved_n_jobs,
                    verbose=verbose,
                    check_timeout=check_timeout,
                    shared_image=shared_image_spec,
                )
        else:
            seeds.generate_tracing_seeds(
                signal_image,
                binary_image,
                n_jobs=resolved_n_jobs,
                verbose=verbose,
                check_timeout=check_timeout,
            )
        if output_path is not None:
            save_trace_stage(seeds, output_path)
        if visualization_path is not None:
            save_seed_overlay_figure(
                volume,
                seeds,
                visualization_path,
                title=Path(visualization_path).stem,
            )
        return seeds


def generate_trace_chains(
    seeds: Seeds,
    image: np.ndarray | str | Path,
    *,
    max_seeds: int | None = None,
    filter_chains: bool = True,
    timeout: float | None = None,
    verbose: int = 1,
    output_path: str | Path | None = None,
    visualization_path: str | Path | None = None,
    config: str | ModuleType | None = None,
    seed_strategy: str | None = "auto",
    check_timeout: Callable[[str | None], None] | None = None,
) -> SegmentChains:
    with _temporary_trace_config(config):
        resolved_seed_strategy = _resolve_chain_seed_strategy(seed_strategy, seeds)
        if resolved_seed_strategy == "eager" and not getattr(seeds, "scored", True):
            raise ValueError("Unscored seed candidates require seed_strategy='lazy' or 'auto'.")
        if check_timeout is None:
            _, check_timeout = _make_timeout_checker(timeout)
        volume = _load_image_volume(image)
        signal_image = _prepare_signal_image(volume, verbose=verbose)
        chains = SegmentChains(image_shape=signal_image.shape)
        if resolved_seed_strategy == "lazy":
            chains.generate_neuron_trace_lazy_seed_scoring(
                seeds,
                signal_image,
                max_seeds=max_seeds,
                verbose=verbose,
                check_timeout=check_timeout,
                progress_callback=None,
            )
        else:
            chains.generate_neuron_trace(
                seeds,
                signal_image,
                max_seeds=max_seeds,
                verbose=verbose,
                check_timeout=check_timeout,
            )
        if filter_chains:
            check_timeout("filter_chains")
            chains.filter_chains(verbose=verbose)
        if output_path is not None:
            save_trace_stage(chains, output_path)
        if visualization_path is not None:
            save_chain_overlay_figure(
                volume,
                chains,
                visualization_path,
                title=Path(visualization_path).stem,
            )
        return chains


def connect_trace_chains(
    chains: SegmentChains,
    image: np.ndarray | str | Path,
    *,
    timeout: float | None = None,
    verbose: int = 1,
    visualization_path: str | Path | None = None,
    config: str | ModuleType | None = None,
    check_timeout: Callable[[str | None], None] | None = None,
) -> Neuron:
    with _temporary_trace_config(config):
        if check_timeout is None:
            _, check_timeout = _make_timeout_checker(timeout)
        volume = _load_image_volume(image)
        signal_image = _prepare_signal_image(volume, verbose=verbose)
        connector = ChainConnector(verbose=verbose)
        neuron = connector.reconstruct(chains, signal_image, check_timeout=check_timeout)
        if neuron is None:
            raise ValueError("No neuron reconstruction is available.")
        if visualization_path is not None:
            save_overlay_figure(
                volume,
                neuron,
                visualization_path,
                title=Path(visualization_path).stem,
            )
        return neuron


__all__ = [
    "Seed",
    "Seeds",
    "TracingSegment",
    "SegmentChain",
    "SegmentChains",
    "TracingResult",
    "PreprocessedVolume",
    "TraceTimeoutError",
    "SUPPORTED_IMAGE_SUFFIXES",
    "trace_file",
    "trace_volume",
    "trace_files",
    "trace_directory",
    "save_overlay_figure",
    "save_seed_overlay_figure",
    "save_chain_overlay_figure",
    "ImageParser",
    "Neuron",
    "preprocess_volume",
    "extract_trace_seeds",
    "generate_trace_chains",
    "connect_trace_chains",
    "save_trace_stage",
    "load_trace_stage",
]
