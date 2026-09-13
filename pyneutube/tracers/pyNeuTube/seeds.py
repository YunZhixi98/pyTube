# tracers/pyNeuTube/seeds.py

"""
seeds.py

Definition of tracing seeds and functions about generating and processing tracing seeds.
"""

import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory

import edt  # for faster distance transformation
import numpy as np
from fast_histogram import histogram1d
from tqdm import tqdm

from pyneutube.core.processing.filtering import maximum_filter_mask
from pyneutube.core.processing.segmentation import label_connected_components

from .config import Defaults, TraceDirection
from .tracing import TracingSegment
from .tracing_utils import label_tracing_mask


def _vprint(verbose: int, message: str) -> None:
    if verbose:
        print(message)


def _resolve_n_jobs(n_jobs: int) -> int:
    if n_jobs == -1:
        return max(1, os.cpu_count() or 1)
    if n_jobs <= 0:
        raise ValueError("`n_jobs` must be a positive integer or -1.")
    return n_jobs


def _seed_priority_order(coords: np.ndarray, values: np.ndarray) -> np.ndarray:
    xyz_coords = coords[:, ::-1]
    # Widen first so the ordering cannot depend on the distance map's precision.
    priority = np.abs(values.astype(np.float64, copy=False) - Defaults.MAX_CONF_RADIUS)
    return np.lexsort((xyz_coords[:, 2], xyz_coords[:, 1], xyz_coords[:, 0], -priority))


def _filter_interior_seed_coords(coords: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if coords.size == 0:
        return coords

    upper_bound = np.asarray(shape, dtype=coords.dtype) - 1
    valid = np.all(coords > 0, axis=1) & np.all(coords < upper_bound, axis=1)
    return coords[valid]


_SEED_SCORE_IMAGE = None
_SEED_SCORE_SHM = None
_PARALLEL_SCORE_WARNING_EMITTED = False
SharedImageSpec = tuple[str, tuple[int, ...], str]


def _seed_score_mp_context():
    if os.name == "posix":
        return get_context("fork")
    return get_context("spawn")


def _init_seed_score_worker(
    shm_name: str,
    shape: tuple[int, ...],
    dtype_str: str,
) -> None:
    global _SEED_SCORE_IMAGE, _SEED_SCORE_SHM

    _SEED_SCORE_SHM = SharedMemory(name=shm_name)
    _SEED_SCORE_IMAGE = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_SEED_SCORE_SHM.buf)


def _score_seed_payload_batch(
    batch: list[tuple[np.ndarray, float]],
) -> list[
    tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        object,
        float | None,
    ]
]:
    image = _SEED_SCORE_IMAGE
    if image is None:
        raise RuntimeError("Seed scoring worker image is not initialized.")

    scored_batch = []
    for coord, value in batch:
        seed = Seed(coord=np.asarray(coord, dtype=np.uint16), value=float(value))
        seed.score_seed(image)
        seg = seed.seg
        scored_batch.append(
            (
                float(seed.score),
                float(seg.radius),
                float(seg.length),
                float(seg.theta),
                float(seg.psi),
                float(seg.scale),
                seg.start_coord.copy(),
                seg.center_coord.copy(),
                seg.end_coord.copy(),
                seg.dir_v.copy(),
                seg.trace_direction,
                None if seg.mean_intensity is None else float(seg.mean_intensity),
            )
        )

    return scored_batch


def _apply_scored_seed_state(
    seed: "Seed",
    state: tuple[
        float,
        float,
        float,
        float,
        float,
        float,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        object,
        float | None,
    ],
) -> None:
    (
        score,
        radius,
        length,
        theta,
        psi,
        scale,
        start_coord,
        center_coord,
        end_coord,
        dir_v,
        trace_direction,
        mean_intensity,
    ) = state

    seg = seed.seg
    seed.score = score
    seg.score = score
    seg.radius = radius
    seg.length = length
    seg.theta = theta
    seg.psi = psi
    seg.scale = scale
    seg.start_coord = np.asarray(start_coord, dtype=np.float64)
    seg.center_coord = np.asarray(center_coord, dtype=np.float64)
    seg.end_coord = np.asarray(end_coord, dtype=np.float64)
    seg.dir_v = np.asarray(dir_v, dtype=np.float64)
    seg.trace_direction = trace_direction
    seg.mean_intensity = mean_intensity
    seg.ball_radius = None


def _batched_seed_payloads(
    seeds: list["Seed"],
    batch_size: int,
) -> list[list[tuple[np.ndarray, float]]]:
    payloads = [(seed.coord.copy(), seed.value) for seed in seeds]
    return [payloads[i : i + batch_size] for i in range(0, len(payloads), batch_size)]


class Seed:
    """
    Class representing a single tracing seed in 3D space.
    A seed is a point in the image that serves as a starting point for tracing algorithms.
    """

    def __init__(self, coord: np.ndarray, value: float, score: float = 0.0):
        self.coord = np.asarray(coord, dtype=np.uint16)
        self.value = float(value)  # intensity or distance transform value at the seed location
        self.score = float(score)
        self.seg = TracingSegment(
            radius=self.value if self.value >= 3.0 else self.value + Defaults.MIN_SEG_RADIUS,
            coord=self.coord,
            alignment="center",
            direction=TraceDirection.BOTH,
        )

    def __repr__(self):
        return f"Seed(coord={self.coord.tolist()}, value={self.value:.3f}, score={self.score:.3f})"

    def score_seed(self, image: np.ndarray) -> None:
        self.seg.centroid_shift(image)
        self.seg.orientation_grid_search(image)
        for _ in range(3):
            self.seg.centroid_shift(image)
        self.seg.fit_segment(image)

        self.score = self.seg.score


class Seeds:
    """
    Class for managing tracing seeds in a 3D image.
    Seeds are points in the image that serve as starting points for tracing algorithms.
    Generally, initial seeds are the position of local maxima of the distance transformed image.
    """

    def __init__(
        self,
        seeds: list[Seed] | Seed | None = None,
        *,
        seed_strategy: str = "eager",
        scored: bool = True,
    ):
        seeds = [] if seeds is None else ([seeds] if isinstance(seeds, Seed) else seeds)
        self._seeds: list[Seed] = list(seeds)
        self.seed_strategy = seed_strategy
        self.scored = scored

    def __len__(self):
        return len(self._seeds)

    def __getitem__(self, idx: int | slice) -> Seed | list[Seed]:
        return self._seeds[idx]

    def __iter__(self):
        return iter(self._seeds)

    def append(self, seed: Seed):
        if not isinstance(seed, Seed):
            raise TypeError("Can only add Seed instances")
        self._seeds.append(seed)

    @property
    def coords(self) -> np.ndarray:
        return (
            np.stack([s.coord for s in self._seeds], axis=0)
            if self._seeds
            else np.empty((0, 3), dtype=np.float64)
        )

    @property
    def values(self) -> np.ndarray:
        return np.array([s.value for s in self._seeds], dtype=np.float64)

    @property
    def scores(self) -> np.ndarray:
        return np.array([s.score for s in self._seeds], dtype=np.float64)

    def _initialize_seeds(
        self, binary_image: np.ndarray, *, n_jobs: int = 1, verbose: int = 1
    ) -> None:
        """
        Initialize seeds based on the local maxima of the distance transformed binary image.
        """
        self._seeds = []
        # `edt` computes in single precision; only the extracted maxima are promoted.
        dt_image = edt.edt(
            binary_image,
            anisotropy=(1, 1, 1),
            black_border=True,
            parallel=_resolve_n_jobs(n_jobs),
        )

        dt_local_max_mask = maximum_filter_mask(dt_image, verbose=max(verbose - 1, 0))
        coords = _filter_interior_seed_coords(np.argwhere(dt_local_max_mask), dt_image.shape)
        coords_values = dt_image[tuple(coords.T)].astype(np.float64)

        arg_idx = _seed_priority_order(coords, coords_values)
        # arg_idx = np.arange(len(coords))
        for coord, value in zip(coords[arg_idx], coords_values[arg_idx], strict=True):
            self.append(Seed(coord=coord[::-1], value=value))  # xyz-order
        _vprint(verbose, f"{len(self)} seeds found")

        return

    def _reduce_seeds(self, binary_image: np.ndarray, *, verbose: int = 1) -> None:
        """
        Filter seeds based on the size of connected components in the binary image.
        """
        min_seed_size = 0
        if self.coords is not None:
            num_seeds = len(self.coords)
            if num_seeds > 15000:
                min_seed_size = 125
            elif num_seeds > 5000:
                min_seed_size = 64

        if min_seed_size > 0:
            _vprint(verbose, "filtering seeds...")
            image_conn_labeled, _ = label_connected_components(binary_image, n_neighbors=26)
            # counts = np.bincount(image_conn_labeled.ravel())  # original version
            flat = image_conn_labeled.ravel()
            imin, imax = int(flat.min()), int(flat.max())
            counts = histogram1d(flat, bins=imax - imin + 1, range=(imin, imax + 1)).astype("int64")
            large_component_mask = counts >= min_seed_size
            large_component_mask[0] = False

            # Dropping whole components leaves the EDT of the survivors intact
            # (the ball of radius EDT(p) is all-foreground and 26-connected to
            # `p`), so re-running EDT + maximum filter on the reduced mask would
            # return these very seeds; select them directly instead.
            zyx = np.asarray(self.coords[:, ::-1], dtype=np.intp)
            seed_labels = image_conn_labeled[zyx[:, 0], zyx[:, 1], zyx[:, 2]]
            keep = large_component_mask[seed_labels]
            self._seeds = [seed for seed, keep_seed in zip(self._seeds, keep) if keep_seed]
            _vprint(verbose, f"{len(self)} seeds found")

        return

    def _sort_seeds(self):
        """
        Sort seeds by their score in descending order.
        """
        self._seeds.sort(
            key=lambda seed: (
                -seed.score,
                -abs(seed.value - Defaults.MAX_CONF_RADIUS),
                int(seed.coord[0]),
                int(seed.coord[1]),
                int(seed.coord[2]),
            )
        )

        return

    def _score_seeds_serial(
        self,
        image: np.ndarray,
        *,
        seeds_mask: np.ndarray,
        verbose: int = 1,
        check_timeout=None,
        progress_callback=None,
    ) -> int:
        nlabeled = 0
        total = len(self._seeds)
        if progress_callback is not None:
            progress_callback("generate_tracing_seeds", 0, total)
        for seed_idx, seed in enumerate(
            tqdm(
            self,
            total=len(self._seeds),
            desc="Scoring seeds",
            unit="seed",
            disable=verbose < 1,
            )
        ):
            if check_timeout is not None and seed_idx % 16 == 0:
                check_timeout("seed scoring")
            if not seeds_mask[tuple(seed.coord[::-1])]:
                seed.score_seed(image)
                label_tracing_mask(seed.seg, seeds_mask, dilate=False)
            else:
                nlabeled += 1
            if progress_callback is not None:
                progress_callback("generate_tracing_seeds", seed_idx + 1, total)
        return nlabeled

    def score_seeds(
        self,
        image: np.ndarray,
        *,
        n_jobs: int = 1,
        verbose: int = 1,
        check_timeout=None,
        shared_image: SharedImageSpec | None = None,
        progress_callback=None,
    ) -> None:
        """
        Score all seeds in the image.
        """
        seeds_mask = np.zeros_like(image, dtype=np.uint8)
        nlabeled = 0
        resolved_n_jobs = _resolve_n_jobs(n_jobs)
        worker_jobs = min(resolved_n_jobs, len(self._seeds))

        if worker_jobs <= 1:
            nlabeled = self._score_seeds_serial(
                image,
                seeds_mask=seeds_mask,
                verbose=verbose,
                check_timeout=check_timeout,
                progress_callback=progress_callback,
            )
        else:
            shm = None
            owns_shared_image = shared_image is None
            try:
                batch_size = max(16, len(self._seeds) // (worker_jobs * 8) or 1)
                payload_batches = _batched_seed_payloads(self._seeds, batch_size)
                total = len(self._seeds)
                if progress_callback is not None:
                    progress_callback("generate_tracing_seeds", 0, total)
                if owns_shared_image:
                    image_shared = np.ascontiguousarray(image)
                    shm = SharedMemory(create=True, size=image_shared.nbytes)
                    shared_array = np.ndarray(image_shared.shape, dtype=image_shared.dtype, buffer=shm.buf)
                    shared_array[...] = image_shared
                    shm_name = shm.name
                    shm_shape = image_shared.shape
                    shm_dtype_str = image_shared.dtype.str
                else:
                    shm_name, shm_shape, shm_dtype_str = shared_image

                try:
                    with ProcessPoolExecutor(
                        max_workers=worker_jobs,
                        mp_context=_seed_score_mp_context(),
                        initializer=_init_seed_score_worker,
                        initargs=(shm_name, shm_shape, shm_dtype_str),
                    ) as executor:
                        seed_offset = 0
                        scored_batches = executor.map(
                            _score_seed_payload_batch,
                            payload_batches,
                            chunksize=1,
                        )
                        with tqdm(
                            total=len(self._seeds),
                            desc="Scoring seeds",
                            unit="seed",
                            disable=verbose < 1,
                        ) as progress:
                            for scored_batch in scored_batches:
                                if check_timeout is not None:
                                    check_timeout("seed scoring")
                                seeds_batch = self._seeds[seed_offset : seed_offset + len(scored_batch)]
                                for seed, scored_state in zip(
                                    seeds_batch,
                                    scored_batch,
                                    strict=True,
                                ):
                                    if not seeds_mask[tuple(seed.coord[::-1])]:
                                        _apply_scored_seed_state(seed, scored_state)
                                        label_tracing_mask(seed.seg, seeds_mask, dilate=False)
                                    else:
                                        nlabeled += 1
                                seed_offset += len(scored_batch)
                                progress.update(len(scored_batch))
                                if progress_callback is not None:
                                    progress_callback("generate_tracing_seeds", seed_offset, total)
                except (OSError, PermissionError) as exc:
                    global _PARALLEL_SCORE_WARNING_EMITTED

                    seeds_mask.fill(0)
                    nlabeled = 0
                    warning_message = (
                        "Parallel seed scoring is unavailable in this environment; "
                        f"requested n_jobs={resolved_n_jobs}, falling back to serial execution "
                        f"(n_jobs=1). Root cause: {type(exc).__name__}: {exc}"
                    )
                    if not _PARALLEL_SCORE_WARNING_EMITTED:
                        warnings.warn(warning_message, RuntimeWarning, stacklevel=2)
                        _PARALLEL_SCORE_WARNING_EMITTED = True
                    else:
                        _vprint(verbose, warning_message)
                    nlabeled = self._score_seeds_serial(
                        image,
                        seeds_mask=seeds_mask,
                        verbose=verbose,
                        check_timeout=check_timeout,
                        progress_callback=progress_callback,
                    )
            finally:
                if owns_shared_image and shm is not None:
                    shm.close()
                    shm.unlink()

        _vprint(verbose, f"Number of labeled seeds: {nlabeled}")
        self.scored = True

        return

    def _filter_seeds(
        self, min_score=Defaults.MIN_SEED_SCORE, max_radius=Defaults.MAX_SEED_RADIUS
    ) -> None:
        """
        Filter seeds based on threshold of score and TracingSegment radius.
        """
        self._seeds = [
            seed
            for seed in self._seeds
            if seed.score > seed.seg.get_norm_min_score(min_score) and seed.seg.radius <= max_radius
        ]

        return

    def generate_tracing_seeds(
        self,
        signal_image: np.ndarray,
        binary_image: np.ndarray,
        *,
        n_jobs: int = 1,
        verbose: int = 1,
        check_timeout=None,
        shared_image: SharedImageSpec | None = None,
        progress_callback=None,
    ):
        self.seed_strategy = "eager"
        self.scored = False
        t0 = time.time()
        if check_timeout is not None:
            check_timeout("seed initialization")
        self._initialize_seeds(binary_image, n_jobs=n_jobs, verbose=verbose)
        self._reduce_seeds(binary_image, verbose=verbose)
        _vprint(verbose, f"--> seed_init: {time.time() - t0:.6f}s")
        if check_timeout is not None:
            check_timeout("seed scoring")
        self.score_seeds(
            signal_image,
            n_jobs=n_jobs,
            verbose=verbose,
            check_timeout=check_timeout,
            shared_image=shared_image,
            progress_callback=progress_callback,
        )
        _vprint(verbose, f"--> seed_score: {time.time() - t0:.6f}s")
        if check_timeout is not None:
            check_timeout("seed filtering")
        self._filter_seeds()
        _vprint(verbose, f"Number of seed after filtering: {len(self)}")
        self._sort_seeds()
        self.scored = True

        return

    def generate_seed_candidates(
        self,
        binary_image: np.ndarray,
        *,
        n_jobs: int = 1,
        verbose: int = 1,
        check_timeout=None,
    ):
        self.seed_strategy = "lazy"
        self.scored = False
        t0 = time.time()
        if check_timeout is not None:
            check_timeout("seed initialization")
        self._initialize_seeds(binary_image, n_jobs=n_jobs, verbose=verbose)
        self._reduce_seeds(binary_image, verbose=verbose)
        self._sort_seeds()
        _vprint(verbose, f"--> seed_candidates: {time.time() - t0:.6f}s")

        return
