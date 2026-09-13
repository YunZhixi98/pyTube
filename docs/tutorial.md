# Tutorial

## 1. Install

Install PyTube first. The recommended published-package installs are:

```bash
python -m pip install "git+https://github.com/YunZhixi98/pyTube.git"
```

For a local repository checkout, the fuller source-install options are documented in the main `README.md`.

Published wheels do not require Cython at install time.

Normal source installs use `pyproject.toml` build isolation and compile from the `.pyx` sources. Under ordinary `pip install .` usage, `pip` installs the build requirements automatically. Only `--no-build-isolation` or direct developer build commands need `Cython` to be preinstalled.

Create a local environment from the bundled file:

```bash
conda env create -f environment.yml
conda activate pyneutube
```

If you modify `.pyx` files, regenerate the tracked C sources before release:

```bash
python tools/dev/regenerate_cython_sources.py
```

## 2. Load an image volume

```python
from pyneutube import ImageParser

image = ImageParser("examples/data/reference_volume.nii.gz").load()
print(image.shape)
```

Supported input formats:

- TIFF, Vaa3D raw, Vaa3D PBD (`.v3dpbd`), NIfTI, and NRRD

Saving uses the output suffix to choose a format:

```python
from pyneutube import ImageParser

ImageParser.save(image, "volume.v3draw")
ImageParser.save(image, "volume.nii.gz")
```

## 3. Run a trace

```python
from pyneutube import trace_file

result = trace_file(
    "examples/data/reference_volume.nii.gz",
    output_swc="reference_trace.swc",
    visualization_dir="visualizations",
    config=None,
    n_jobs=1,
    timeout=600,
    verbose=1,
    overwrite=False,
)

print(result.threshold, len(result.seeds), len(result.chains))
```

Key runtime controls:

- `n_jobs`: per-volume parallelism for single-image tracing
- `batch_n_jobs`: file-level batch parallelism
- `trace_n_jobs`: per-file parallelism inside each batch worker
- `timeout`: optional per-image timeout in seconds; `None` disables it
- `overwrite`: replace existing outputs
- `verbose`: controls progress logging
- `visualization_dir`: optional output directory for `result/`, `seeds/`, and `chains/` PNG overlays
- `config`: optional Python module path for trace config overrides
- `seed_strategy`: `"eager"` by default; `"lazy"` defers seed scoring until tracing reaches each candidate

The public tracing entry points intentionally keep this parameter set small. They are designed for stable file- and runtime-level control, not for exposing every internal tracing heuristic. Lower-level tracing constants and reconstruction rules still live in internal modules and may change between revisions, so method-tuning experiments should pin a specific commit and document any internal overrides explicitly.

If you need a staged workflow without dropping to internal modules, the public wrappers are:

- `preprocess_volume`
- `extract_trace_seeds`
- `generate_trace_chains`
- `connect_trace_chains`
- `save_trace_stage`
- `load_trace_stage`

## 4. Preprocess without full tracing

```python
from pyneutube import ImageParser, preprocess_volume

image = ImageParser("examples/data/reference_volume.nii.gz").load()
signal, binary_mask, threshold = preprocess_volume(
    image,
    verbose=1,
    output_path="artifacts/preprocess.pkl",
)
```

When `output_path` is set, the saved preprocess artifact only stores the threshold. It does not duplicate the signal image or binary mask.

## 5. Stage-by-stage tracing

```python
from pyneutube import (
    connect_trace_chains,
    extract_trace_seeds,
    generate_trace_chains,
    load_trace_stage,
)

preprocessed = load_trace_stage("artifacts/preprocess.pkl")
seeds = extract_trace_seeds(
    image,
    threshold=preprocessed.threshold,
    output_path="artifacts/seeds.pkl",
    visualization_path="visualizations/seeds/custom_run.png",
)
chains = generate_trace_chains(
    seeds,
    image,
    output_path="artifacts/chains.pkl",
    visualization_path="visualizations/chains/custom_run.png",
)
neuron = connect_trace_chains(
    chains,
    image,
    visualization_path="visualizations/result/custom_run.png",
)
```

This staged mode lets you reuse intermediate artifacts, compare config overrides, and inspect seeds and chains before the final morphology reconstruction.

`extract_trace_seeds(...)` accepts the original image plus either:

- `threshold`, in which case the binary mask is rebuilt internally
- `binary_image`, in which case that binary mask is used directly and `threshold` is ignored

If neither is provided, it runs the full preprocess path internally.

For staged lazy tracing, the seed stage returns unscored candidates:

```python
seeds = extract_trace_seeds(image, seed_strategy="lazy", n_jobs=4)
chains = generate_trace_chains(seeds, image)  # auto-routes lazy candidates
```

## 6. Batch processing

```python
from pyneutube import trace_directory

outputs = trace_directory(
    "input_volumes",
    "swc_outputs",
    visualization_dir="visualizations",
    batch_n_jobs=4,
    trace_n_jobs=1,
    trace_timeout=600,
    verbose=1,
    manifest_path="trace_manifest.jsonl",
    overwrite=False,
)
```

Batch tracing separates outer file-level parallelism from per-file tracing parallelism. With `overwrite=False`, existing SWC outputs are skipped. `trace_timeout` applies to each file independently. The optional manifest file records completed, failed, timed-out, and skipped entries in JSONL format.

## 7. Manual visualization

```python
from pyneutube import save_chain_overlay_figure, save_overlay_figure, save_seed_overlay_figure

save_overlay_figure(
    "examples/data/reference_volume.nii.gz",
    "examples/data/reference_neutube.swc",
    "reference_overlay.png",
)

save_seed_overlay_figure(
    "examples/data/reference_volume.nii.gz",
    seeds,
    "reference_seeds.png",
)

save_chain_overlay_figure(
    "examples/data/reference_volume.nii.gz",
    chains,
    "reference_chains.png",
)
```

`image` can be either a path or a loaded volume. `trace` can be an SWC path, a `Neuron`, or an `(N, 3)` coordinate array. The rendered background uses a `log1p`-transformed maximum-intensity projection and the axes are fixed to the image shape. Seed overlays color seeds by z depth and show each fitted segment projected into the XY plane. Chain overlays color each chain independently and include node markers.

## 8. Trace config

Tracing APIs accept an optional `config` module path. By default they use `pyneutube.tracers.pyNeuTube.config`. To override it:

```python
result = trace_file(
    "examples/data/reference_volume.nii.gz",
    output_swc="reference_trace.swc",
    config="my_project.pyneutube_config",
)
```

The custom module should expose `Defaults` and optionally `Optimization` with the same attribute names as the built-in config.

## 9. Local examples and tools

- `docs/tutorial.ipynb`
- `python -m examples.trace_reference_volume`
- `python -m examples.overlay_reference_reconstruction`
- `python tools/dev/smoke_imports.py`
- `python tools/dev/convert_reference_formats.py`
- `python tools/dev/profile_reference_pipeline.py --max-seeds 2`

The `examples.*` scripts are repository-local developer helpers. If you run them directly from the source tree, run `python Cython_setup.py build_ext --inplace` first. If you want to validate the installed package instead, run the same workflows from outside the repository source tree.
