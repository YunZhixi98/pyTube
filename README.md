# PyTube

PyTube is a Python reimplementation of the NeuTube tracing workflow for 3D microscopy volumes. This repository is trimmed to a minimal reusable release surface focused on image I/O, preprocessing, tracing, and SWC export.

## Current scope

PyTube currently provides:

- multi-format 3D image I/O through `ImageParser`
- SWC parsing and export through `Neuron`
- preprocessing, seed generation, tracing, and chain connection
- optional lightweight overlay visualization for tracing outputs
- public APIs for tracing a single file, multiple files, a directory, or an in-memory volume
- a single command-line entry point for SWC generation

Heavy debug inspection modules are intentionally not part of the release surface.

## Installation

Recommended Python versions: `3.10` to `3.12`.

If you need the latest source version instead of a published package, install from GitHub:

```bash
python -m pip install "git+https://github.com/YunZhixi98/pyTube.git"
```

For a local source tree:

```bash
python -m pip install .
```

Published wheels do not require Cython at install time.

Local source installs use the `pyproject.toml` build environment and compile the Cython extensions from `.pyx`. Under normal `pip install .` usage, `pip` installs those build dependencies automatically in an isolated build environment. You only need to preinstall `Cython` yourself if you intentionally disable build isolation or run developer build commands directly.

For a local `conda` environment from this repository:

```bash
conda env create -f environment.yml
conda activate pyneutube
```

All of these install paths include the supported runtime I/O formats.

## Supported formats

Default input support:

- TIFF stacks: `.tif`, `.tiff`
- Vaa3D raw volumes: `.v3draw`, `.raw`
- Vaa3D PBD-compressed volumes: `.v3dpbd`
- NIfTI: `.nii`, `.nii.gz`
- NRRD: `.nrrd`, `.nhdr`

Current output support:

- SWC export through `Neuron.save_swc`
- image export through `ImageParser.save`
- lightweight overlay export through `save_overlay_figure` or `visualization_dir`

## Quick start

Single volume:

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

Explicit file-list batch:

```python
from pyneutube import trace_files

outputs = trace_files(
    ["sample_a.tif", "sample_b.tif", "sample_c.nii.gz"],
    "swc_outputs",
    batch_n_jobs=4,
    trace_n_jobs=1,
    trace_timeout=600,
    overwrite=False,
)
```

Directory batch:

```python
from pyneutube import trace_directory

outputs = trace_directory(
    "input_volumes",
    "swc_outputs",
    batch_n_jobs=4,
    trace_n_jobs=1,
    trace_timeout=600,
    verbose=1,
    manifest_path="trace_manifest.jsonl",
    overwrite=False,
)
```

Tracing entry points also accept `seed_strategy="lazy"` to defer seed scoring until tracing reaches each candidate; `n_jobs` or `trace_n_jobs` still applies to EDT-based candidate generation.

Command line:

```bash
pyneutube-trace examples/data/reference_volume.nii.gz --verbose 1
pyneutube-trace examples/data/reference_volume.nii.gz --timeout 600 --verbose 1
pyneutube-trace input_volumes --output-dir swc_outputs --visualization-dir visualizations --batch-n-jobs 4 --trace-n-jobs 1 --timeout 600
pyneutube-trace examples/data/reference_volume.nii.gz --config my_project.pyneutube_config
```

## Lightweight visualization

Visualization is optional and disabled by default during tracing. When `visualization_dir` is set, tracing writes PNG maximum-intensity projection overlays for each processed image under:

- `visualization_dir/result/`: final reconstruction overlay
- `visualization_dir/seeds/`: filtered tracing seeds after scoring
- `visualization_dir/chains/`: traced chains before morphology reconstruction

The background image uses a `log1p`-transformed MIP and is clipped to the full image extent.

Manual use is also supported:

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

`image` may be either an image path or an in-memory volume. `trace` may be an SWC path, a `Neuron`, or an `(N, 3)` coordinate array.

## Public API

The supported top-level API is:

- `trace_file`
- `trace_volume`
- `trace_files`
- `trace_directory`
- `save_overlay_figure`
- `save_seed_overlay_figure`
- `save_chain_overlay_figure`
- `ImageParser`
- `Neuron`

The staged public API is:

- `preprocess_volume`
- `extract_trace_seeds`
- `generate_trace_chains`
- `connect_trace_chains`
- `save_trace_stage`
- `load_trace_stage`

Additional public utilities:

- `SUPPORTED_IMAGE_SUFFIXES`
- `subtract_background`
- `threshold_filter`
- `triangle_threshold`
- `refine_local_max_threshold`
- `local_max_filter`
- `connectivity_filter`

Modules under `pyneutube.core.*` and `pyneutube.tracers.*` should be treated as internal implementation details unless explicitly documented otherwise.

The high-level tracing API is intentionally narrow: it exposes stable runtime controls such as I/O, parallelism, timeout, overwrite policy, and verbosity, but keeps most tracing heuristics internal. This keeps the release surface smaller and easier to maintain, at the cost of not exposing every low-level tracing knob through `trace_volume()` and `trace_file()`. For controlled experiments or method development, use the staged public API or inspect the internal tracer modules directly and pin the exact revision you evaluate.

## Staged workflow

You can reproduce `trace_file()` step by step and store intermediate artifacts for reuse:

```python
from pyneutube import (
    ImageParser,
    connect_trace_chains,
    extract_trace_seeds,
    generate_trace_chains,
    load_trace_stage,
    preprocess_volume,
)

image = ImageParser("examples/data/reference_volume.nii.gz").load()
signal_image, binary_image, threshold = preprocess_volume(
    image,
    output_path="artifacts/preprocess.pkl",
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

`preprocess_volume(..., output_path=...)` stores only lightweight threshold metadata, not the signal or binary volumes. This keeps preprocess artifacts small. Downstream staged APIs therefore take the original image again.

For `extract_trace_seeds(...)`:

- if `binary_image` is provided, it is used directly
- otherwise, if `threshold` is provided, binary thresholding is rebuilt internally from the original image
- otherwise, the full preprocess path is run internally

This staged workflow is useful when you want to adjust tracing config between steps or inspect seeds and chains before the final morphology reconstruction.

For staged lazy tracing, pass `seed_strategy="lazy"` to `extract_trace_seeds(...)`. The returned seeds are unscored candidates; `generate_trace_chains(...)` routes them through lazy seed scoring by default:

```python
seeds = extract_trace_seeds(image, seed_strategy="lazy", n_jobs=4)
chains = generate_trace_chains(seeds, image)
```

## Trace config

Tracing APIs accept an optional `config` argument. By default, PyTube uses the built-in tracer config in `pyneutube.tracers.pyNeuTube.config`. To override it, pass a Python module path that exposes `Defaults` and optionally `Optimization` with the same attribute names as the built-in config:

```python
result = trace_file(
    "examples/data/reference_volume.nii.gz",
    output_swc="reference_trace.swc",
    config="my_project.pyneutube_config",
)
```

The same `config` argument is supported by `trace_volume`, `trace_files`, `trace_directory`, `extract_trace_seeds`, `generate_trace_chains`, and `connect_trace_chains`.

## Examples and developer tools

Bundled examples and local release helpers:

- `docs/tutorial.ipynb`
- `python -m examples.trace_reference_volume`
- `python -m examples.overlay_reference_reconstruction`
- `python tools/dev/smoke_imports.py`
- `python tools/dev/regenerate_cython_sources.py`
- `python tools/dev/profile_reference_pipeline.py --max-seeds 2`
- `python tools/dev/convert_reference_formats.py`

The `examples.*` commands above are repository-local development scripts. If you run them directly from the source tree, build the Cython extensions first with `python Cython_setup.py build_ext --inplace`. If you want to exercise the installed package instead, run equivalent API snippets from outside the repository source tree.

## Development checks

Recommended local checks:

```bash
python Cython_setup.py build_ext --inplace
python tools/dev/smoke_imports.py
```

`python Cython_setup.py build_ext --inplace` is only needed for local extension-development workflows. End users should install the published wheel or run `python -m pip install .`, not manage `Cython` manually. 

## License

This project is distributed under the BSD 3-Clause License. See `LICENSE` for details.

## Acknowledgement / Contributors

This project benefited greatly from the contributions of:

- [**Yufeng Liu**](https://github.com/crazylyf) - For his work on the development and optimization of code.
- [**Kangxu Fan**](https://github.com/fkxyyds) - For his assistance in testing and cross-version alignment.
