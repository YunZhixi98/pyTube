from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path

import numpy as np
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

try:
    from Cython.Build import cythonize
except ImportError:  # pragma: no cover - optional at build time
    cythonize = None


ROOT = Path(__file__).parent.resolve()
COMPILER_DIRECTIVES = {
    "language_level": "3",
    "boundscheck": False,
    "wraparound": False,
    "cdivision": True,
    "embedsignature": True,
    "profile": False,
}


def _compiled_source(stem: str) -> str:
    for suffix in (".c", ".cpp"):
        candidate = ROOT / f"{stem}{suffix}"
        if candidate.exists():
            return f"{stem}{suffix}".replace("\\", "/")
    raise FileNotFoundError(f"Missing generated C/C++ source for {stem!r}.")


def _extension_source(stem: str, *, use_cython: bool) -> str:
    if use_cython:
        return f"{stem}.pyx".replace("\\", "/")
    return _compiled_source(stem)


def _extra_compile_args() -> list[str]:
    if sys.platform == "win32":
        return ["/O2"]
    return ["-O3", "-std=gnu99"]


def _strip_compiler_compat_flags(command):
    if isinstance(command, str):
        parts = shlex.split(command)
        stripped = _strip_compiler_compat_flags(parts)
        return " ".join(shlex.quote(part) for part in stripped)

    if not isinstance(command, (list, tuple)):
        return command

    result: list[str] = []
    skip_next = False
    for index, part in enumerate(command):
        if skip_next:
            skip_next = False
            continue
        if part == "-B" and index + 1 < len(command) and "compiler_compat" in command[index + 1]:
            skip_next = True
            continue
        if isinstance(part, str) and part.startswith("-B") and "compiler_compat" in part:
            continue
        result.append(part)
    return result


class PyNeuTubeBuildExt(build_ext):
    def build_extensions(self):
        if (
            sys.platform.startswith("linux")
            and os.environ.get("CONDA_PREFIX")
            and os.environ.get("PYNEUTUBE_KEEP_CONDA_COMPILER_COMPAT") != "1"
        ):
            # Some Linux+conda environments inject "-B .../compiler_compat" into the
            # linker command and then fail while linking even trivial extensions.
            # Keep the compile-side command line intact and only strip the linker-side
            # compiler_compat wrapper, which is the path implicated by the observed
            # "collect2: error: ld returned 255 exit status" failures.
            for attr in ("linker_so", "linker_exe"):
                command = getattr(self.compiler, attr, None)
                if command is not None:
                    setattr(self.compiler, attr, _strip_compiler_compat_flags(command))
        super().build_extensions()


def build_extensions(*, use_cython: bool = False) -> list[Extension]:
    extensions = [
        Extension(
            "pyneutube.core.io.vaa3d_accel",
            [_extension_source("pyneutube/core/io/vaa3d_accel", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.core.processing.local_maximum",
            [_extension_source("pyneutube/core/processing/local_maximum", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.core.processing.transform",
            [_extension_source("pyneutube/core/processing/transform", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.core.processing.sampling",
            [_extension_source("pyneutube/core/processing/sampling", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.tracers.pyNeuTube.filters",
            [_extension_source("pyneutube/tracers/pyNeuTube/filters", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.tracers.pyNeuTube.seg_utils",
            [_extension_source("pyneutube/tracers/pyNeuTube/seg_utils", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.tracers.pyNeuTube.stack_graph_utils",
            [_extension_source("pyneutube/tracers/pyNeuTube/stack_graph_utils", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
        Extension(
            "pyneutube.tracers.pyNeuTube.geometry_accel",
            [_extension_source("pyneutube/tracers/pyNeuTube/geometry_accel", use_cython=use_cython)],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=_extra_compile_args(),
        ),
    ]

    shortest_path_stem = "pyneutube/tracers/pyNeuTube/shortest_path_accel"
    shortest_path_pyx = ROOT / f"{shortest_path_stem}.pyx"
    shortest_path_c = ROOT / f"{shortest_path_stem}.c"
    if use_cython or shortest_path_c.exists():
        extensions.append(
            Extension(
                "pyneutube.tracers.pyNeuTube.shortest_path_accel",
                [_extension_source(shortest_path_stem, use_cython=use_cython)],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                extra_compile_args=_extra_compile_args(),
            )
        )

    optimization_accel_stem = "pyneutube/tracers/pyNeuTube/optimization_accel"
    optimization_accel_c = ROOT / f"{optimization_accel_stem}.c"
    if use_cython or optimization_accel_c.exists():
        extensions.append(
            Extension(
                "pyneutube.tracers.pyNeuTube.optimization_accel",
                [_extension_source(optimization_accel_stem, use_cython=use_cython)],
                include_dirs=[np.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
                extra_compile_args=_extra_compile_args(),
            )
        )

    if use_cython:
        if cythonize is None:
            raise RuntimeError(
                "PYNEUTUBE_USE_CYTHON=1 requires Cython to be installed in the build environment."
            )
        return cythonize(extensions, compiler_directives=COMPILER_DIRECTIVES)

    return extensions


def build_setup_kwargs(*, use_cython: bool = False) -> dict[str, object]:
    return {
        "ext_modules": build_extensions(use_cython=use_cython),
        "cmdclass": {"build_ext": PyNeuTubeBuildExt},
    }


if __name__ == "__main__":
    setup(**build_setup_kwargs(use_cython=False))
