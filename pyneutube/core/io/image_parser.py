"""Utilities for reading and writing 3D microscopy and medical-image volumes."""

from __future__ import annotations

import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

try:  # Optional runtime dependency.
    import nibabel as nib
except ImportError:  # pragma: no cover - optional dependency
    nib = None

try:  # Optional runtime dependency.
    import nrrd
except ImportError:  # pragma: no cover - optional dependency
    nrrd = None

try:  # Optional compiled accelerator.
    from . import vaa3d_accel
except ImportError:  # pragma: no cover - extension may be absent in source checkouts
    vaa3d_accel = None


def _vprint(verbose: int, message: str) -> None:
    if verbose:
        print(message)


def _normalized_shape(shape: tuple[int, ...]) -> tuple[int, int, int]:
    dims = tuple(int(dim) for dim in shape)
    dims = tuple(dim for dim in dims if dim != 1) or (1,)
    if len(dims) == 2:
        return (1, dims[0], dims[1])
    if len(dims) != 3:
        raise ValueError(f"Expected a 3D volume, got shape {shape!r}.")
    return dims


def _normalized_volume(image: np.ndarray) -> np.ndarray:
    volume = np.asarray(image)
    while volume.ndim > 3 and 1 in volume.shape:
        volume = np.squeeze(volume)
    if volume.ndim == 2:
        volume = volume[np.newaxis, ...]
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got array with shape {volume.shape!r}.")
    return volume


def _require_optional_dependency(module: Any, dependency_name: str, fmt: str) -> Any:
    if module is None:
        raise ImportError(
            f"Reading {fmt} volumes requires the optional dependency {dependency_name!r}. "
            f"Install it with `pip install {dependency_name}`."
        )
    return module


def _dtype_from_vaa3d(datatype: int, endian: str) -> np.dtype:
    dtype_map = {
        1: np.dtype("u1"),
        2: np.dtype(f"{endian}u2"),
        4: np.dtype(f"{endian}f4"),
        33: np.dtype("u1"),
    }
    if datatype not in dtype_map:
        raise ValueError(f"Unsupported Vaa3D datatype code {datatype!r}.")
    return dtype_map[datatype]


def _detect_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".tif", ".tiff")):
        return "tiff"
    if name.endswith((".v3draw", ".raw")):
        return "v3draw"
    if name.endswith(".v3dpbd"):
        return "v3dpbd"
    if name.endswith((".nii", ".nii.gz")):
        return "nifti"
    if name.endswith((".nrrd", ".nhdr")):
        return "nrrd"
    raise ValueError(f"Unsupported image format for {path.name!r}.")


def _read_v3draw_header(path: Path) -> tuple[tuple[int, int, int], np.dtype, dict[str, Any]]:
    format_key = b"raw_image_stack_by_hpeng"
    with path.open("rb") as handle:
        if path.stat().st_size < len(format_key) + 1 + 2 + 16:
            raise ValueError(f"Vaa3D raw file is truncated: {path}")
        if handle.read(len(format_key)) != format_key:
            raise ValueError(f"Unsupported Vaa3D raw header in {path}.")

        endian_code = handle.read(1).decode("ascii")
        if endian_code == "B":
            endian = ">"
        elif endian_code == "L":
            endian = "<"
        else:
            raise ValueError(f"Unsupported Vaa3D endian code {endian_code!r} in {path}.")

        datatype = struct.unpack(f"{endian}h", handle.read(2))[0]
        dims = struct.unpack(f"{endian}iiii", handle.read(16))

    dtype = _dtype_from_vaa3d(datatype, endian)
    return (
        _normalized_shape(dims[-1:-5:-1]),
        np.dtype(dtype).newbyteorder("="),
        {
            "format": "v3draw",
            "endian_code": endian_code,
            "vaa3d_datatype": datatype,
            "vaa3d_shape_xyzc": tuple(int(dim) for dim in dims),
        },
    )


def load_v3draw(path: str | os.PathLike[str]) -> np.ndarray:
    path = Path(path)
    if vaa3d_accel is not None:
        return _normalized_volume(vaa3d_accel.load_v3draw(path))

    format_key = b"raw_image_stack_by_hpeng"
    with path.open("rb") as handle:
        if handle.read(len(format_key)) != format_key:
            raise ValueError(f"Unsupported Vaa3D raw header in {path}.")

        endian_code = handle.read(1).decode("ascii")
        if endian_code == "B":
            endian = ">"
        elif endian_code == "L":
            endian = "<"
        else:
            raise ValueError(f"Unsupported Vaa3D endian code {endian_code!r} in {path}.")

        datatype = struct.unpack(f"{endian}h", handle.read(2))[0]
        dims = struct.unpack(f"{endian}iiii", handle.read(16))
        dtype = _dtype_from_vaa3d(datatype, endian)
        image = np.frombuffer(handle.read(), dtype=dtype)

    return _normalized_volume(image.reshape(dims[-1:-5:-1])).astype(
        np.dtype(dtype).newbyteorder("="), copy=False
    )


def save_v3draw(image: np.ndarray, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    volume = np.ascontiguousarray(_normalized_volume(np.asarray(image)))
    path.parent.mkdir(parents=True, exist_ok=True)

    if volume.dtype.byteorder == ">":
        endian_code = "B"
    elif volume.dtype.byteorder == "<":
        endian_code = "L"
    elif volume.dtype.byteorder == "|":
        endian_code = "L" if sys.byteorder == "little" else "B"
    else:
        endian_code = "L" if sys.byteorder == "little" else "B"

    if volume.dtype == np.uint8:
        datatype = 1
    elif volume.dtype == np.uint16:
        datatype = 2
    elif volume.dtype == np.float32:
        datatype = 4
    else:
        raise ValueError("Vaa3D raw saving supports uint8, uint16, and float32 volumes only.")

    endian = ">" if endian_code == "B" else "<"
    shape_xyzc = [volume.shape[2], volume.shape[1], volume.shape[0], 1]

    with path.open("wb") as handle:
        handle.write(b"raw_image_stack_by_hpeng")
        handle.write(endian_code.encode("ascii"))
        handle.write(struct.pack(f"{endian}h", datatype))
        handle.write(struct.pack(f"{endian}iiii", *shape_xyzc))
        handle.write(
            np.asarray(volume, dtype=np.dtype(_dtype_from_vaa3d(datatype, endian))).tobytes(
                order="C"
            )
        )


def save_v3dpbd(image: np.ndarray, path: str | os.PathLike[str]) -> None:
    path = Path(path)
    volume = np.ascontiguousarray(_normalized_volume(np.asarray(image)))
    path.parent.mkdir(parents=True, exist_ok=True)
    if vaa3d_accel is None:
        raise RuntimeError("Vaa3D PBD saving requires the compiled image I/O extension.")
    vaa3d_accel.save_v3dpbd(volume, path)


class PBD:
    """Loader for Vaa3D PBD-compressed volumes."""

    def __init__(self) -> None:
        self.total_read_bytes = 0
        self.max_decompression_size = 0
        self.channel_len = 0
        self.compression_buffer = bytearray()
        self.decompression_buffer = bytearray()
        self.compression_pos = 0
        self.decompression_pos = 0
        self.decompression_prior = 0
        self.pbd3_src_min = 0
        self.pbd3_src_max = 0
        self.pbd3_cur_min = 0
        self.pbd3_cur_max = 0
        self.pbd3_cur_chan = 0
        self.load_datatype = 0
        self.pbd_sz = [0, 0, 0, 0]
        self.endian = ""

    def decompress_pbd8(self, compression_len: int) -> int:
        cp = 0
        dp = 0
        pva = 0
        pvb = 0
        mask = 3
        while cp < compression_len:
            value = self.compression_buffer[self.compression_pos + cp]
            if value < 33:
                count = value + 1
                self.decompression_buffer[
                    self.decompression_pos + dp : self.decompression_pos + dp + count
                ] = self.compression_buffer[
                    self.compression_pos + cp + 1 : self.compression_pos + cp + 1 + count
                ]
                cp += count + 1
                dp += count
                self.decompression_prior = self.decompression_buffer[
                    self.decompression_pos + dp - 1
                ]
            elif value < 128:
                left_to_fill = value - 32
                while left_to_fill > 0:
                    fill_num = min(left_to_fill, 4)
                    cp += 1
                    src_char = self.compression_buffer[self.compression_pos + cp]
                    to_fill = self.decompression_pos + dp
                    p0 = src_char & mask
                    src_char >>= 2
                    p1 = src_char & mask
                    src_char >>= 2
                    p2 = src_char & mask
                    src_char >>= 2
                    p3 = src_char & mask
                    pva = self.decompression_prior + (-1 if p0 == 3 else p0)
                    self.decompression_buffer[to_fill] = pva
                    if fill_num > 1:
                        to_fill += 1
                        pvb = pva + (-1 if p1 == 3 else p1)
                        self.decompression_buffer[to_fill] = pvb
                        if fill_num > 2:
                            to_fill += 1
                            pva = pvb + (-1 if p2 == 3 else p2)
                            self.decompression_buffer[to_fill] = pva
                            if fill_num > 3:
                                to_fill += 1
                                self.decompression_buffer[to_fill] = pva + (-1 if p3 == 3 else p3)
                    self.decompression_prior = self.decompression_buffer[to_fill]
                    dp += fill_num
                    left_to_fill -= fill_num
                cp += 1
            else:
                repeat_count = value - 127
                cp += 1
                repeat_value = self.compression_buffer[
                    self.compression_pos + cp : self.compression_pos + cp + 1
                ]
                self.decompression_buffer[
                    self.decompression_pos + dp : self.decompression_pos + dp + repeat_count
                ] = repeat_value * repeat_count
                dp += repeat_count
                self.decompression_prior = struct.unpack("B", repeat_value)[0]
                cp += 1
        return dp

    def decompress_pbd16(self, compression_len: int) -> int:
        cp = 0
        dp = 0

        def get_pre() -> int:
            return struct.unpack(
                f"{self.endian}H",
                bytes(
                    self.decompression_buffer[
                        self.decompression_pos + dp - 2 : self.decompression_pos + dp
                    ]
                ),
            )[0]

        while cp < compression_len:
            code = self.compression_buffer[self.compression_pos + cp]
            if code < 32:
                count = code + 1
                self.decompression_buffer[
                    self.decompression_pos + dp : self.decompression_pos + dp + count * 2
                ] = self.compression_buffer[
                    self.compression_pos + cp + 1 : self.compression_pos + cp + 1 + count * 2
                ]
                cp += count * 2 + 1
                dp += count * 2
                self.decompression_prior = get_pre()
            elif code < 80:
                left_to_fill = code - 31
                while left_to_fill > 0:
                    cp += 1
                    src_char = self.compression_buffer[self.compression_pos + cp]
                    d0 = src_char >> 5
                    self.decompression_buffer[
                        self.decompression_pos + dp : self.decompression_pos + dp + 2
                    ] = struct.pack(
                        f"{self.endian}H", self.decompression_prior + (d0 if d0 < 5 else 4 - d0)
                    )
                    dp += 2
                    left_to_fill -= 1
                    if left_to_fill == 0:
                        break

                    d1 = (src_char >> 2) & 7
                    self.decompression_buffer[
                        self.decompression_pos + dp : self.decompression_pos + dp + 2
                    ] = struct.pack(f"{self.endian}H", get_pre() + (d1 if d1 < 5 else 4 - d1))
                    dp += 2
                    left_to_fill -= 1
                    if left_to_fill == 0:
                        break

                    carry_over = src_char & 3
                    cp += 1
                    src_char = self.compression_buffer[self.compression_pos + cp]
                    d0 = (src_char >> 7) | (carry_over << 1)
                    self.decompression_buffer[
                        self.decompression_pos + dp : self.decompression_pos + dp + 2
                    ] = struct.pack(f"{self.endian}H", get_pre() + (d0 if d0 < 5 else 4 - d0))
                    dp += 2
                    left_to_fill -= 1
                    if left_to_fill == 0:
                        break

                    d1 = (src_char >> 4) & 7
                    self.decompression_buffer[
                        self.decompression_pos + dp : self.decompression_pos + dp + 2
                    ] = struct.pack(f"{self.endian}H", get_pre() + (d1 if d1 < 5 else 4 - d1))
                    dp += 2
                    left_to_fill -= 1
                    if left_to_fill == 0:
                        break

                    d2 = (src_char >> 1) & 7
                    self.decompression_buffer[
                        self.decompression_pos + dp : self.decompression_pos + dp + 2
                    ] = struct.pack(f"{self.endian}H", get_pre() + (d2 if d2 < 5 else 4 - d2))
                    dp += 2
                    left_to_fill -= 1
                    if left_to_fill == 0:
                        break

                    carry_over = src_char & 1
                    cp += 1
                    src_char = self.compression_buffer[self.compression_pos + cp]
                    d0 = (src_char >> 6) | (carry_over << 2)
                    self.decompression_buffer[
                        self.decompression_pos + dp : self.decompression_pos + dp + 2
                    ] = struct.pack(f"{self.endian}H", get_pre() + (d0 if d0 < 5 else 4 - d0))
                    dp += 2
                    left_to_fill -= 1
                    if left_to_fill == 0:
                        break

                    d1 = (src_char >> 3) & 7
                    self.decompression_buffer[
                        self.decompression_pos + dp : self.decompression_pos + dp + 2
                    ] = struct.pack(f"{self.endian}H", get_pre() + (d1 if d1 < 5 else 4 - d1))
                    dp += 2
                    left_to_fill -= 1
                    if left_to_fill == 0:
                        break

                    d2 = src_char & 7
                    self.decompression_buffer[
                        self.decompression_pos + dp : self.decompression_pos + dp + 2
                    ] = struct.pack(f"{self.endian}H", get_pre() + (d2 if d2 < 5 else 4 - d2))
                    dp += 2
                    left_to_fill -= 1
                    self.decompression_prior = get_pre()
                self.decompression_prior = get_pre()
                cp += 1
            elif code < 223:
                raise NotImplementedError("Vaa3D PBD datatype 3 is not implemented.")
            else:
                repeat_count = code - 222
                cp += 1
                repeat_value = self.compression_buffer[
                    self.compression_pos + cp : self.compression_pos + cp + 2
                ]
                self.decompression_buffer[
                    self.decompression_pos + dp : self.decompression_pos + dp + repeat_count * 2
                ] = repeat_value * repeat_count
                dp += repeat_count * 2
                cp += 2
                self.decompression_prior = struct.unpack(f"{self.endian}H", repeat_value)[0]
        return dp

    def update_compression_buffer8(self) -> None:
        look_ahead = self.compression_pos
        while look_ahead < self.total_read_bytes:
            lav = self.compression_buffer[look_ahead]
            if lav < 33:
                if look_ahead + lav + 1 < self.total_read_bytes:
                    look_ahead += lav + 2
                else:
                    break
            elif lav < 128:
                compressed_diff_entries = (lav - 33) // 4 + 1
                if look_ahead + compressed_diff_entries < self.total_read_bytes:
                    look_ahead += compressed_diff_entries + 1
                else:
                    break
            else:
                if look_ahead + 1 < self.total_read_bytes:
                    look_ahead += 2
                else:
                    break
        compression_len = look_ahead - self.compression_pos
        d_length = self.decompress_pbd8(compression_len)
        self.compression_pos = look_ahead
        self.decompression_pos += d_length

    def update_compression_buffer16(self) -> None:
        look_ahead = self.compression_pos
        while look_ahead < self.total_read_bytes:
            lav = self.compression_buffer[look_ahead]
            if lav < 32:
                if look_ahead + (lav + 1) * 2 < self.total_read_bytes:
                    look_ahead += (lav + 1) * 2 + 1
                else:
                    break
            elif lav < 80:
                compressed_diff_bytes = int(((lav - 31) * 3 / 8) - 0.0001) + 1
                if look_ahead + compressed_diff_bytes < self.total_read_bytes:
                    look_ahead += compressed_diff_bytes + 1
                else:
                    break
            elif lav < 183:
                compressed_diff_bytes = int(((lav - 79) * 4 / 8) - 0.0001) + 1
                if look_ahead + compressed_diff_bytes < self.total_read_bytes:
                    look_ahead += compressed_diff_bytes + 1
                else:
                    break
            elif lav < 223:
                compressed_diff_bytes = int(((lav - 182) * 5 / 8) - 0.0001) + 1
                if look_ahead + compressed_diff_bytes < self.total_read_bytes:
                    look_ahead += compressed_diff_bytes + 1
                else:
                    break
            else:
                if look_ahead + 2 < self.total_read_bytes:
                    look_ahead += 3
                else:
                    break
        compression_len = look_ahead - self.compression_pos
        d_length = self.decompress_pbd16(compression_len)
        self.compression_pos = look_ahead
        self.decompression_pos += d_length

    def load_image(self, path: str | os.PathLike[str]) -> np.ndarray:
        path = Path(path)
        if vaa3d_accel is not None:
            return _normalized_volume(vaa3d_accel.load_v3dpbd(path))

        self.decompression_prior = 0
        format_key = b"v3d_volume_pkbitdf_encod"
        with path.open("rb") as handle:
            header_size = len(format_key) + 1 + 2 + 16
            if path.stat().st_size < header_size:
                raise ValueError(f"Vaa3D PBD file is truncated: {path}")
            if handle.read(len(format_key)) != format_key:
                raise ValueError(f"Unsupported Vaa3D PBD header in {path}.")

            endian_code = handle.read(1).decode("ascii")
            if endian_code == "B":
                self.endian = ">"
            elif endian_code == "L":
                self.endian = "<"
            else:
                raise ValueError(f"Unsupported Vaa3D endian code {endian_code!r} in {path}.")

            datatype = struct.unpack(f"{self.endian}h", handle.read(2))[0]
            if datatype not in (1, 2, 33):
                raise ValueError(f"Unsupported Vaa3D PBD datatype code {datatype!r}.")

            dims = struct.unpack(f"{self.endian}iiii", handle.read(16))
            total_units = dims[0] * dims[1] * dims[2] * dims[3]
            remaining_bytes = path.stat().st_size - header_size
            self.max_decompression_size = total_units * (1 if datatype in (1, 33) else datatype)
            self.channel_len = dims[0] * dims[1] * dims[2]
            self.total_read_bytes = 0
            self.compression_pos = 0
            self.decompression_pos = 0
            self.load_datatype = datatype
            self.pbd_sz = list(dims)
            self.compression_buffer = bytearray(remaining_bytes)
            self.decompression_buffer = bytearray(self.max_decompression_size)

            while remaining_bytes > 0:
                current_read_bytes = min(remaining_bytes, 1024 * 20000)
                current_read_bytes = min(
                    current_read_bytes,
                    (self.total_read_bytes // self.channel_len + 1) * self.channel_len
                    - self.total_read_bytes,
                )
                self.compression_buffer[
                    self.total_read_bytes : self.total_read_bytes + current_read_bytes
                ] = handle.read(current_read_bytes)
                self.total_read_bytes += current_read_bytes
                remaining_bytes -= current_read_bytes
                if datatype == 1:
                    self.update_compression_buffer8()
                elif datatype == 2:
                    self.update_compression_buffer16()
                else:
                    raise NotImplementedError("Vaa3D PBD datatype 33 is not implemented.")

        dtype = _dtype_from_vaa3d(datatype, self.endian)
        image = np.frombuffer(self.decompression_buffer, dtype=dtype).reshape(dims[-1:-5:-1])
        return _normalized_volume(image).astype(np.dtype(dtype).newbyteorder("="), copy=False)


def _read_v3dpbd_header(path: Path) -> tuple[tuple[int, int, int], np.dtype, dict[str, Any]]:
    format_key = b"v3d_volume_pkbitdf_encod"
    with path.open("rb") as handle:
        if path.stat().st_size < len(format_key) + 1 + 2 + 16:
            raise ValueError(f"Vaa3D PBD file is truncated: {path}")
        if handle.read(len(format_key)) != format_key:
            raise ValueError(f"Unsupported Vaa3D PBD header in {path}.")

        endian_code = handle.read(1).decode("ascii")
        if endian_code == "B":
            endian = ">"
        elif endian_code == "L":
            endian = "<"
        else:
            raise ValueError(f"Unsupported Vaa3D endian code {endian_code!r} in {path}.")

        datatype = struct.unpack(f"{endian}h", handle.read(2))[0]
        dims = struct.unpack(f"{endian}iiii", handle.read(16))

    dtype = _dtype_from_vaa3d(datatype, endian)
    return (
        _normalized_shape(dims[-1:-5:-1]),
        np.dtype(dtype).newbyteorder("="),
        {
            "format": "v3dpbd",
            "endian_code": endian_code,
            "vaa3d_datatype": datatype,
            "vaa3d_shape_xyzc": tuple(int(dim) for dim in dims),
        },
    )


class ImageParser:
    """Load and inspect microscopy volumes across supported file formats."""

    def __init__(
        self,
        filepath: str | os.PathLike[str],
        *,
        verbose: int = 0,
    ) -> None:
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Image file not found: {self.filepath}")

        self.verbose = verbose
        self.format = _detect_format(self.filepath)
        self._metadata: dict[str, Any] = {}
        self._shape: tuple[int, int, int] | None = None
        self._dtype: np.dtype | None = None
        self._cached_volume: np.ndarray | None = None
        self._read_header()

    def _read_header(self) -> None:
        if self.format == "tiff":
            with tifffile.TiffFile(self.filepath) as tif:
                series = tif.series[0]
                self._shape = _normalized_shape(tuple(series.shape))
                self._dtype = np.dtype(series.dtype)
                self._metadata = {
                    "format": "tiff",
                    "axes": getattr(series, "axes", None),
                    "series_shape": tuple(int(dim) for dim in series.shape),
                    "imagej_metadata": tif.imagej_metadata or {},
                }
            return

        if self.format == "v3draw":
            self._shape, self._dtype, self._metadata = _read_v3draw_header(self.filepath)
            return

        if self.format == "v3dpbd":
            self._shape, self._dtype, self._metadata = _read_v3dpbd_header(self.filepath)
            return

        if self.format == "nifti":
            nib_module = _require_optional_dependency(nib, "nibabel", "NIfTI")
            image = nib_module.load(str(self.filepath))
            self._shape = _normalized_shape(tuple(image.shape))
            self._dtype = np.dtype(image.get_data_dtype())
            self._metadata = {
                "format": "nifti",
                "zooms": tuple(float(value) for value in image.header.get_zooms()[: image.ndim]),
            }
            return

        if self.format == "nrrd":
            nrrd_module = _require_optional_dependency(nrrd, "pynrrd", "NRRD")
            volume, header = nrrd_module.read(str(self.filepath), index_order="C")
            self._cached_volume = _normalized_volume(np.asarray(volume))
            self._shape = self._cached_volume.shape
            self._dtype = self._cached_volume.dtype
            self._metadata = {
                "format": "nrrd",
                "encoding": header.get("encoding"),
                "space": header.get("space"),
                "space_directions": header.get("space directions"),
                "sizes": tuple(int(dim) for dim in header.get("sizes", self._cached_volume.shape)),
            }
            return

        raise ValueError(f"Unsupported image format for {self.filepath!s}.")

    @property
    def shape(self) -> tuple[int, int, int]:
        assert self._shape is not None
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        assert self._dtype is not None
        return self._dtype

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def load(self) -> np.ndarray:
        _vprint(self.verbose, f"Loading {self.filepath} as {self.format}")
        if self.format == "tiff":
            return _normalized_volume(np.asarray(tifffile.imread(self.filepath)))

        if self.format == "v3draw":
            return load_v3draw(self.filepath)

        if self.format == "v3dpbd":
            return PBD().load_image(self.filepath)

        if self.format == "nifti":
            nib_module = _require_optional_dependency(nib, "nibabel", "NIfTI")
            image = nib_module.load(str(self.filepath))
            return _normalized_volume(np.asanyarray(image.dataobj))

        if self.format == "nrrd":
            if self._cached_volume is not None:
                return self._cached_volume
            nrrd_module = _require_optional_dependency(nrrd, "pynrrd", "NRRD")
            volume, _header = nrrd_module.read(str(self.filepath), index_order="C")
            return _normalized_volume(np.asarray(volume))

        raise ValueError(f"Unsupported image format for {self.filepath!s}.")

    @staticmethod
    def save(
        image: np.ndarray,
        out_path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        affine: np.ndarray | None = None,
        verbose: int = 0,
    ) -> None:
        """Save a 3D volume, choosing the output format from `out_path` suffix.

        Format-specific arguments:
        - `affine` is only used for NIfTI outputs and defaults to the identity matrix.
        """
        out_path = Path(out_path)
        if out_path.exists() and not overwrite:
            raise OSError(f"File exists: {out_path}")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        volume = _normalized_volume(np.asarray(image))
        fmt = _detect_format(out_path)

        if fmt == "tiff":
            tifffile.imwrite(out_path, volume)
            _vprint(verbose, f"Saved TIFF volume to {out_path}")
            return

        if fmt == "v3draw":
            save_v3draw(volume, out_path)
            _vprint(verbose, f"Saved Vaa3D raw volume to {out_path}")
            return

        if fmt == "v3dpbd":
            save_v3dpbd(volume, out_path)
            _vprint(verbose, f"Saved Vaa3D PBD volume to {out_path}")
            return

        if fmt == "nifti":
            nib_module = _require_optional_dependency(nib, "nibabel", "NIfTI")
            nii = nib_module.Nifti1Image(volume, np.eye(4) if affine is None else affine)
            nib_module.save(nii, str(out_path))
            _vprint(verbose, f"Saved NIfTI volume to {out_path}")
            return

        if fmt == "nrrd":
            nrrd_module = _require_optional_dependency(nrrd, "pynrrd", "NRRD")
            nrrd_module.write(str(out_path), volume, index_order="C")
            _vprint(verbose, f"Saved NRRD volume to {out_path}")
            return

        raise ValueError(f"Saving {fmt!r} volumes is not supported.")

    @staticmethod
    def convert(
        input_path: str | os.PathLike[str],
        output_path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        affine: np.ndarray | None = None,
        verbose: int = 0,
    ) -> None:
        parser = ImageParser(input_path, verbose=verbose)
        ImageParser.save(
            parser.load(),
            output_path,
            overwrite=overwrite,
            affine=affine,
            verbose=verbose,
        )
