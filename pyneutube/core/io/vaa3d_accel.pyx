# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython accelerators for Vaa3D image loading."""

from __future__ import annotations

import os
import struct

import numpy as np
cimport numpy as np
from cpython.bytearray cimport PyByteArray_AsString
from cpython.bytes cimport PyBytes_AS_STRING, PyBytes_GET_SIZE
from libc.stdint cimport uint8_t, uint16_t


DEF V3DRAW_HEADER_SIZE = 43
DEF V3DPBD_HEADER_SIZE = 43

FORMAT_KEY_V3DRAW = b"raw_image_stack_by_hpeng"
FORMAT_KEY_V3DPBD = b"v3d_volume_pkbitdf_encod"


cdef class _PBDStreamWriter:
    cdef:
        object handle
        bytearray buffer
        uint8_t* data
        Py_ssize_t pos
        Py_ssize_t capacity

    def __cinit__(self, object handle, Py_ssize_t capacity=1024 * 1024):
        self.handle = handle
        self.buffer = bytearray(capacity)
        self.data = <uint8_t*>PyByteArray_AsString(self.buffer)
        self.pos = 0
        self.capacity = capacity

    cdef void flush(self):
        if self.pos > 0:
            self.handle.write(self.buffer[:self.pos])
            self.pos = 0

    cdef inline void write_byte(self, uint8_t value):
        if self.pos == self.capacity:
            self.flush()
        self.data[self.pos] = value
        self.pos += 1


cdef void _flush_literal8(
    _PBDStreamWriter writer,
    uint8_t* literal,
    Py_ssize_t count,
):
    cdef Py_ssize_t index
    if count == 0:
        return
    writer.write_byte(<uint8_t>(count - 1))
    for index in range(count):
        writer.write_byte(literal[index])


cdef void _compress_pbd8(
    const uint8_t* src,
    Py_ssize_t length,
    _PBDStreamWriter writer,
):
    cdef:
        Py_ssize_t pos = 0
        Py_ssize_t scan
        Py_ssize_t run_len
        Py_ssize_t delta_len
        Py_ssize_t delta_bytes
        Py_ssize_t literal_count = 0
        Py_ssize_t index
        Py_ssize_t group_index
        int delta
        int prior
        double run_efficiency
        double delta_efficiency
        uint8_t value
        uint8_t packed
        uint8_t literal[33]
        uint8_t delta_codes[95]

    while pos < length:
        value = src[pos]
        run_len = 1
        while (
            run_len < 128
            and pos + run_len < length
            and src[pos + run_len] == value
        ):
            run_len += 1

        delta_len = 0
        if pos > 0:
            prior = src[pos - 1]
            scan = pos
            while delta_len < 95 and scan < length:
                delta = <int>src[scan] - prior
                if delta < -1 or delta > 2:
                    break
                delta_codes[delta_len] = 3 if delta == -1 else <uint8_t>delta
                prior = src[scan]
                delta_len += 1
                scan += 1

        run_efficiency = run_len / 2.0
        delta_bytes = 1 + ((delta_len + 3) >> 2)
        delta_efficiency = delta_len / <double>delta_bytes if delta_len > 0 else 0.0

        if run_efficiency > 1.0 and run_efficiency >= delta_efficiency:
            _flush_literal8(writer, literal, literal_count)
            literal_count = 0
            writer.write_byte(<uint8_t>(run_len + 127))
            writer.write_byte(value)
            pos += run_len
            continue

        if delta_efficiency > 1.0:
            _flush_literal8(writer, literal, literal_count)
            literal_count = 0
            writer.write_byte(<uint8_t>(delta_len + 32))
            for index in range(0, delta_len, 4):
                packed = 0
                for group_index in range(4):
                    if index + group_index < delta_len:
                        packed |= delta_codes[index + group_index] << (2 * group_index)
                writer.write_byte(packed)
            pos += delta_len
            continue

        literal[literal_count] = value
        literal_count += 1
        pos += 1
        if literal_count == 33:
            _flush_literal8(writer, literal, literal_count)
            literal_count = 0

    _flush_literal8(writer, literal, literal_count)


cdef void _flush_literal16(
    _PBDStreamWriter writer,
    uint16_t* literal,
    Py_ssize_t count,
):
    cdef:
        Py_ssize_t index
        uint16_t value
    if count == 0:
        return
    writer.write_byte(<uint8_t>(count - 1))
    for index in range(count):
        value = literal[index]
        writer.write_byte(<uint8_t>(value & 0xFF))
        writer.write_byte(<uint8_t>(value >> 8))


cdef void _compress_pbd16(
    const uint16_t* src,
    Py_ssize_t length,
    _PBDStreamWriter writer,
):
    cdef:
        Py_ssize_t pos = 0
        Py_ssize_t scan
        Py_ssize_t run_len
        Py_ssize_t literal_count = 0
        Py_ssize_t index
        Py_ssize_t delta_bytes
        Py_ssize_t delta_lengths[3]
        Py_ssize_t max_lengths[3]
        int code_offsets[3]
        int mode
        int best_mode
        int bits
        int threshold
        int delta
        int bit_count
        unsigned int accumulator
        double run_efficiency
        double delta_efficiency
        double best_delta_efficiency
        uint16_t value
        uint16_t prior
        uint16_t literal[32]
        uint8_t delta_codes[3][103]

    max_lengths[0] = 48
    max_lengths[1] = 103
    max_lengths[2] = 40
    code_offsets[0] = 31
    code_offsets[1] = 79
    code_offsets[2] = 182

    while pos < length:
        value = src[pos]
        run_len = 1
        while (
            run_len < 33
            and pos + run_len < length
            and src[pos + run_len] == value
        ):
            run_len += 1

        best_mode = -1
        best_delta_efficiency = 0.0
        if pos > 0:
            for mode in range(3):
                bits = mode + 3
                threshold = 1 << (bits - 1)
                prior = src[pos - 1]
                scan = pos
                delta_lengths[mode] = 0
                while delta_lengths[mode] < max_lengths[mode] and scan < length:
                    delta = <int>src[scan] - <int>prior
                    if delta < 1 - threshold or delta > threshold:
                        break
                    delta_codes[mode][delta_lengths[mode]] = (
                        <uint8_t>(threshold - delta) if delta < 0 else <uint8_t>delta
                    )
                    prior = src[scan]
                    delta_lengths[mode] += 1
                    scan += 1

                delta_bytes = 1 + ((delta_lengths[mode] * bits + 7) >> 3)
                delta_efficiency = delta_lengths[mode] / <double>delta_bytes
                if delta_efficiency > best_delta_efficiency:
                    best_delta_efficiency = delta_efficiency
                    best_mode = mode

        run_efficiency = run_len / 3.0
        if run_efficiency > 1.0 and run_efficiency >= best_delta_efficiency:
            _flush_literal16(writer, literal, literal_count)
            literal_count = 0
            writer.write_byte(<uint8_t>(run_len + 222))
            writer.write_byte(<uint8_t>(value & 0xFF))
            writer.write_byte(<uint8_t>(value >> 8))
            pos += run_len
            continue

        if best_delta_efficiency > 1.0:
            _flush_literal16(writer, literal, literal_count)
            literal_count = 0
            bits = best_mode + 3
            writer.write_byte(<uint8_t>(delta_lengths[best_mode] + code_offsets[best_mode]))
            accumulator = 0
            bit_count = 0
            for index in range(delta_lengths[best_mode]):
                accumulator = (accumulator << bits) | delta_codes[best_mode][index]
                bit_count += bits
                while bit_count >= 8:
                    bit_count -= 8
                    writer.write_byte(<uint8_t>((accumulator >> bit_count) & 0xFF))
                    if bit_count == 0:
                        accumulator = 0
                    else:
                        accumulator &= (1 << bit_count) - 1
            if bit_count > 0:
                writer.write_byte(<uint8_t>((accumulator << (8 - bit_count)) & 0xFF))
            pos += delta_lengths[best_mode]
            continue

        literal[literal_count] = value
        literal_count += 1
        pos += 1
        if literal_count == 32:
            _flush_literal16(writer, literal, literal_count)
            literal_count = 0

    _flush_literal16(writer, literal, literal_count)


cdef inline uint16_t _read_u16(const uint8_t* data, Py_ssize_t pos, bint little) noexcept:
    if little:
        return <uint16_t>(data[pos] | (data[pos + 1] << 8))
    return <uint16_t>((data[pos] << 8) | data[pos + 1])


cdef inline int _pbd_delta(unsigned int encoded, unsigned int threshold) noexcept:
    if encoded > threshold:
        return <int>threshold - <int>encoded
    return <int>encoded


cdef void _decompress_pbd8(
    const uint8_t* comp,
    Py_ssize_t comp_len,
    uint8_t* out,
    Py_ssize_t out_len,
):
    cdef:
        Py_ssize_t cp = 0
        Py_ssize_t dp = 0
        Py_ssize_t count
        Py_ssize_t j
        unsigned int code
        unsigned int packed = 0
        unsigned int delta_code
        int delta
        int prior = 0
        int value
        uint8_t repeat_value

    while cp < comp_len:
        code = comp[cp]
        cp += 1

        if code < 33:
            count = code + 1
            if dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD8 literal block.")
            if cp + count > comp_len:
                break
            for j in range(count):
                out[dp + j] = comp[cp + j]
            prior = out[dp + count - 1]
            dp += count
            cp += count
        elif code < 128:
            count = code - 32
            if dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD8 delta block.")
            if cp + ((count + 3) >> 2) > comp_len:
                break
            for j in range(count):
                if (j & 3) == 0:
                    packed = comp[cp]
                    cp += 1
                delta_code = (packed >> (2 * (j & 3))) & 3
                delta = -1 if delta_code == 3 else <int>delta_code
                value = prior + delta
                out[dp] = <uint8_t>value
                prior = out[dp]
                dp += 1
        else:
            count = code - 127
            if dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD8 repeat block.")
            if cp >= comp_len:
                break
            repeat_value = comp[cp]
            cp += 1
            for j in range(count):
                out[dp + j] = repeat_value
            prior = repeat_value
            dp += count

cdef void _decompress_pbd16(
    const uint8_t* comp,
    Py_ssize_t comp_len,
    uint16_t* out,
    Py_ssize_t out_len,
    bint little,
):
    cdef:
        Py_ssize_t cp = 0
        Py_ssize_t dp = 0
        Py_ssize_t count
        Py_ssize_t j
        Py_ssize_t k
        Py_ssize_t bitpos
        Py_ssize_t byte_count
        unsigned int code
        unsigned int bits
        unsigned int threshold
        unsigned int delta_code
        int delta
        int value
        unsigned int prior = 0
        uint16_t repeat_value

    while cp < comp_len:
        code = comp[cp]
        cp += 1

        if code < 32:
            count = code + 1
            if dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD16 literal block.")
            if cp + count * 2 > comp_len:
                break
            for j in range(count):
                out[dp + j] = _read_u16(comp, cp + j * 2, little)
            prior = out[dp + count - 1]
            dp += count
            cp += count * 2
            continue

        if code >= 223:
            count = code - 222
            if dp + count > out_len:
                raise ValueError("Malformed Vaa3D PBD16 repeat block.")
            if cp + 2 > comp_len:
                break
            repeat_value = _read_u16(comp, cp, little)
            cp += 2
            for j in range(count):
                out[dp + j] = repeat_value
            prior = repeat_value
            dp += count
            continue

        if code < 80:
            count = code - 31
            bits = 3
            threshold = 4
        elif code < 183:
            count = code - 79
            bits = 4
            threshold = 8
        else:
            count = code - 182
            bits = 5
            threshold = 16

        byte_count = (count * bits + 7) >> 3
        if dp + count > out_len:
            raise ValueError("Malformed Vaa3D PBD16 delta block.")
        if cp + byte_count > comp_len:
            break

        bitpos = 0
        for j in range(count):
            delta_code = 0
            for k in range(bits):
                delta_code = (
                    (delta_code << 1)
                    | ((comp[cp + (bitpos >> 3)] >> (7 - (bitpos & 7))) & 1)
                )
                bitpos += 1
            delta = _pbd_delta(delta_code, threshold)
            value = <int>prior + delta
            out[dp] = <uint16_t>(value & 0xFFFF)
            prior = out[dp]
            dp += 1

        cp += byte_count

def load_v3draw(path):
    """Load a Vaa3D v3draw/raw file using direct NumPy file reads."""
    cdef:
        bytes header
        str endian
        short datatype
        tuple dims
        object dtype
        object native_dtype
        object image
        long long total_units
        long long expected_size

    path = os.fspath(path)
    with open(path, "rb") as handle:
        header = handle.read(V3DRAW_HEADER_SIZE)

    if len(header) < V3DRAW_HEADER_SIZE:
        raise ValueError(f"Vaa3D raw file is truncated: {path}")
    if header[:len(FORMAT_KEY_V3DRAW)] != FORMAT_KEY_V3DRAW:
        raise ValueError(f"Unsupported Vaa3D raw header in {path}.")

    if header[len(FORMAT_KEY_V3DRAW):len(FORMAT_KEY_V3DRAW) + 1] == b"B":
        endian = ">"
    elif header[len(FORMAT_KEY_V3DRAW):len(FORMAT_KEY_V3DRAW) + 1] == b"L":
        endian = "<"
    else:
        raise ValueError(f"Unsupported Vaa3D endian code in {path}.")

    datatype = struct.unpack(f"{endian}h", header[25:27])[0]
    dims = struct.unpack(f"{endian}iiii", header[27:43])
    if datatype == 1:
        dtype = np.dtype("u1")
    elif datatype == 2:
        dtype = np.dtype(f"{endian}u2")
    elif datatype == 4:
        dtype = np.dtype(f"{endian}f4")
    else:
        raise ValueError(f"Unsupported Vaa3D datatype code {datatype!r}.")

    total_units = <long long>dims[0] * dims[1] * dims[2] * dims[3]
    expected_size = V3DRAW_HEADER_SIZE + total_units * np.dtype(dtype).itemsize
    if os.path.getsize(path) != expected_size:
        raise ValueError(f"Vaa3D raw file size does not match the header: {path}")

    image = np.fromfile(path, dtype=dtype, count=total_units, offset=V3DRAW_HEADER_SIZE)
    image = image.reshape((dims[3], dims[2], dims[1], dims[0]))
    native_dtype = np.dtype(dtype).newbyteorder("=")
    return image.astype(native_dtype, copy=False)


def load_v3dpbd(path):
    """Load a Vaa3D PBD-compressed file with C-level decompression loops."""
    cdef:
        bytes data
        const uint8_t* comp
        Py_ssize_t comp_len
        str endian
        bint little
        short datatype
        tuple dims
        Py_ssize_t total_units
        np.ndarray[np.uint8_t, ndim=1] out8
        np.ndarray[np.uint16_t, ndim=1] out16

    path = os.fspath(path)
    with open(path, "rb") as handle:
        data = handle.read()

    if PyBytes_GET_SIZE(data) < V3DPBD_HEADER_SIZE:
        raise ValueError(f"Vaa3D PBD file is truncated: {path}")
    if data[:len(FORMAT_KEY_V3DPBD)] != FORMAT_KEY_V3DPBD:
        raise ValueError(f"Unsupported Vaa3D PBD header in {path}.")

    if data[len(FORMAT_KEY_V3DPBD):len(FORMAT_KEY_V3DPBD) + 1] == b"B":
        endian = ">"
        little = False
    elif data[len(FORMAT_KEY_V3DPBD):len(FORMAT_KEY_V3DPBD) + 1] == b"L":
        endian = "<"
        little = True
    else:
        raise ValueError(f"Unsupported Vaa3D endian code in {path}.")

    datatype = struct.unpack(f"{endian}h", data[25:27])[0]
    dims = struct.unpack(f"{endian}iiii", data[27:43])
    total_units = <Py_ssize_t>dims[0] * dims[1] * dims[2] * dims[3]
    comp = <const uint8_t*>PyBytes_AS_STRING(data) + V3DPBD_HEADER_SIZE
    comp_len = PyBytes_GET_SIZE(data) - V3DPBD_HEADER_SIZE

    # Match Vaa3D: an undersized payload leaves the unwritten image tail at zero.
    # Block output that would exceed the header-declared size is still rejected.
    if datatype == 1:
        out8 = np.zeros(total_units, dtype=np.uint8)
        _decompress_pbd8(comp, comp_len, <uint8_t*>out8.data, total_units)
        return out8.reshape((dims[3], dims[2], dims[1], dims[0]))

    if datatype == 2:
        out16 = np.zeros(total_units, dtype=np.uint16)
        _decompress_pbd16(comp, comp_len, <uint16_t*>out16.data, total_units, little)
        return out16.reshape((dims[3], dims[2], dims[1], dims[0]))

    if datatype == 33:
        raise NotImplementedError("Vaa3D PBD datatype 33 is not implemented.")
    raise ValueError(f"Unsupported Vaa3D PBD datatype code {datatype!r}.")


def save_v3dpbd(image, path):
    """Save a 3D uint8 or uint16 volume using Vaa3D PBD compression."""
    cdef:
        object volume
        object handle
        bytes header
        _PBDStreamWriter writer
        const uint8_t* src8
        const uint16_t* src16
        Py_ssize_t total_units
        short datatype

    volume = np.asarray(image)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got array with shape {volume.shape!r}.")
    if volume.dtype == np.uint8:
        datatype = 1
    elif volume.dtype == np.uint16:
        datatype = 2
    else:
        raise ValueError("Vaa3D PBD saving supports uint8 and uint16 volumes only.")

    volume = np.ascontiguousarray(volume)
    total_units = volume.size
    header = (
        FORMAT_KEY_V3DPBD
        + b"L"
        + struct.pack(
            "<hiiii",
            datatype,
            volume.shape[2],
            volume.shape[1],
            volume.shape[0],
            1,
        )
    )

    path = os.fspath(path)
    with open(path, "wb") as handle:
        handle.write(header)
        writer = _PBDStreamWriter(handle)
        if datatype == 1:
            src8 = <const uint8_t*>np.PyArray_DATA(volume)
            _compress_pbd8(src8, total_units, writer)
        else:
            src16 = <const uint16_t*>np.PyArray_DATA(volume)
            _compress_pbd16(src16, total_units, writer)
        writer.flush()
