"""Dependency-free image metadata readers used during input preparation."""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JpegMetadata:
    stored_width: int
    stored_height: int
    exif_orientation: int

    @property
    def display_size(self) -> tuple[int, int]:
        if self.exif_orientation in {5, 6, 7, 8}:
            return self.stored_height, self.stored_width
        return self.stored_width, self.stored_height


@dataclass(frozen=True)
class PngAlphaMetadata:
    width: int
    height: int
    color_type: int
    transparent_pixels: int
    opaque_pixels: int
    partial_pixels: int

    @property
    def has_alpha(self) -> bool:
        return self.color_type in {4, 6}

    @property
    def nonzero_fraction(self) -> float:
        total = self.transparent_pixels + self.opaque_pixels + self.partial_pixels
        return (self.opaque_pixels + self.partial_pixels) / total if total else 0.0


def read_jpeg_metadata(path: Path) -> JpegMetadata:
    data = path.read_bytes()
    if data[:2] != b"\xff\xd8":
        raise ValueError(f"Not a JPEG file: {path}")

    width = height = None
    orientation = 1
    offset = 2
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        marker = data[offset + 1]
        offset += 2
        if marker in {0xD8, 0xD9}:
            continue
        if marker == 0xDA:
            break
        segment_length = struct.unpack(">H", data[offset : offset + 2])[0]
        segment = data[offset + 2 : offset + segment_length]
        offset += segment_length

        if marker == 0xE1 and segment.startswith(b"Exif\x00\x00"):
            orientation = _read_tiff_orientation(segment[6:]) or orientation
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            height, width = struct.unpack(">HH", segment[1:5])

    if width is None or height is None:
        raise ValueError(f"JPEG dimensions not found: {path}")
    return JpegMetadata(width, height, orientation)


def _read_tiff_orientation(tiff: bytes) -> int | None:
    if len(tiff) < 8 or tiff[:2] not in {b"II", b"MM"}:
        return None
    endian = "<" if tiff[:2] == b"II" else ">"
    if struct.unpack(f"{endian}H", tiff[2:4])[0] != 42:
        return None
    ifd_offset = struct.unpack(f"{endian}I", tiff[4:8])[0]
    if ifd_offset + 2 > len(tiff):
        return None
    count = struct.unpack(f"{endian}H", tiff[ifd_offset : ifd_offset + 2])[0]
    for index in range(count):
        entry_offset = ifd_offset + 2 + index * 12
        entry = tiff[entry_offset : entry_offset + 12]
        if len(entry) < 12:
            break
        tag, value_type, value_count = struct.unpack(f"{endian}HHI", entry[:8])
        if tag == 0x0112 and value_type == 3 and value_count == 1:
            return struct.unpack(f"{endian}H", entry[8:10])[0]
    return None


def read_png_alpha_metadata(path: Path) -> PngAlphaMetadata:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG file: {path}")

    chunks: list[bytes] = []
    width = height = bit_depth = color_type = None
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type = struct.unpack(">IIBB", chunk_data[:10])
            if chunk_data[12] != 0:
                raise ValueError(f"Interlaced PNG is not supported: {path}")
        elif chunk_type == b"IDAT":
            chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    if None in {width, height, bit_depth, color_type}:
        raise ValueError(f"PNG header not found: {path}")
    if bit_depth != 8 or color_type not in {4, 6}:
        raise ValueError(f"Expected an 8-bit grayscale/RGBA PNG with alpha: {path}")

    channels = 2 if color_type == 4 else 4
    stride = width * channels
    raw = zlib.decompress(b"".join(chunks))
    previous = bytearray(stride)
    transparent = opaque = partial = 0
    raw_offset = 0
    for _ in range(height):
        filter_type = raw[raw_offset]
        scanline = bytearray(raw[raw_offset + 1 : raw_offset + 1 + stride])
        raw_offset += stride + 1
        _unfilter(scanline, previous, channels, filter_type)
        alpha_offset = 1 if color_type == 4 else 3
        for alpha in scanline[alpha_offset::channels]:
            if alpha == 0:
                transparent += 1
            elif alpha == 255:
                opaque += 1
            else:
                partial += 1
        previous = scanline

    return PngAlphaMetadata(width, height, color_type, transparent, opaque, partial)


def _unfilter(row: bytearray, previous: bytearray, bpp: int, filter_type: int) -> None:
    for index in range(len(row)):
        left = row[index - bpp] if index >= bpp else 0
        up = previous[index]
        upper_left = previous[index - bpp] if index >= bpp else 0
        if filter_type == 1:
            row[index] = (row[index] + left) & 0xFF
        elif filter_type == 2:
            row[index] = (row[index] + up) & 0xFF
        elif filter_type == 3:
            row[index] = (row[index] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            row[index] = (row[index] + _paeth(left, up, upper_left)) & 0xFF
        elif filter_type != 0:
            raise ValueError(f"Unsupported PNG filter: {filter_type}")


def _paeth(left: int, up: int, upper_left: int) -> int:
    prediction = left + up - upper_left
    left_distance = abs(prediction - left)
    up_distance = abs(prediction - up)
    upper_left_distance = abs(prediction - upper_left)
    if left_distance <= up_distance and left_distance <= upper_left_distance:
        return left
    if up_distance <= upper_left_distance:
        return up
    return upper_left
