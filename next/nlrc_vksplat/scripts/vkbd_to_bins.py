#!/usr/bin/env python3
"""Convert ref .vkbd dumps to raw .bin fixture payloads."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

HEADER = struct.Struct("<8sII6Q")
MAGIC = b"VKBDUMP\0"

DTYPE_SIZES = {
    "uint8": 1,
    "int32": 4,
    "uint32": 4,
    "int64": 8,
    "uint64": 8,
    "float32": 4,
}


def parse_shape(value: str | None) -> list[int] | None:
    if value is None or value == "":
        return None
    parts = value.replace("x", ",").split(",")
    try:
        shape = [int(part.strip()) for part in parts if part.strip()]
    except ValueError as exc:
        raise ValueError(f"invalid --shape {value!r}") from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError("--shape must contain positive dimensions")
    return shape


def checked_element_count(shape: list[int]) -> int:
    count = 1
    for dim in shape:
        count *= dim
    return count


def read_vkbd(path: Path) -> tuple[dict, bytes, dict]:
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: file is too small for a VKBD header")

    (
        magic,
        version,
        header_size,
        metadata_json_bytes,
        payload_offset,
        payload_bytes,
        logical_bytes,
        alloc_bytes,
        flags,
    ) = HEADER.unpack_from(data, 0)

    if magic != MAGIC:
        raise ValueError(f"{path}: invalid VKBD magic {magic!r}")
    if version != 1:
        raise ValueError(f"{path}: unsupported VKBD version {version}")
    if header_size != HEADER.size:
        raise ValueError(f"{path}: unexpected VKBD header size {header_size}")

    metadata_start = header_size
    metadata_end = metadata_start + metadata_json_bytes
    payload_end = payload_offset + payload_bytes
    if metadata_end > len(data) or payload_end > len(data):
        raise ValueError(f"{path}: truncated VKBD file")
    if payload_offset < metadata_end:
        raise ValueError(f"{path}: VKBD payload overlaps metadata")

    metadata = json.loads(data[metadata_start:metadata_end].decode("utf-8"))
    payload = data[payload_offset:payload_end]
    header = {
        "version": version,
        "header_size": header_size,
        "metadata_json_bytes": metadata_json_bytes,
        "payload_offset": payload_offset,
        "payload_bytes": payload_bytes,
        "logical_bytes": logical_bytes,
        "alloc_bytes": alloc_bytes,
        "flags": flags,
    }
    return metadata, payload, header


def validate_payload(path: Path, metadata: dict, payload: bytes, dtype: str | None, shape: list[int] | None) -> None:
    metadata_dtype = metadata.get("dtype")
    if dtype is not None:
        if dtype not in DTYPE_SIZES:
            raise ValueError(f"unsupported dtype {dtype!r}")
        if metadata_dtype is not None and metadata_dtype != dtype:
            raise ValueError(f"{path}: dtype mismatch {metadata_dtype!r} != {dtype!r}")
    elif metadata_dtype is not None and metadata_dtype not in DTYPE_SIZES:
        raise ValueError(f"{path}: unsupported metadata dtype {metadata_dtype!r}")

    metadata_shape = metadata.get("shape")
    if shape is not None:
        if metadata_shape is not None and [int(dim) for dim in metadata_shape] != shape:
            raise ValueError(f"{path}: shape mismatch {metadata_shape!r} != {shape!r}")

    checked_dtype = dtype or metadata_dtype
    checked_shape = shape or ([int(dim) for dim in metadata_shape] if metadata_shape else None)
    if checked_dtype is None or checked_shape is None:
        return

    expected_bytes = checked_element_count(checked_shape) * DTYPE_SIZES[checked_dtype]
    if len(payload) != expected_bytes:
        raise ValueError(f"{path}: payload has {len(payload)} bytes, expected {expected_bytes}")


def convert(args: argparse.Namespace) -> int:
    vkbd_path = Path(args.vkbd)
    out_path = Path(args.out)
    shape = parse_shape(args.shape)

    metadata, payload, _header = read_vkbd(vkbd_path)
    validate_payload(vkbd_path, metadata, payload, args.dtype, shape)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(payload)
    print(f"[OK] Wrote {out_path} ({len(payload)} bytes)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vkbd", required=True, help="Source .vkbd path")
    parser.add_argument("--out", required=True, help="Destination raw .bin path")
    parser.add_argument("--dtype", choices=sorted(DTYPE_SIZES), default=None, help="Optional dtype validation")
    parser.add_argument("--shape", default=None, help="Optional shape validation, e.g. 4 or 2,3")
    args = parser.parse_args()

    try:
        return convert(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
