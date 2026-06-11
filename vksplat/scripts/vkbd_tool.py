#!/usr/bin/env python3
"""Read and compare VkSplat Vulkan buffer dump files."""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


HEADER = struct.Struct("<8sII6Q")
MAGIC = b"VKBDUMP\0"

DTYPES = {
    "uint8": np.uint8,
    "int32": np.int32,
    "uint32": np.uint32,
    "int64": np.int64,
    "uint64": np.uint64,
    "float32": np.float32,
}


@dataclass(frozen=True)
class VkbdFile:
    path: Path
    metadata: dict
    payload: bytes
    header: dict

    def array(self) -> np.ndarray:
        dtype_name = self.metadata["dtype"]
        if dtype_name not in DTYPES:
            raise ValueError(f"Unsupported dtype {dtype_name!r} in {self.path}")
        array = np.frombuffer(self.payload, dtype=DTYPES[dtype_name])
        shape = tuple(int(dim) for dim in self.metadata.get("shape", []))
        if shape and int(np.prod(shape, dtype=np.int64)) == array.size:
            return array.reshape(shape)
        return array


def read_vkbd(path: Path) -> VkbdFile:
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
        raise ValueError(f"{path}: unexpected header size {header_size}")

    metadata_start = header_size
    metadata_end = metadata_start + metadata_json_bytes
    payload_end = payload_offset + payload_bytes
    if metadata_end > len(data) or payload_end > len(data):
        raise ValueError(f"{path}: truncated VKBD file")
    if payload_offset < metadata_end:
        raise ValueError(f"{path}: payload overlaps metadata")

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
    if len(payload) != payload_bytes:
        raise ValueError(f"{path}: payload byte count mismatch")
    return VkbdFile(path=path, metadata=metadata, payload=payload, header=header)


def iter_vkbd_files(root: Path, include_capacity: bool) -> Iterable[Path]:
    for path in sorted(root.rglob("*.vkbd")):
        if not include_capacity and path.name.endswith(".capacity.vkbd"):
            continue
        yield path


def compare_file(expected_path: Path, actual_path: Path, rtol: float, atol: float) -> str | None:
    expected = read_vkbd(expected_path)
    actual = read_vkbd(actual_path)

    keys = ["buffer_name", "dtype", "shape", "payload_kind", "logical_bytes", "payload_bytes"]
    for key in keys:
        if expected.metadata.get(key) != actual.metadata.get(key):
            return f"metadata mismatch for {key}: {expected.metadata.get(key)!r} != {actual.metadata.get(key)!r}"

    dtype = expected.metadata["dtype"]
    if dtype == "float32":
        expected_array = expected.array()
        actual_array = actual.array()
        if expected_array.shape != actual_array.shape:
            return f"shape mismatch: {expected_array.shape} != {actual_array.shape}"
        if not np.allclose(expected_array, actual_array, rtol=rtol, atol=atol, equal_nan=True):
            diff = np.abs(expected_array - actual_array)
            return f"float mismatch: max_abs_diff={float(np.nanmax(diff))}"
        return None

    if expected.payload != actual.payload:
        return "byte mismatch"
    return None


def compare_dirs(args: argparse.Namespace) -> int:
    expected_root = Path(args.expected)
    actual_root = Path(args.actual)
    expected_files = {
        path.relative_to(expected_root): path
        for path in iter_vkbd_files(expected_root, args.include_capacity)
    }
    actual_files = {
        path.relative_to(actual_root): path
        for path in iter_vkbd_files(actual_root, args.include_capacity)
    }

    failures: list[str] = []
    for rel in sorted(expected_files.keys() - actual_files.keys()):
        failures.append(f"missing actual file: {rel.as_posix()}")
    for rel in sorted(actual_files.keys() - expected_files.keys()):
        failures.append(f"unexpected actual file: {rel.as_posix()}")
    for rel in sorted(expected_files.keys() & actual_files.keys()):
        result = compare_file(expected_files[rel], actual_files[rel], args.rtol, args.atol)
        if result:
            failures.append(f"{rel.as_posix()}: {result}")

    if failures:
        for failure in failures[: args.max_failures]:
            print(failure)
        if len(failures) > args.max_failures:
            print(f"... {len(failures) - args.max_failures} more failures")
        return 1

    print(f"OK: compared {len(expected_files)} VKBD files")
    return 0


def inspect_file(args: argparse.Namespace) -> int:
    vkbd = read_vkbd(Path(args.file))
    print(json.dumps({"header": vkbd.header, "metadata": vkbd.metadata}, indent=2))
    if args.array:
        array = vkbd.array()
        print(f"array shape={array.shape} dtype={array.dtype}")
        if array.size:
            print(f"min={np.nanmin(array)} max={np.nanmax(array)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Inspect one .vkbd file")
    inspect_parser.add_argument("file")
    inspect_parser.add_argument("--array", action="store_true", help="Also materialize and summarize the NumPy array")
    inspect_parser.set_defaults(func=inspect_file)

    compare_parser = subparsers.add_parser("compare", help="Compare two dump directories")
    compare_parser.add_argument("expected")
    compare_parser.add_argument("actual")
    compare_parser.add_argument("--rtol", type=float, default=1e-5)
    compare_parser.add_argument("--atol", type=float, default=1e-6)
    compare_parser.add_argument("--include-capacity", action="store_true")
    compare_parser.add_argument("--max-failures", type=int, default=50)
    compare_parser.set_defaults(func=compare_dirs)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
