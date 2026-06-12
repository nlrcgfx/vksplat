#!/usr/bin/env python3
"""Generate deterministic synthetic fixture and golden test data."""

from __future__ import annotations

import argparse
import filecmp
import json
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

REF_BASELINE_TAG = "ref-baseline-2026-06-12"
DEFAULT_CMAKE_PRESET = "windows-debug"
CUMSUM_BLOCK_SIZE = 512
SUM_BLOCK_SIZE = 512
WHERE_BLOCK_SIZE = 256

DTYPE_FORMATS = {
    "float32": "f",
    "uint32": "I",
    "int32": "i",
    "int64": "q",
}

GENERATED_SUBDIRS = ("fixtures", "golden_masters")


@dataclass(frozen=True)
class BufferData:
    name: str
    dtype: str
    shape: tuple[int, ...]
    values: tuple[int | float, ...]
    epsilon: float | None = None

    @property
    def file_name(self) -> str:
        return f"{self.name}.bin"


@dataclass(frozen=True)
class FixtureCase:
    stage_name: str
    subgraph: str
    fixture_bindings: tuple[str, ...]
    fixture_buffers: tuple[BufferData, ...]
    golden_bindings: tuple[str, ...]
    golden_buffers: tuple[BufferData, ...]
    fixture_notes: str
    golden_notes: str


def element_count(shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        if dim <= 0:
            raise ValueError(f"shape dimensions must be positive, got {dim}")
        count *= dim
    return count


def prefix_sum(values: Iterable[int]) -> tuple[int, ...]:
    total = 0
    result: list[int] = []
    for value in values:
        total += value
        result.append(total)
    return tuple(result)


def mask_cumsum(mask: Iterable[int]) -> tuple[int, ...]:
    total = 0
    result: list[int] = []
    for value in mask:
        if value > 0:
            total += 1
        result.append(total)
    return tuple(result)


def where_indices(mask: Iterable[int]) -> tuple[int, ...]:
    return tuple(index for index, value in enumerate(mask) if value > 0)


def buffer_data(name: str, dtype: str, values: Iterable[int | float], epsilon: float | None = None) -> BufferData:
    materialized = tuple(values)
    return BufferData(name=name, dtype=dtype, shape=(len(materialized),), values=materialized, epsilon=epsilon)


def cumsum_case(stage_name: str, values: Iterable[int], notes: str) -> FixtureCase:
    input_values = tuple(values)
    expected = prefix_sum(input_values)
    return FixtureCase(
        stage_name=stage_name,
        subgraph="D",
        fixture_bindings=("input", "output", "block_sums"),
        fixture_buffers=(
            buffer_data("input", "int32", input_values),
            buffer_data("output", "int32", [0] * len(input_values)),
            buffer_data("block_sums", "int32", [0]),
        ),
        golden_bindings=("output",),
        golden_buffers=(buffer_data("output", "int32", expected),),
        fixture_notes=f"Synthetic Subgraph D utility fixture for {notes}",
        golden_notes=f"Synthetic Subgraph D utility golden for {notes}",
    )


def cumsum_multi_block_case(stage_name: str, values: Iterable[int], notes: str, two_level: bool = False) -> FixtureCase:
    input_values = tuple(values)
    expected = prefix_sum(input_values)
    block_sums_count = (len(input_values) + CUMSUM_BLOCK_SIZE - 1) // CUMSUM_BLOCK_SIZE
    buffers = [
        buffer_data("input", "int32", input_values),
        buffer_data("output", "int32", [0] * len(input_values)),
        buffer_data("block_sums", "int32", [0] * block_sums_count),
    ]
    bindings = ["input", "output", "block_sums"]

    if two_level:
        block_sums2_count = (block_sums_count + CUMSUM_BLOCK_SIZE - 1) // CUMSUM_BLOCK_SIZE
        buffers.append(buffer_data("block_sums2", "int32", [0] * block_sums2_count))
        bindings.append("block_sums2")

    return FixtureCase(
        stage_name=stage_name,
        subgraph="D",
        fixture_bindings=tuple(bindings),
        fixture_buffers=tuple(buffers),
        golden_bindings=("output",),
        golden_buffers=(buffer_data("output", "int32", expected),),
        fixture_notes=f"Synthetic Subgraph D utility fixture for {notes}",
        golden_notes=f"Synthetic Subgraph D utility golden for {notes}",
    )


def sum_case(stage_name: str, values: Iterable[int], notes: str) -> FixtureCase:
    input_values = tuple(values)
    return FixtureCase(
        stage_name=stage_name,
        subgraph="D",
        fixture_bindings=("input", "output"),
        fixture_buffers=(
            buffer_data("input", "int32", input_values),
            buffer_data("output", "int32", [0]),
        ),
        golden_bindings=("output",),
        golden_buffers=(buffer_data("output", "int32", [sum(input_values)]),),
        fixture_notes=f"Synthetic Subgraph D utility fixture for {notes}",
        golden_notes=f"Synthetic Subgraph D utility golden for {notes}",
    )


def where_case(stage_name: str, mask: Iterable[int], initial_fill: int, notes: str) -> FixtureCase:
    mask_values = tuple(mask)
    indices = where_indices(mask_values)
    output_len = max(1, len(indices))
    expected = indices if indices else (initial_fill,)
    return FixtureCase(
        stage_name=stage_name,
        subgraph="D",
        fixture_bindings=("mask", "mask_cumsum", "out_indices"),
        fixture_buffers=(
            buffer_data("mask", "int32", mask_values),
            buffer_data("mask_cumsum", "int32", mask_cumsum(mask_values)),
            buffer_data("out_indices", "int32", [initial_fill] * output_len),
        ),
        golden_bindings=("out_indices",),
        golden_buffers=(buffer_data("out_indices", "int32", expected),),
        fixture_notes=f"Synthetic Subgraph D utility fixture for {notes}",
        golden_notes=f"Synthetic Subgraph D utility golden for {notes}",
    )


def fixture_cases() -> tuple[FixtureCase, ...]:
    near_block_values = tuple((index % 7) - 3 for index in range(CUMSUM_BLOCK_SIZE - 1))
    exact_block_values = tuple(1 for _ in range(CUMSUM_BLOCK_SIZE))
    multi_block_values = tuple((index % 11) - 5 for index in range(CUMSUM_BLOCK_SIZE + 1))
    two_level_values = tuple((index % 5) - 2 for index in range((CUMSUM_BLOCK_SIZE * CUMSUM_BLOCK_SIZE) + 1))
    where_boundary_mask = tuple(1 if index in (WHERE_BLOCK_SIZE - 1, WHERE_BLOCK_SIZE) else 0 for index in range(WHERE_BLOCK_SIZE + 1))

    return (
        FixtureCase(
            stage_name="harness_smoke",
            subgraph="harness",
            fixture_bindings=(),
            fixture_buffers=(buffer_data("input_a", "float32", [1.0, 2.0, 3.0, 4.0]),),
            golden_bindings=(),
            golden_buffers=(buffer_data("output_a", "float32", [1.0, 2.0, 3.0, 4.0], epsilon=1e-5),),
            fixture_notes="Synthetic harness fixture for loader tests",
            golden_notes="Synthetic golden for harness compare helpers",
        ),
        cumsum_case("D_cumsum_single_pass", [3, 0, -2, 5, 1], "isolated cumsum_single_pass shader testing"),
        cumsum_case(
            "D_cumsum_single_pass_near_block",
            near_block_values,
            "near-block cumsum_single_pass shader testing",
        ),
        cumsum_case(
            "D_cumsum_single_pass_exact_block",
            exact_block_values,
            "exact-block cumsum_single_pass shader testing",
        ),
        cumsum_multi_block_case(
            "D_cumsum_multi_block",
            multi_block_values,
            "multi-block cumsum shader testing",
        ),
        cumsum_multi_block_case(
            "D_cumsum_multi_block_two_level",
            two_level_values,
            "two-level multi-block cumsum shader testing",
            two_level=True,
        ),
        sum_case("D_sum", [1, 0, 2, 3], "isolated sum shader testing"),
        sum_case("D_sum_multi_block", [1] * (SUM_BLOCK_SIZE + 1), "multi-block sum shader testing"),
        where_case("D_where", [0, 1, 0, 1, 1], 0, "isolated where shader testing"),
        where_case("D_where_no_true", [0, 0, 0, 0], -1, "where shader no-true-mask testing"),
        where_case("D_where_first_last", [1, 0, 0, 0, 1], -1, "where shader first-last-mask testing"),
        where_case("D_where_block_boundary", where_boundary_mask, -1, "where shader block-boundary testing"),
    )


def validate_buffer(buffer: BufferData) -> None:
    if buffer.dtype not in DTYPE_FORMATS:
        raise ValueError(f"{buffer.name}: unsupported dtype {buffer.dtype}")
    if len(buffer.shape) == 0:
        raise ValueError(f"{buffer.name}: shape must not be empty")
    expected_values = element_count(buffer.shape)
    if expected_values != len(buffer.values):
        raise ValueError(f"{buffer.name}: shape expects {expected_values} values, got {len(buffer.values)}")


def manifest_for(case: FixtureCase, buffers: tuple[BufferData, ...], bindings: tuple[str, ...], notes: str) -> dict:
    buffer_names = {buffer.name for buffer in buffers}
    missing_bindings = [binding for binding in bindings if binding not in buffer_names]
    if missing_bindings:
        raise ValueError(f"{case.stage_name}: bindings missing buffers: {missing_bindings}")

    manifest = {
        "ref_baseline_tag": REF_BASELINE_TAG,
        "stage_name": case.stage_name,
        "subgraph": case.subgraph,
        "bindings": list(bindings),
        "buffers": {buffer.name: buffer.file_name for buffer in buffers},
        "shapes": {buffer.name: list(buffer.shape) for buffer in buffers},
        "dtypes": {buffer.name: buffer.dtype for buffer in buffers},
        "cmake_preset": DEFAULT_CMAKE_PRESET,
        "emulate_int64": 0,
        "emulate_f32_atomic": 0,
        "vkbd_source": "synthetic",
    }

    epsilons = {buffer.name: buffer.epsilon for buffer in buffers if buffer.epsilon is not None}
    if epsilons:
        manifest["epsilon"] = epsilons
    manifest["notes"] = notes
    return manifest


def write_binary(path: Path, buffer: BufferData) -> None:
    validate_buffer(buffer)
    pack_format = "<" + DTYPE_FORMATS[buffer.dtype]
    path.write_bytes(b"".join(struct.pack(pack_format, value) for value in buffer.values))


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def write_case(root: Path, case: FixtureCase) -> None:
    fixture_dir = root / "fixtures" / case.stage_name
    golden_dir = root / "golden_masters" / case.stage_name
    fixture_dir.mkdir(parents=True, exist_ok=True)
    golden_dir.mkdir(parents=True, exist_ok=True)

    for buffer in case.fixture_buffers:
        write_binary(fixture_dir / buffer.file_name, buffer)
    write_manifest(
        fixture_dir / "manifest.json",
        manifest_for(case, case.fixture_buffers, case.fixture_bindings, case.fixture_notes),
    )

    for buffer in case.golden_buffers:
        write_binary(golden_dir / buffer.file_name, buffer)
    write_manifest(
        golden_dir / "manifest.json",
        manifest_for(case, case.golden_buffers, case.golden_bindings, case.golden_notes),
    )


def clean_generated_dirs(root: Path) -> None:
    for subdir in GENERATED_SUBDIRS:
        path = root / subdir
        if path.exists():
            shutil.rmtree(path)


def generate(root: Path) -> None:
    clean_generated_dirs(root)
    for case in fixture_cases():
        write_case(root, case)


def generated_files(root: Path) -> set[Path]:
    files: set[Path] = set()
    for subdir in GENERATED_SUBDIRS:
        base = root / subdir
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                files.add(path.relative_to(root))
    return files


def compare_generated(actual_root: Path, expected_root: Path) -> list[str]:
    actual_files = generated_files(actual_root)
    expected_files = generated_files(expected_root)

    problems: list[str] = []
    for rel_path in sorted(expected_files - actual_files):
        problems.append(f"missing: {rel_path.as_posix()}")
    for rel_path in sorted(actual_files - expected_files):
        problems.append(f"extra: {rel_path.as_posix()}")
    for rel_path in sorted(actual_files & expected_files):
        if not filecmp.cmp(actual_root / rel_path, expected_root / rel_path, shallow=False):
            problems.append(f"drifted: {rel_path.as_posix()}")
    return problems


def check(root: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="nlrc_vksplat_fixtures_") as temp:
        expected_root = Path(temp) / "test_data"
        generate(expected_root)
        problems = compare_generated(root, expected_root)

    if not problems:
        print("[OK] Fixture data is up to date")
        return 0

    print("Fixture data drift detected.", file=sys.stderr)
    print("Regenerate with: python test_data/generate_fixtures.py --write", file=sys.stderr)
    for problem in problems[:40]:
        print(f"  {problem}", file=sys.stderr)
    if len(problems) > 40:
        print(f"  ... {len(problems) - 40} more", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Regenerate committed fixture and golden files")
    mode.add_argument("--check", action="store_true", help="Check committed files against generated output")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent, help="test_data root directory")
    args = parser.parse_args()

    root = args.root.resolve()
    if args.write:
        generate(root)
        print(f"[OK] Wrote fixture data under {root}")
        return 0
    if args.check:
        return check(root)
    raise AssertionError("unreachable")


if __name__ == "__main__":
    sys.exit(main())
