#!/usr/bin/env python3
"""Generate deterministic synthetic fixture and golden test data."""

from __future__ import annotations

import argparse
import filecmp
import json
import math
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
SH_REORDER_SIZE = 32
TILE_HEIGHT = 16
TILE_WIDTH = 16
SORTING_KEY_BITS = 32
FLOAT32_FRACTION_BITS = 23
RADIX_SORT_RADIX = 256
RADIX_WORKGROUP_SIZE = 512
RADIX_PARTITION_DIVISION = 8
RADIX_PARTITION_SIZE = RADIX_WORKGROUP_SIZE * RADIX_PARTITION_DIVISION
RADIX_BITS_PER_PASS = 8
RADIX_SORT_PASSES = SORTING_KEY_BITS // RADIX_BITS_PER_PASS

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
    emulate_int64: int = 0
    profile_agnostic: bool = True
    uniforms: dict[str, int | float | list[float]] | None = None


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


def sorted_indices_by_key(keys: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(range(len(keys)), key=lambda index: keys[index]))


def float32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def depth_bits_for_grid(grid_width: int, grid_height: int) -> int:
    return min(int(31.99 - math.log2(grid_width * grid_height)), FLOAT32_FRACTION_BITS)


def generate_keys_depth_code(depth: float, depth_bits: int) -> int:
    depth32 = float32(depth)
    transformed = float32(float32(float32(2.0) * depth32 + float32(1.0)) / float32(depth32 + float32(1.0)))
    bits = struct.unpack("<I", struct.pack("<f", transformed))[0]
    return (bits & ((1 << FLOAT32_FRACTION_BITS) - 1)) >> (FLOAT32_FRACTION_BITS - depth_bits)


def pack_rect_tile_space(rects: Iterable[tuple[int, int, int, int]], emulate_int64: int) -> tuple[int, ...]:
    words: list[int] = []
    for min_x, min_y, max_x, max_y in rects:
        min_word = min_x | (min_y << 16)
        max_word = max_x | (max_y << 16)
        if emulate_int64:
            words.extend((min_word, max_word))
        else:
            words.append(min_word | (max_word << 32))
    return tuple(words)


def stage_relpath(stage_name: str, subgraph: str) -> Path:
    if subgraph == "harness":
        if stage_name == "harness_smoke":
            return Path("harness") / "smoke"
        raise ValueError(f"unsupported harness stage: {stage_name}")

    if subgraph == "utility":
        if stage_name.startswith("cumsum_"):
            return Path("cumsum") / stage_name.removeprefix("cumsum_")
        if stage_name == "sum":
            return Path("sum") / "basic"
        if stage_name.startswith("sum_"):
            return Path("sum") / stage_name.removeprefix("sum_")
        if stage_name == "where":
            return Path("where") / "basic"
        if stage_name.startswith("where_"):
            return Path("where") / stage_name.removeprefix("where_")
        raise ValueError(f"unsupported utility stage: {stage_name}")

    if subgraph == "radix_sort":
        if stage_name.startswith("radix_sort_"):
            return Path("radix_sort") / stage_name.removeprefix("radix_sort_")
        raise ValueError(f"unsupported radix sort stage: {stage_name}")

    if subgraph == "projection":
        if stage_name.startswith("projection_forward_"):
            return Path("projection_forward") / stage_name.removeprefix("projection_forward_")
        raise ValueError(f"unsupported projection stage: {stage_name}")

    if subgraph == "generate_keys":
        if stage_name.startswith("generate_keys_"):
            return Path("generate_keys") / stage_name.removeprefix("generate_keys_")
        raise ValueError(f"unsupported generate_keys stage: {stage_name}")

    if subgraph == "compute_tile_ranges":
        if stage_name == "compute_tile_ranges":
            return Path("compute_tile_ranges") / "basic"
        if stage_name.startswith("compute_tile_ranges_"):
            return Path("compute_tile_ranges") / stage_name.removeprefix("compute_tile_ranges_")
        raise ValueError(f"unsupported compute_tile_ranges stage: {stage_name}")

    if subgraph == "rasterize_forward":
        if stage_name.startswith("rasterize_forward_"):
            return Path("rasterize_forward") / stage_name.removeprefix("rasterize_forward_")
        raise ValueError(f"unsupported rasterize_forward stage: {stage_name}")

    raise ValueError(f"unsupported subgraph for stage layout: {subgraph}")


def buffer_data(name: str, dtype: str, values: Iterable[int | float], epsilon: float | None = None) -> BufferData:
    materialized = tuple(values)
    return BufferData(name=name, dtype=dtype, shape=(len(materialized),), values=materialized, epsilon=epsilon)


def cumsum_case(stage_name: str, values: Iterable[int], notes: str) -> FixtureCase:
    input_values = tuple(values)
    expected = prefix_sum(input_values)
    return FixtureCase(
        stage_name=stage_name,
        subgraph="utility",
        fixture_bindings=("input", "output", "block_sums"),
        fixture_buffers=(
            buffer_data("input", "int32", input_values),
            buffer_data("output", "int32", [0] * len(input_values)),
            buffer_data("block_sums", "int32", [0]),
        ),
        golden_bindings=("output",),
        golden_buffers=(buffer_data("output", "int32", expected),),
        fixture_notes=f"Synthetic utility fixture for {notes}",
        golden_notes=f"Synthetic utility golden for {notes}",
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
        subgraph="utility",
        fixture_bindings=tuple(bindings),
        fixture_buffers=tuple(buffers),
        golden_bindings=("output",),
        golden_buffers=(buffer_data("output", "int32", expected),),
        fixture_notes=f"Synthetic utility fixture for {notes}",
        golden_notes=f"Synthetic utility golden for {notes}",
    )


def sum_case(stage_name: str, values: Iterable[int], notes: str) -> FixtureCase:
    input_values = tuple(values)
    return FixtureCase(
        stage_name=stage_name,
        subgraph="utility",
        fixture_bindings=("input", "output"),
        fixture_buffers=(
            buffer_data("input", "int32", input_values),
            buffer_data("output", "int32", [0]),
        ),
        golden_bindings=("output",),
        golden_buffers=(buffer_data("output", "int32", [sum(input_values)]),),
        fixture_notes=f"Synthetic utility fixture for {notes}",
        golden_notes=f"Synthetic utility golden for {notes}",
    )


def where_case(stage_name: str, mask: Iterable[int], initial_fill: int, notes: str) -> FixtureCase:
    mask_values = tuple(mask)
    indices = where_indices(mask_values)
    output_len = max(1, len(indices))
    expected = indices if indices else (initial_fill,)
    return FixtureCase(
        stage_name=stage_name,
        subgraph="utility",
        fixture_bindings=("mask", "mask_cumsum", "out_indices"),
        fixture_buffers=(
            buffer_data("mask", "int32", mask_values),
            buffer_data("mask_cumsum", "int32", mask_cumsum(mask_values)),
            buffer_data("out_indices", "int32", [initial_fill] * output_len),
        ),
        golden_bindings=("out_indices",),
        golden_buffers=(buffer_data("out_indices", "int32", expected),),
        fixture_notes=f"Synthetic utility fixture for {notes}",
        golden_notes=f"Synthetic utility golden for {notes}",
    )


def radix_sort_case(stage_name: str, keys: Iterable[int], notes: str) -> FixtureCase:
    key_values = tuple(keys)
    if not key_values:
        raise ValueError(f"{stage_name}: radix sort fixtures must contain at least one key")
    if any(key < 0 or key > 0xFFFFFFFF for key in key_values):
        raise ValueError(f"{stage_name}: uint32 radix sort key out of range")

    indices = tuple(range(len(key_values)))
    sorted_source_indices = sorted_indices_by_key(key_values)
    sorted_keys = tuple(key_values[index] for index in sorted_source_indices)
    sorted_gauss_idx = tuple(indices[index] for index in sorted_source_indices)
    num_parts = (len(key_values) + RADIX_PARTITION_SIZE - 1) // RADIX_PARTITION_SIZE

    return FixtureCase(
        stage_name=stage_name,
        subgraph="radix_sort",
        fixture_bindings=(
            "sorting_keys_1",
            "sorting_gauss_idx_1",
            "sorting_keys_2",
            "sorting_gauss_idx_2",
            "_sorting_histogram",
            "_sorting_histogram_cumsum",
        ),
        fixture_buffers=(
            buffer_data("sorting_keys_1", "uint32", key_values),
            buffer_data("sorting_gauss_idx_1", "uint32", indices),
            buffer_data("sorting_keys_2", "uint32", [0] * len(key_values)),
            buffer_data("sorting_gauss_idx_2", "uint32", [0] * len(key_values)),
            buffer_data("_sorting_histogram", "uint32", [0] * (RADIX_SORT_PASSES * RADIX_SORT_RADIX)),
            buffer_data("_sorting_histogram_cumsum", "uint32", [0] * (num_parts * RADIX_SORT_RADIX)),
        ),
        golden_bindings=("sorted_keys", "sorted_gauss_idx"),
        golden_buffers=(
            buffer_data("sorted_keys", "uint32", sorted_keys),
            buffer_data("sorted_gauss_idx", "uint32", sorted_gauss_idx),
        ),
        fixture_notes=f"Synthetic radix sort fixture for {notes}",
        golden_notes=f"Synthetic radix sort golden for {notes}",
    )


def projection_forward_case(stage_name: str, emulate_int64: int, visible: bool = True) -> FixtureCase:
    sh_coeffs = [0.0] * (12 * SH_REORDER_SIZE * 4)
    rect_dtype = "int32" if emulate_int64 else "int64"
    rect_shape = (2,) if emulate_int64 else (1,)
    rect_initial = [0, 0] if emulate_int64 else [0]
    profile_name = "emulated int64" if emulate_int64 else "native int64"
    xyz_ws = (0.0, 0.0, 4.0) if visible else (0.0, 0.0, 0.0)
    visibility_note = "one centered visible splat" if visible else "one no-visible splat behind the near plane"

    return FixtureCase(
        stage_name=stage_name,
        subgraph="projection",
        fixture_bindings=(
            "xyz_ws",
            "sh_coeffs",
            "rotations",
            "scales_opacs",
            "tiles_touched",
            "rect_tile_space",
            "radii",
            "xy_vs",
            "depths",
            "inv_cov_vs_opacity",
            "rgb",
        ),
        fixture_buffers=(
            BufferData("xyz_ws", "float32", (1, 3), xyz_ws),
            BufferData("sh_coeffs", "float32", (12 * SH_REORDER_SIZE, 4), tuple(sh_coeffs)),
            BufferData("rotations", "float32", (1, 4), (1.0, 0.0, 0.0, 0.0)),
            BufferData("scales_opacs", "float32", (1, 4), (0.2, 0.2, 0.2, 0.5)),
            BufferData("tiles_touched", "int32", (1,), (0,)),
            BufferData("rect_tile_space", rect_dtype, rect_shape, tuple(rect_initial)),
            BufferData("radii", "int32", (1,), (0,)),
            BufferData("xy_vs", "float32", (1, 2), (0.0, 0.0)),
            BufferData("depths", "float32", (1,), (0.0,)),
            BufferData("inv_cov_vs_opacity", "float32", (1, 4), (0.0, 0.0, 0.0, 0.0)),
            BufferData("rgb", "float32", (1, 3), (0.0, 0.0, 0.0)),
        ),
        golden_bindings=(),
        golden_buffers=(),
        fixture_notes=(
            f"Synthetic projection_forward invariant fixture for {visibility_note} using "
            f"{profile_name} rect_tile_space layout"
        ),
        golden_notes=(
            "Invariant-only projection_forward case; no ref-derived golden outputs are committed for this stage"
        ),
        emulate_int64=emulate_int64,
        profile_agnostic=False,
        uniforms={
            "image_height": 32,
            "image_width": 32,
            "grid_height": 2,
            "grid_width": 2,
            "num_splats": 1,
            "active_sh": 0,
            "step": 0,
            "camera_model": 0,
            "fx": 16.0,
            "fy": 16.0,
            "cx": 16.0,
            "cy": 16.0,
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0],
            "world_view_transform": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
    )


def generate_keys_case(stage_name: str, emulate_int64: int) -> FixtureCase:
    grid_width = 2
    grid_height = 2
    num_splats = 4
    tiles_touched = (1, 0, 1, 1)
    index_buffer_offset = prefix_sum(tiles_touched)
    num_indices = index_buffer_offset[-1]
    depth_bits = depth_bits_for_grid(grid_width, grid_height)

    rects = (
        (0, 0, 1, 1),
        (0, 0, 0, 0),
        (1, 0, 2, 1),
        (0, 1, 1, 2),
    )
    xy_vs = (
        8.0,
        8.0,
        0.0,
        0.0,
        24.0,
        8.0,
        8.0,
        24.0,
    )
    depths = (1.0, 2.0, 3.0, 7.0)
    inv_cov_vs_opacity = (
        1.0,
        0.0,
        1.0,
        0.5,
        1.0,
        0.0,
        1.0,
        0.5,
        1.0,
        0.0,
        1.0,
        0.5,
        1.0,
        0.0,
        1.0,
        0.5,
    )

    expected_keys: list[int] = [0] * num_indices
    expected_indices: list[int] = [0] * num_indices
    for splat_index, touched in enumerate(tiles_touched):
        if touched == 0:
            continue
        start = 0 if splat_index == 0 else index_buffer_offset[splat_index - 1]
        rect = rects[splat_index]
        tile_id = rect[1] * grid_width + rect[0]
        expected_keys[start] = (tile_id << depth_bits) | generate_keys_depth_code(depths[splat_index], depth_bits)
        expected_indices[start] = splat_index

    rect_dtype = "int32" if emulate_int64 else "int64"
    rect_shape = (num_splats * 2,) if emulate_int64 else (num_splats,)
    profile_name = "emulated int64" if emulate_int64 else "native int64"

    return FixtureCase(
        stage_name=stage_name,
        subgraph="generate_keys",
        fixture_bindings=(
            "xy_vs",
            "inv_cov_vs_opacity",
            "depths",
            "rect_tile_space",
            "index_buffer_offset",
            "unsorted_keys",
            "unsorted_gauss_idx",
        ),
        fixture_buffers=(
            BufferData("xy_vs", "float32", (num_splats, 2), xy_vs),
            BufferData("inv_cov_vs_opacity", "float32", (num_splats, 4), inv_cov_vs_opacity),
            BufferData("depths", "float32", (num_splats,), depths),
            BufferData("rect_tile_space", rect_dtype, rect_shape, pack_rect_tile_space(rects, emulate_int64)),
            BufferData("tiles_touched", "int32", (num_splats,), tiles_touched),
            BufferData("index_buffer_offset", "int32", (num_splats,), index_buffer_offset),
            BufferData("unsorted_keys", "uint32", (num_indices,), [0] * num_indices),
            BufferData("unsorted_gauss_idx", "int32", (num_indices,), [0] * num_indices),
        ),
        golden_bindings=("unsorted_keys", "unsorted_gauss_idx"),
        golden_buffers=(
            BufferData("unsorted_keys", "uint32", (num_indices,), tuple(expected_keys)),
            BufferData("unsorted_gauss_idx", "int32", (num_indices,), tuple(expected_indices)),
        ),
        fixture_notes=(
            "Synthetic generate_keys fixture using single-tile rects, inclusive cumsum offsets, "
            f"and {profile_name} rect_tile_space layout"
        ),
        golden_notes=(
            "CPU-generated generate_keys golden for tile/depth sort-key packing and splat index emission"
        ),
        emulate_int64=emulate_int64,
        profile_agnostic=False,
        uniforms={
            "image_height": TILE_HEIGHT * grid_height,
            "image_width": TILE_WIDTH * grid_width,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "num_splats": num_splats,
            "active_sh": 0,
            "step": 0,
            "camera_model": 0,
            "fx": 16.0,
            "fy": 16.0,
            "cx": 16.0,
            "cy": 16.0,
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0],
            "world_view_transform": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
    )


def compute_tile_ranges_case() -> FixtureCase:
    grid_width = 3
    grid_height = 2
    tile_ids = (1, 1, 3, 4)
    depths = (1.0, 2.0, 3.0, 7.0)
    depth_bits = depth_bits_for_grid(grid_width, grid_height)
    sorted_keys = tuple(
        (tile_id << depth_bits) | generate_keys_depth_code(depth, depth_bits)
        for tile_id, depth in zip(tile_ids, depths)
    )
    num_tiles = grid_width * grid_height
    tile_ranges = [0] * (num_tiles + 1)

    prev_tile = -1
    for index in range(len(sorted_keys) + 1):
        curr_tile = num_tiles if index == len(sorted_keys) else sorted_keys[index] >> depth_bits
        for tile_id in range(prev_tile + 1, curr_tile + 1):
            tile_ranges[tile_id] = index
        prev_tile = curr_tile

    return FixtureCase(
        stage_name="compute_tile_ranges",
        subgraph="compute_tile_ranges",
        fixture_bindings=("sorted_keys", "tile_ranges"),
        fixture_buffers=(
            BufferData("sorted_keys", "uint32", (len(sorted_keys),), sorted_keys),
            BufferData("tile_ranges", "int32", (num_tiles + 1,), [-1] * (num_tiles + 1)),
        ),
        golden_bindings=("tile_ranges",),
        golden_buffers=(BufferData("tile_ranges", "int32", (num_tiles + 1,), tuple(tile_ranges)),),
        fixture_notes=(
            "Synthetic compute_tile_ranges fixture with a 3x2 tile grid, duplicate tile entries, "
            "leading empty tile, interior gap, and trailing sentinel range"
        ),
        golden_notes="CPU-generated compute_tile_ranges golden for per-tile start offsets",
        uniforms={
            "image_height": TILE_HEIGHT * grid_height,
            "image_width": TILE_WIDTH * grid_width,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "num_splats": 0,
            "active_sh": len(sorted_keys),
            "step": 0,
            "camera_model": 0,
            "fx": 16.0,
            "fy": 16.0,
            "cx": 16.0,
            "cy": 16.0,
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0],
            "world_view_transform": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
    )


def tile_ranges_from_sorted_keys(sorted_keys: tuple[int, ...], grid_width: int, grid_height: int) -> tuple[int, ...]:
    depth_bits = depth_bits_for_grid(grid_width, grid_height)
    num_tiles = grid_width * grid_height
    tile_ranges = [0] * (num_tiles + 1)

    prev_tile = -1
    for index in range(len(sorted_keys) + 1):
        curr_tile = num_tiles if index == len(sorted_keys) else sorted_keys[index] >> depth_bits
        for tile_id in range(prev_tile + 1, curr_tile + 1):
            tile_ranges[tile_id] = index
        prev_tile = curr_tile

    return tuple(tile_ranges)


def rasterize_forward_case(
    stage_name: str,
    grid_width: int,
    grid_height: int,
    tile_ids: tuple[int, ...],
    depths: tuple[float, ...],
    xy_vs_pairs: tuple[tuple[float, float], ...],
    inv_cov_vs_opacity_values: tuple[tuple[float, float, float, float], ...],
    rgb_values: tuple[tuple[float, float, float], ...],
    notes: str,
) -> FixtureCase:
    if not (
        len(tile_ids)
        == len(depths)
        == len(xy_vs_pairs)
        == len(inv_cov_vs_opacity_values)
        == len(rgb_values)
    ):
        raise ValueError(f"{stage_name}: raster fixture input lengths must match")

    depth_bits = depth_bits_for_grid(grid_width, grid_height)
    sorted_keys = tuple(
        (tile_id << depth_bits) | generate_keys_depth_code(depth, depth_bits)
        for tile_id, depth in zip(tile_ids, depths)
    )
    tile_ranges = tile_ranges_from_sorted_keys(sorted_keys, grid_width, grid_height)
    num_splats = len(tile_ids)
    image_width = TILE_WIDTH * grid_width
    image_height = TILE_HEIGHT * grid_height
    num_pixels = image_width * image_height

    xy_vs = tuple(component for pair in xy_vs_pairs for component in pair)
    inv_cov_vs_opacity = tuple(component for value in inv_cov_vs_opacity_values for component in value)
    rgb = tuple(component for value in rgb_values for component in value)

    return FixtureCase(
        stage_name=stage_name,
        subgraph="rasterize_forward",
        fixture_bindings=(
            "sorted_gauss_idx",
            "tile_ranges",
            "xy_vs",
            "inv_cov_vs_opacity",
            "rgb",
            "pixel_state",
            "n_contributors",
        ),
        fixture_buffers=(
            BufferData("sorted_keys", "uint32", (len(sorted_keys),), sorted_keys),
            BufferData("sorted_gauss_idx", "int32", (num_splats,), tuple(range(num_splats))),
            BufferData("tile_ranges", "int32", (grid_width * grid_height + 1,), tile_ranges),
            BufferData("xy_vs", "float32", (num_splats, 2), xy_vs),
            BufferData("inv_cov_vs_opacity", "float32", (num_splats, 4), inv_cov_vs_opacity),
            BufferData("rgb", "float32", (num_splats, 3), rgb),
            BufferData("pixel_state", "float32", (image_height, image_width, 4), [-1.0] * (num_pixels * 4)),
            BufferData("n_contributors", "int32", (image_height, image_width), [-1] * num_pixels),
        ),
        golden_bindings=(),
        golden_buffers=(),
        fixture_notes=(
            f"Synthetic rasterize_forward invariant fixture for {notes}; sorted_keys are included "
            "as unbound provenance for tile_ranges"
        ),
        golden_notes=(
            "Invariant-only rasterize_forward case; no exact rendered pixel golden is committed for this stage"
        ),
        uniforms={
            "image_height": image_height,
            "image_width": image_width,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "num_splats": num_splats,
            "active_sh": 0,
            "step": 0,
            "camera_model": 0,
            "fx": 16.0,
            "fy": 16.0,
            "cx": 16.0,
            "cy": 16.0,
            "dist_coeffs": [0.0, 0.0, 0.0, 0.0],
            "world_view_transform": [
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        },
    )


def fixture_cases() -> tuple[FixtureCase, ...]:
    near_block_values = tuple((index % 7) - 3 for index in range(CUMSUM_BLOCK_SIZE - 1))
    exact_block_values = tuple(1 for _ in range(CUMSUM_BLOCK_SIZE))
    multi_block_values = tuple((index % 11) - 5 for index in range(CUMSUM_BLOCK_SIZE + 1))
    two_level_values = tuple((index % 5) - 2 for index in range((CUMSUM_BLOCK_SIZE * CUMSUM_BLOCK_SIZE) + 1))
    where_boundary_mask = tuple(1 if index in (WHERE_BLOCK_SIZE - 1, WHERE_BLOCK_SIZE) else 0 for index in range(WHERE_BLOCK_SIZE + 1))
    single_partition_keys = (
        0xFFFFFFFF,
        0x00000000,
        0x01000000,
        0x000000FF,
        0x0000FF00,
        0x00FF0000,
        0x80000000,
        0x7FFFFFFF,
        0x12345678,
        0x12345670,
        0x00000001,
        0xABCDEF01,
    )
    partition_boundary_keys = tuple(((index * 1103515245 + 12345) & 0xFFFFFFFF) for index in range(RADIX_PARTITION_SIZE))
    multi_partition_keys = tuple(
        (((RADIX_PARTITION_SIZE - index) * 2654435761) ^ (index * 97)) & 0xFFFFFFFF
        for index in range(RADIX_PARTITION_SIZE + 1)
    )
    duplicate_keys = (17, 4, 17, 9, 4, 17, 9, 4, 0, 17, 0, 9)
    sorted_keys = tuple(index * 65537 for index in range(64))
    reverse_keys = tuple(reversed(sorted_keys))

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
        cumsum_case("cumsum_single_pass", [3, 0, -2, 5, 1], "isolated cumsum_single_pass shader testing"),
        cumsum_case(
            "cumsum_single_pass_near_block",
            near_block_values,
            "near-block cumsum_single_pass shader testing",
        ),
        cumsum_case(
            "cumsum_single_pass_exact_block",
            exact_block_values,
            "exact-block cumsum_single_pass shader testing",
        ),
        cumsum_multi_block_case(
            "cumsum_multi_block",
            multi_block_values,
            "multi-block cumsum shader testing",
        ),
        cumsum_multi_block_case(
            "cumsum_multi_block_two_level",
            two_level_values,
            "two-level multi-block cumsum shader testing",
            two_level=True,
        ),
        sum_case("sum", [1, 0, 2, 3], "isolated sum shader testing"),
        sum_case("sum_multi_block", [1] * (SUM_BLOCK_SIZE + 1), "multi-block sum shader testing"),
        where_case("where", [0, 1, 0, 1, 1], 0, "isolated where shader testing"),
        where_case("where_no_true", [0, 0, 0, 0], -1, "where shader no-true-mask testing"),
        where_case("where_first_last", [1, 0, 0, 0, 1], -1, "where shader first-last-mask testing"),
        where_case("where_block_boundary", where_boundary_mask, -1, "where shader block-boundary testing"),
        radix_sort_case("radix_sort_minimum_one", [7], "one-element edge dispatch"),
        radix_sort_case("radix_sort_single_partition", single_partition_keys, "single-partition mixed-key sorting"),
        radix_sort_case("radix_sort_partition_boundary", partition_boundary_keys, "exact partition-size sorting"),
        radix_sort_case("radix_sort_multi_partition", multi_partition_keys, "multi-partition sorting"),
        radix_sort_case("radix_sort_duplicates", duplicate_keys, "duplicate-key stable sorting"),
        radix_sort_case("radix_sort_sorted", sorted_keys, "already-sorted input"),
        radix_sort_case("radix_sort_reverse", reverse_keys, "reverse-sorted input"),
        projection_forward_case("projection_forward_native_int64", 0),
        projection_forward_case("projection_forward_emulated_int64", 1),
        projection_forward_case("projection_forward_no_visible_native_int64", 0, visible=False),
        projection_forward_case("projection_forward_no_visible_emulated_int64", 1, visible=False),
        generate_keys_case("generate_keys_native_int64", 0),
        generate_keys_case("generate_keys_emulated_int64", 1),
        compute_tile_ranges_case(),
        rasterize_forward_case(
            "rasterize_forward_single_splat",
            2,
            2,
            (0,),
            (1.0,),
            ((8.0, 8.0),),
            ((0.5, 0.0, 0.5, 0.8),),
            ((1.0, 0.25, 0.125),),
            "one centered splat in the first tile of a 2x2 grid",
        ),
        rasterize_forward_case(
            "rasterize_forward_multi_tile",
            3,
            2,
            (1, 1, 3, 4),
            (1.0, 2.0, 3.0, 7.0),
            ((20.0, 8.0), (28.0, 8.0), (8.0, 24.0), (24.0, 24.0)),
            (
                (0.5, 0.0, 0.5, 0.75),
                (0.5, 0.0, 0.5, 0.7),
                (0.5, 0.0, 0.5, 0.65),
                (0.5, 0.0, 0.5, 0.6),
            ),
            ((1.0, 0.2, 0.1), (0.1, 1.0, 0.2), (0.2, 0.1, 1.0), (1.0, 1.0, 0.2)),
            "duplicate tile entries, empty tile gaps, and non-empty lower-row tiles in a 3x2 grid",
        ),
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
        "emulate_int64": case.emulate_int64,
        "emulate_f32_atomic": 0,
        "profile_agnostic": case.profile_agnostic,
        "vkbd_source": "synthetic",
    }

    if case.uniforms is not None:
        manifest["uniforms"] = case.uniforms

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
    stage_path = stage_relpath(case.stage_name, case.subgraph)
    fixture_dir = root / "fixtures" / stage_path
    golden_dir = root / "golden_masters" / stage_path
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
