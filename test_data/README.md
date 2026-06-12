# test_data

Shared regression fixtures and golden masters for the rewrite (`next/nlrc_vksplat/`).
Synthetic fixture values are defined in `generate_fixtures.py`; the checked-in `.bin`
and `manifest.json` files are generated outputs kept in git so CTest can run without
an extra setup step.

For the broader test coverage matrix, known gaps, and hardening roadmap, see the
[rewrite testing report](../next/nlrc_vksplat/docs/testing-report.md).

## Layout

- `generate_fixtures.py`: source of truth for synthetic fixture values, expected values, manifests, binary payloads, and nested folder layout
- `fixtures/<group>/<case>/`: inputs, initial mutable output buffers, and `manifest.json`
- `golden_masters/<group>/<case>/`: expected outputs and `manifest.json` (mirrors `fixtures/` paths)

Groups: `harness`, `cumsum`, `sum`, `where`, `radix_sort`, `projection_forward`, `generate_keys`, `compute_tile_ranges`. Each case directory holds one globally unique `stage_name` in its manifest (for example `cumsum_single_pass` under `fixtures/cumsum/single_pass/`).

All `.bin` files are raw little-endian buffers interpreted through the stage manifest
`dtype` and `shape`.

## Testing Terms

- `fixture`: buffers loaded before dispatch. This includes read-only inputs and initial contents for writable output buffers.
- `actual`: buffer contents read back from the GPU or harness after the test action runs.
- `expected`: buffers under `golden_masters/` used for exact equality or epsilon comparison.
- `bindings`: descriptor binding order used by the shader and the test. Keep this order in sync with the dispatch contract.
- `vkbd_source`: `synthetic` for generator-authored cases; ref-derived cases should record the dump path, run identifier, or source note.

## Workflow

1. Inspect this catalog and the current tests under `next/nlrc_vksplat/tests/`.
2. Edit `test_data/generate_fixtures.py`.
3. Regenerate files:

```powershell
python test_data\generate_fixtures.py --write
```

4. Check for drift:

```powershell
python test_data\generate_fixtures.py --check
```

5. Run the test suite:

```powershell
cd next\nlrc_vksplat
ctest --preset windows-debug --output-on-failure
```

CMake also registers `nlrc_vksplat_fixture_generation_check`, which runs the generator
in `--check` mode with the configured Python interpreter. If `python` is not on `PATH`,
use the interpreter from `build/<preset>/CMakeCache.txt` or rerun CMake with
`-DPython_EXECUTABLE=<path>`.

## Generated Data Assumptions

Block-size mirrors, boundary input-size formulas, and the fixture catalog summary
live in the [rewrite testing report](../next/nlrc_vksplat/docs/testing-report.md#generated-data-assumptions).
This README keeps per-stage buffer values and shader targets in the catalog below.

Current radix sort fixture assumptions mirror `nlrc_vksplat_config.hpp`:

- `SORTING_KEY_BITS = 32`
- `RADIX_SORT_RADIX = 256`
- `RADIX_BITS_PER_PASS = 8`
- `RADIX_WORKGROUP_SIZE = 512`
- `RADIX_PARTITION_DIVISION = 8`
- `RADIX_PARTITION_SIZE = 4096`

Current projection fixture assumptions mirror `nlrc_vksplat_config.hpp`:

- `SH_REORDER_SIZE = 32`
- `TILE_WIDTH = TILE_HEIGHT = 16`
- `rect_tile_space` uses either one `int64` word or two emulated `int32` words, depending on manifest profile

Current `generate_keys` fixture assumptions mirror `nlrc_vksplat_config.hpp`:

- `SORTING_KEY_BITS = 32`
- `TILE_WIDTH = TILE_HEIGHT = 16`
- `grid_width = grid_height = 2`, so `depth_bits = 23`
- `rect_tile_space` uses either one `int64` word or two emulated `int32` words, depending on manifest profile

Current `compute_tile_ranges` fixture assumptions mirror `nlrc_vksplat_config.hpp`:

- `SORTING_KEY_BITS = 32`
- `TILE_WIDTH = TILE_HEIGHT = 16`
- `grid_width = 3`, `grid_height = 2`, so `num_tiles = 6` and `depth_bits = 23`
- `tile_ranges` contains `num_tiles + 1` start offsets and uses the terminal sentinel at index `num_tiles`

## When Block Sizes Change

Treat any change to `VKSPLAT_*_BLOCK_SIZE`, `VKSPLAT_RADIX_*`, or
`VKSPLAT_SORTING_KEY_BITS` as a fixture catalog change:

1. Update the mirrored constants in `test_data/generate_fixtures.py`.
2. Regenerate generated files:

```powershell
python test_data\generate_fixtures.py --write
```

3. Check for drift:

```powershell
python test_data\generate_fixtures.py --check
```

4. Re-check generated binary sizes, especially `cumsum_multi_block_two_level`.
5. Revisit `.pre-commit-config.yaml` large-file exceptions if the two-level cumsum fixture grows.
6. Run the Windows validation workflow with CTest.

## Current Fixture Catalog

| Stage | Test target | Fixture values | Expected values | What it validates |
|-------|-------------|----------------|-----------------|-------------------|
| `harness_smoke` | Host fixture loader and compare helpers | `input_a = [1, 2, 3, 4]` as `float32` | `output_a = [1, 2, 3, 4]` as `float32` | Manifest parsing, typed `.bin` loading, and golden compare plumbing; no shader behavior. |
| `cumsum_single_pass` | `cumsum_single_pass` GPU utility shader | `[3, 0, -2, 5, 1]` | `[3, 3, 1, 6, 7]` | Small inclusive prefix sum over signed `int32`. |
| `cumsum_single_pass_near_block` | `cumsum_single_pass` GPU utility shader | `B - 1` values, currently 511, generated by `(index % 7) - 3` | Inclusive prefix sum | Single-pass behavior just below `VKSPLAT_CUMSUM_BLOCK_SIZE`. |
| `cumsum_single_pass_exact_block` | `cumsum_single_pass` GPU utility shader | `B` ones, currently 512 | `[1..B]` | Single-pass behavior at `VKSPLAT_CUMSUM_BLOCK_SIZE`. |
| `cumsum_multi_block` | `cumsum_block_scan` + `cumsum_scan_block_sums` + `cumsum_add_block_offsets` | `B + 1` signed cyclic values, currently 513, generated by `(index % 11) - 5` | Inclusive prefix sum | One-level multi-block cumsum path with signed values. |
| `cumsum_multi_block_two_level` | two-level multi-phase cumsum with `block_sums2` | `B * B + 1` signed cyclic values, currently 262145, generated by `(index % 5) - 2` | Inclusive prefix sum | Two-level `_cumsum_blockSums2` path above `VKSPLAT_CUMSUM_BLOCK_SIZE * VKSPLAT_CUMSUM_BLOCK_SIZE`. |
| `sum` | `sum` GPU utility shader | `[1, 0, 2, 3]` | `[6]` | Small integer reduction. |
| `sum_multi_block` | `sum` GPU utility shader | `S + 1` ones, currently 513 | `[S + 1]` | Multi-workgroup reduction path using atomic accumulation. |
| `where` | `where` GPU utility shader | `mask = [0, 1, 0, 1, 1]` | `out_indices = [1, 3, 4]` | Basic compaction using an inclusive prefix-count buffer. |
| `where_no_true` | `where` GPU utility shader | all-zero mask, sentinel output `[-1]` | `[-1]` | No-write path leaves output unchanged. |
| `where_first_last` | `where` GPU utility shader | true values at first and last positions | `[0, 4]` | Boundary index handling for `gid == 0` and final element. |
| `where_block_boundary` | `where` GPU utility shader | true values at `W - 1` and `W`, currently 255 and 256 | `[W - 1, W]` | Dispatch behavior across `VKSPLAT_WHERE_BLOCK_SIZE`. |
| `radix_sort_minimum_one` | `radix_sort/upsweep` -> `spine` -> `downsweep` GPU pipeline | one `uint32` key `[7]`, one index `[0]` | stable sorted key/index pair | Minimum non-empty sort dispatch. |
| `radix_sort_single_partition` | full radix sort pipeline | 12 mixed `uint32` keys across low and high bytes | CPU stable-sort by key | One partition, four 8-bit passes, ping-pong buffers. |
| `radix_sort_partition_boundary` | full radix sort pipeline | `P` generated keys, currently 4096 | CPU stable-sort by key | Exact `VKSPLAT_RADIX_PARTITION_SIZE` boundary. |
| `radix_sort_multi_partition` | full radix sort pipeline | `P + 1` generated keys, currently 4097 | CPU stable-sort by key | `num_parts > 1` upsweep/downsweep dispatch. |
| `radix_sort_duplicates` | full radix sort pipeline | duplicate-key sequence `[17, 4, 17, 9, 4, 17, 9, 4, 0, 17, 0, 9]` | CPU stable-sort by key | Duplicate-key stability and histogram correctness. |
| `radix_sort_sorted` | full radix sort pipeline | 64 already-sorted generated keys | unchanged stable sorted keys/indices | Regression guard for sorted input. |
| `radix_sort_reverse` | full radix sort pipeline | 64 reverse-sorted generated keys | CPU stable-sort by key | Regression guard for reverse-sorted input. |
| `projection_forward_native_int64` | `projection_forward` GPU shader | `N=1`; `xyz_ws=[0,0,4]`; zero SH coefficients padded to `12 * 32` `float4` entries; `rotations=[1,0,0,0]`; `scales_opacs=[0.2,0.2,0.2,0.5]`; outputs zero-initialized; uniforms use a 32x32 pinhole camera, 2x2 tile grid, identity world-view transform, `active_sh=0` | none; invariant-only oracle | Native-`int64` `rect_tile_space` layout plus finite/bounded projection outputs for one visible centered splat. |
| `projection_forward_emulated_int64` | `projection_forward` GPU shader | Same as `projection_forward_native_int64`, except `rect_tile_space` is two `int32` words | none; invariant-only oracle | Emulated-`int64` `rect_tile_space` layout plus finite/bounded projection outputs for one visible centered splat. |
| `generate_keys_native_int64` | `generate_keys` GPU shader | `N=4`; 2x2 tile grid; `xy_vs=[(8,8),(0,0),(24,8),(8,24)]`; `depths=[1,2,3,7]`; each `inv_cov_vs_opacity=[1,0,1,0.5]`; rects `[(0,0)-(1,1), empty, (1,0)-(2,1), (0,1)-(1,2)]`; `tiles_touched=[1,0,1,1]`; `index_buffer_offset=[1,1,2,3]`; outputs zero-initialized | `unsorted_keys=[4194304,14680064,24117248]`; `unsorted_gauss_idx=[0,2,3]` | Native-`int64` key packing, inclusive cumsum offset consumption, zero-touch splat skip, and valid splat-index emission. |
| `generate_keys_emulated_int64` | `generate_keys` GPU shader | Same as `generate_keys_native_int64`, except `rect_tile_space` is two `int32` words per splat | Same as `generate_keys_native_int64` | Emulated-`int64` key packing, inclusive cumsum offset consumption, zero-touch splat skip, and valid splat-index emission. |
| `compute_tile_ranges` | `compute_tile_ranges` GPU shader | `3x2` tile grid; sorted tile IDs `[1,1,3,4]`; sorted keys `[12582912,13981013,31457280,40894464]`; `tile_ranges` initialized to seven `-1` values | `tile_ranges=[0,0,2,2,3,4,4]` | Per-tile start offset construction, duplicate tile entries, leading empty tile, interior gap, trailing sentinel, and `active_sh` as `num_indices`. |

Fixture `stage_name` values are globally unique (`cumsum_single_pass`, `radix_sort_single_partition`);
manifest `subgraph` records the porting family (`utility`, `radix_sort`, `projection`, `generate_keys`, `compute_tile_ranges`, `harness`), while
on-disk folders group by kernel or pipeline (`cumsum/`, `sum/`, `where/`, `radix_sort/`, `projection_forward/`, `generate_keys/`, `compute_tile_ranges/`, `harness/`). Utility
shader fixtures validate isolated dispatch behavior before larger ref-parity fixtures
are added. The radix sort fixtures validate the isolated Phase 2 sort pipeline. Their
manifests list the working buffers, while per-pass descriptor order is asserted in
`test_radix_sort.cpp` from the binding contract in `ref/docs/shader-pipeline.md`.
Projection manifests list descriptor bindings `0..10` in the exact order from
`shader-pipeline.md`: `xyz_ws`, `sh_coeffs`, `rotations`, `scales_opacs`,
`tiles_touched`, `rect_tile_space`, `radii`, `xy_vs`, `depths`,
`inv_cov_vs_opacity`, `rgb`.
`generate_keys` manifests list descriptor bindings `0..6` in the exact order from
`shader-pipeline.md`: `xy_vs`, `inv_cov_vs_opacity`, `depths`,
`rect_tile_space`, `index_buffer_offset`, `unsorted_keys`,
`unsorted_gauss_idx`. They also include unbound `tiles_touched` source data so
tests can assert `index_buffer_offset` is the inclusive cumsum consumed by the
shader.
`compute_tile_ranges` manifests list descriptor bindings `0..1` in the exact
order from `shader-pipeline.md`: `sorted_keys`, `tile_ranges`.

## Manifest Schema

Required fields:

- `ref_baseline_tag`
- `stage_name`
- `subgraph`
- `bindings`
- `buffers`
- `shapes`
- `dtypes`
- `cmake_preset`
- `emulate_int64`
- `emulate_f32_atomic`
- `notes`

Optional fields:

- `vkbd_source`
- `epsilon`
- `profile_agnostic`

Generator-owned fields for synthetic fixtures:

- `buffers`
- `shapes`
- `dtypes`
- `vkbd_source`
- `epsilon`
- `notes`

The C++ harness loads manifests into typed `BufferSpec` entries with `file`, `shape`,
`dtype`, and `epsilon`. Supported dtypes are `float32`, `uint32`, `int32`, and `int64`.
Shapes must be non-empty and all dimensions must be positive. Goldens may include
per-buffer `epsilon`; default epsilon is `1e-5` for `float32`.

Default build profile for new synthetic fixtures is `windows-debug` with
`emulate_int64: 0` and `emulate_f32_atomic: 0`. Use `profile_agnostic: true`
only for fixtures whose buffers are valid across native and emulated int64
profiles. Profile-specific fixtures, such as `projection_forward_*`, keep
`profile_agnostic: false` and are selected by tests according to the active
build profile.

## Adding Synthetic Fixtures

- Edit `generate_fixtures.py`; do not hand-edit generated `.bin` or manifest files.
- Keep buffers minimal and deterministic.
- Regenerate with `--write`, then verify with `--check`.
- Keep descriptor-order `bindings` in sync with shader and test code.
- State whether the oracle is synthetic invariant, ref parity, or both.

Generated minimal `.bin` files are committed. `cumsum_multi_block_two_level`
is the smallest fixture that exercises the second cumsum block-sums level, so its
generated payloads are slightly above the normal large-file hook threshold with
the current `B = 512` cumsum block size. If `B` grows, revisit whether the
generated payload should remain committed, move to generated-at-test-time data,
or use a dedicated small shader profile. Do not commit full training-run dumps.

## Creating Curated Goldens From Ref

Use this path only for reference-derived payloads that cannot be expressed directly in
`generate_fixtures.py`.

1. Build ref with `VKSPLAT_ENABLE_BUFFER_DUMPS=ON` and run a minimal dump.
2. Inspect with `ref/vksplat/scripts/vkbd_tool.py`.
3. Convert curated buffers to raw `.bin` with `next/nlrc_vksplat/scripts/vkbd_to_bins.py`.
4. Add the curated fixture to the generator or document why it is maintained separately.

Example:

```powershell
python next\nlrc_vksplat\scripts\vkbd_to_bins.py --vkbd outputs\run\buffer.vkbd --out test_data\fixtures\sum\basic\input.bin --dtype int32 --shape 4
```

## GPU Test Policy

Host tests are the reliable baseline. GPU tests run when Vulkan compute is available
under `NLRC_VKSPLAT_GPU_TESTS=AUTO`. Dedicated GPU validation can configure with
`NLRC_VKSPLAT_GPU_TESTS=REQUIRE`; local or constrained environments can use `OFF`.
