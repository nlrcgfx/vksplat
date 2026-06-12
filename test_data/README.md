# test_data

Shared regression fixtures and golden masters for the rewrite (`next/nlrc_vksplat/`). This file is the canonical catalog for small fixtures; do not add per-fixture README files unless a future ref-generated case is too large to summarize here.

## Layout

- `fixtures/<subgraph>_<stage>/`: inputs, initial mutable output buffers, and `manifest.json`
- `golden_masters/<subgraph>_<stage>/`: expected outputs and `manifest.json`

All `.bin` files are raw little-endian buffers interpreted through the stage manifest `dtype` and `shape`.

## Testing Terms

- `fixture`: buffers loaded before dispatch. This includes read-only inputs and initial contents for writable output buffers.
- `actual`: buffer contents read back from the GPU or harness after the test action runs.
- `expected`: buffers under `golden_masters/` used for exact equality or epsilon comparison.
- `bindings`: descriptor binding order used by the shader and the test. Keep this order in sync with the dispatch contract.
- `vkbd_source`: `synthetic` for hand-authored cases; future ref-generated cases should record the dump path, run identifier, or source note.

## Current Fixture Catalog

| Stage | Test target | Fixture values | Expected values | What it validates |
|-------|-------------|----------------|-----------------|-------------------|
| `harness_smoke` | Host fixture loader and compare helpers | `input_a = [1, 2, 3, 4]` as `float32` | `output_a = [1, 2, 3, 4]` as `float32` | Manifest parsing, typed `.bin` loading, and golden compare plumbing; no shader behavior. |
| `D_cumsum_single_pass` | `cumsum_single_pass` GPU utility shader | `input = [3, 0, -2, 5, 1]`, initial `output = [0, 0, 0, 0, 0]`, `block_sums = [0]` | `output = [3, 3, 1, 6, 7]` | Inclusive prefix sum over a small signed `int32` vector. |
| `D_sum` | `sum` GPU utility shader | `input = [1, 0, 2, 3]`, initial `output = [0]` | `output = [6]` | Integer reduction across the input vector. |
| `D_where` | `where` GPU utility shader | `mask = [0, 1, 0, 1, 1]`, `mask_cumsum = [0, 1, 1, 2, 3]`, initial `out_indices = [0, 0, 0]` | `out_indices = [1, 3, 4]` | Compaction of true mask indices using the prefix-count buffer. |

The utility shader fixtures are synthetic Subgraph D cases. They validate isolated dispatch behavior before larger ref-parity fixtures are added.

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

The C++ harness loads manifests into typed `BufferSpec` entries with `file`, `shape`, `dtype`, and `epsilon`. Supported dtypes are `float32`, `uint32`, `int32`, and `int64`.

Goldens may also include `vkbd_source` and per-buffer `epsilon`; default epsilon is `1e-5` for `float32`.

Default build profile for new fixtures is `windows-debug` with `emulate_int64: 0` and `emulate_f32_atomic: 0`.

## Adding New Fixtures

- Keep `.bin` files minimal and typed in `manifest.json`.
- Document exact buffer values in this file when they are small enough to read directly.
- For large buffers, document shape, dtype, generation source, checksum or small preview, and the exact meaning of each golden output.
- Keep descriptor-order `bindings` in sync with the shader and test code.
- State whether the oracle is synthetic invariant, ref parity, or both.

## Creating Goldens From Ref

1. Build ref with `VKSPLAT_ENABLE_BUFFER_DUMPS=ON` and run a minimal dump.
2. Inspect with `ref/vksplat/scripts/vkbd_tool.py`.
3. Convert curated buffers to raw `.bin` with `next/nlrc_vksplat/scripts/vkbd_to_bins.py`.

Example:

```powershell
python next\nlrc_vksplat\scripts\vkbd_to_bins.py --vkbd outputs\run\buffer.vkbd --out test_data\fixtures\D_sum\input.bin --dtype int32 --shape 4
```

Do not commit full training-run dumps; keep fixtures minimal.
