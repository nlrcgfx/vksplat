# test_data

Shared regression fixtures and golden masters for the rewrite (`next/nlrc_vksplat/`).

## Layout

- `fixtures/<subgraph>_<stage>/` — input buffers + `manifest.json`
- `golden_masters/<subgraph>_<stage>/` — expected outputs + `manifest.json`

## Manifest schema

Required fields: `ref_baseline_tag`, `stage_name`, `subgraph`, `bindings`, `buffers`, `shapes`, `dtypes`, `cmake_preset`, `emulate_int64`, `emulate_f32_atomic`, `notes`.

Goldens also include `vkbd_source` and per-buffer `epsilon` (default `1e-5` for `float32`).

Default build profile for new fixtures: `windows-debug` with `emulate_int64: 0` and `emulate_f32_atomic: 0`.

## Creating goldens from ref

1. Build ref with `VKSPLAT_ENABLE_BUFFER_DUMPS=ON` and run a minimal dump.
2. Inspect with `ref/vksplat/scripts/vkbd_tool.py`.
3. Convert curated buffers to raw `.bin` with `next/nlrc_vksplat/scripts/vkbd_to_bins.py` (wrapper).

Do not commit full training-run dumps — keep fixtures minimal.
