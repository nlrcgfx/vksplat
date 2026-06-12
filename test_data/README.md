# test_data

Shared regression fixtures and golden masters for the rewrite (`next/nlrc_vksplat/`).

## Layout

- `fixtures/<subgraph>_<stage>/`: input buffers and `manifest.json`
- `golden_masters/<subgraph>_<stage>/`: expected outputs and `manifest.json`

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

## Creating Goldens From Ref

1. Build ref with `VKSPLAT_ENABLE_BUFFER_DUMPS=ON` and run a minimal dump.
2. Inspect with `ref/vksplat/scripts/vkbd_tool.py`.
3. Convert curated buffers to raw `.bin` with `next/nlrc_vksplat/scripts/vkbd_to_bins.py`.

Example:

```powershell
python next\nlrc_vksplat\scripts\vkbd_to_bins.py --vkbd outputs\run\buffer.vkbd --out test_data\fixtures\D_sum\input.bin --dtype int32 --shape 4
```

Do not commit full training-run dumps; keep fixtures minimal.
