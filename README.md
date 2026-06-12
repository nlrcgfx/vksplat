# nlrc-vksplat

Monorepo for VkSplat-based 3D Gaussian Splatting training research. The repository keeps a frozen reference implementation alongside a new rewrite.

## Layout

| Path | Purpose |
|------|---------|
| [`ref/`](ref/) | Reference VkSplat copy (mostly readonly; critical bugfixes only) |
| [`ref/vksplat/`](ref/vksplat/) | CMake/Python project root for the reference build |
| [`ref/docs/`](ref/docs/) | Reference architecture and porting notes |
| [`next/nlrc_vksplat/`](next/nlrc_vksplat/) | New implementation (Phase 2 rewrite) |
| `outputs/` | Training runs and buffer dumps (gitignored) |

Reference baseline tag: **`ref-baseline-2026-06-12`**

## Quick start (rewrite)

Requires **Python 3**, **Vulkan SDK** (`slangc` and `glslc` on `PATH`), and a Ninja-capable C++ toolchain. Shaders compile at build time into embedded SPIR-V headers under `build/<preset>/generated/`.

```bash
cd next/nlrc_vksplat
cmake --preset windows-debug
cmake --build --preset windows-debug
ctest --preset windows-debug
```

CMake profile options (subset of config): `NLRC_VKSPLAT_EMULATE_INT64`, `NLRC_VKSPLAT_EMULATE_F32_ATOMIC`. OHOS presets enable emulated int64 by default. Full numeric constants live in `src/nlrc_vksplat_config.hpp`. See `.cursor/rules/shader-build-embed.mdc`.

**VS Code / Cursor:** open the repo root, install the recommended extensions when prompted, then run **CMake: Configure** (or `cmake --preset windows-debug`) so `build/compile_commands.json` is available for clangd.

**C++ lint:** every build runs blocking `clang-format` + `clang-tidy` via the `nlrc_vksplat_lint` target. Shader generation also lint-checks each `*_spirv.hpp`. Run lint alone with `cmake --build --preset windows-debug --target nlrc_vksplat_lint`. Style rules: `.cursor/rules/cpp-coding-style.mdc`.

## Quick start (reference)

Build the C++ trainer:

```bash
cd ref/vksplat
cmake --preset release
cmake --build --preset release --target vksplat_train
```

Recompile shaders:

```bash
cd ref
python compile_shaders.py
```

Install the Python package (editable):

```bash
cd ref/vksplat
pip install -e . --no-build-isolation
```

## Documentation

- [Reference README](ref/README.md) — full VkSplat install, training, and development guide
- [Shader pipeline](ref/docs/shader-pipeline.md) — GPU compute stages and buffer layout
- [OHOS porting report](ref/docs/ohos-maleoon910-porting-report.md) — OpenHarmony / Maleoon notes

## Policy

- **`ref/`**: regression baseline and golden buffer dumps; avoid feature work here
- **`next/nlrc_vksplat/`**: all new architecture and implementation
