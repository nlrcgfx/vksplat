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

On Windows with MSVC, run CMake from a Visual Studio Developer PowerShell / x64 Native Tools prompt, or use the wrapper below. Plain PowerShell usually lacks the MSVC `INCLUDE`, `LIB`, and `LIBPATH` environment.

```powershell
cd next/nlrc_vksplat
powershell -ExecutionPolicy Bypass -File scripts/build-windows.ps1 -Preset windows-debug -RunTests
```

```bash
cd next/nlrc_vksplat
cmake --preset windows-debug
cmake --build --preset windows-debug
ctest --preset windows-debug
```

CMake profile options are limited to build-profile toggles: `NLRC_VKSPLAT_EMULATE_INT64` and `NLRC_VKSPLAT_EMULATE_F32_ATOMIC`. CMake propagates these to C++ compile definitions and generated shader config (`shader_config.json` / `config_generated.*`); OHOS presets enable emulated int64 by default. Numeric shader constants such as tile sizes, subgroup sizes, and tensor tables live in `src/nlrc_vksplat_config.hpp`, not CMake options. See `.cursor/rules/shader-build-embed.mdc`.

**VS Code / Cursor:** open the repo root, install the recommended extensions when prompted, then run **CMake: Configure** (or `cmake --preset windows-debug`) so `build/compile_commands.json` is available for clangd. Use a Developer PowerShell terminal when building with MSVC.

**C++ lint:** builds do not run lint. Install pre-commit and run the full-tree hooks before committing:

```powershell
pipx install pre-commit
pre-commit install
pre-commit run --all-files
```

The hooks run `clang-format` and `clang-tidy` over `next/nlrc_vksplat/`, plus repository hygiene checks for trailing whitespace, final newlines, merge markers, oversized files, JSON, and TOML. Configure `windows-debug` first so `build/windows-debug/compile_commands.json` exists, or set `NLRC_VKSPLAT_TIDY_BUILD_DIR` to another configured build directory. Shader builds do not require `clang-format` or `clang-tidy`; generated `*_spirv.hpp` files remain build artifacts. Style rules: `.cursor/rules/cpp-coding-style.mdc`.

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

- [Reference README](ref/README.md) - full VkSplat install, training, and development guide
- [Rewrite testing report](next/nlrc_vksplat/docs/testing-report.md) - current test coverage, gaps, and hardening roadmap
- [Shader pipeline](ref/docs/shader-pipeline.md) - GPU compute stages and buffer layout
- [OHOS porting report](ref/docs/ohos-maleoon910-porting-report.md) - OpenHarmony / Maleoon notes

## Policy

- **`ref/`**: regression baseline and golden buffer dumps; avoid feature work here
- **`next/nlrc_vksplat/`**: all new architecture and implementation
