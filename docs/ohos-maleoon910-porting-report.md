# OHOS Maleoon 910 Porting Issues Report

## Executive summary

This report covers a native OHOS port of the VkSplat compute core. The Python
training workflow, viewer, and end-user application packaging are treated as
secondary integration work.

The Maleoon 910 target is feasible, but not with the repository's current
default shader/device profile. The highest-risk blockers are:

- The Vulkan device path unconditionally enables subgroup-size-control support,
  while the target has no subgroup control capability.
- Several generated kernels use 1024-thread workgroups, while the target maximum
  workgroup size is 512.
- The default shader profile assumes subgroup size 32, while the target has a
  fixed subgroup size of 64.
- The target has no shader float32 atomic add and no shader int64 support.

Recommended first porting profile:

- Add a Maleoon 910 profile, for example `VKSPLAT_MALEOON910_PROFILE`, that
  expands into:
  - `VKSPLAT_SUBGROUP_SIZE=64`
  - `VKSPLAT_USE_EMULATED_INT64=1`
  - `VKSPLAT_USE_EMULATED_F32_ATOMIC=1`
  - all compute workgroup constants capped at `<=512`
  - no required subgroup-size-control extension or required subgroup-size
    pipeline metadata
- Bring up the native core with default densification first. Treat MCMC and
  Morton reorder as second-stage validation targets because they contain the
  densest use of random sampling, scans, atomics, and sort/reorder logic.

## Target profile and current repository state

### Target constraints

| Capability | Maleoon 910 constraint | Porting implication |
|---|---:|---|
| Max compute workgroup size | 512 | All `[numthreads]` / `local_size_x` values must be `<=512`. |
| Subgroup size | Fixed 64 | Shaders must be compiled and audited for 64-lane subgroups. |
| Subgroup size control | Not available | Runtime must not require `VK_EXT_subgroup_size_control` or required subgroup size metadata. |
| Float32 atomics | Not available | Float atomic add must be emulated or replaced. |
| Shader int64 | Not available | All shader-side int64/uint64 use must be removed or emulated. |

### Existing useful groundwork

- `vksplat/CMakePresets.json` already has `ohos-debug` and `ohos-release`
  presets. The OHOS base preset uses the OHOS native CMake toolchain, targets
  `arm64-v8a`, uses `c++_shared`, and sets `VKSPLAT_BUILD_PYTHON=OFF`.
- `vksplat/CMakeLists.txt` already exposes `VKSPLAT_EMULATE_INT64` and
  `VKSPLAT_EMULATE_F32_ATOMIC`, passes matching C++ definitions, and forwards
  those values to shader config generation.
- `vksplat/scripts/generate_shader_config.py` generates consistent Slang,
  GLSL, and JSON fragments from `vksplat/src/vksplat_config.h`.
- `vksplat/src/gs_renderer.cpp` validates `shader_config.json` against the
  compiled C++ constants at runtime, which is useful for preventing mixed
  host/shader profiles.

### Current default profile mismatch

The default configuration is not Maleoon-ready:

- `VKSPLAT_SUBGROUP_SIZE` defaults to 32 in `vksplat/src/vksplat_config.h`.
- `VKSPLAT_CUMSUM_BLOCK_SIZE`, `VKSPLAT_SUM_BLOCK_SIZE`, and
  `VKSPLAT_MORTON_APPLY_THREADS` are 1024.
- `VKSPLAT_MORTON_STATS_THREADS` is `VKSPLAT_SUBGROUP_SIZE *
  VKSPLAT_SUBGROUP_SIZE`; this becomes 4096 if subgroup size is changed to 64
  without also capping the thread count.
- `VKSPLAT_USE_EMULATED_INT64` and `VKSPLAT_USE_EMULATED_F32_ATOMIC` default to
  0.
- `VulkanGSPipeline::createDevice()` currently adds
  `VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME` unconditionally.

## Issue matrix

| Axis | Severity | Current evidence | Maleoon impact | Recommended path |
|---|---|---|---|---|
| Vulkan device creation | Blocker | `createDevice()` unconditionally enables `VK_EXT_subgroup_size_control`; compute pipelines may attach required subgroup-size metadata. | Device creation or pipeline creation can fail before any shader runs. | Make subgroup-size control optional and disabled in the Maleoon profile. |
| Workgroup size | Blocker | Cumsum, sum, and Morton apply kernels use 1024-thread constants. | Shaders exceed `maxComputeWorkGroupSize[0]=512`. | Cap all profile constants to 512 and adjust algorithms that assume 1024. |
| Subgroup width | Blocker | Default shader manifest records subgroup size 32. | 32-lane assumptions can break scan, ballot, culling, and tensor backward logic on fixed 64-lane hardware. | Compile a native 64-lane shader bundle and audit subgroup-dependent code. |
| Float32 atomics | Blocker | Raster backward and Morton stats use `InterlockedAddF32` or generated `atomicAdd(float)`. | Unsupported shader capability unless emulated. | Use existing CAS emulation first, then reduce contention for performance. |
| Int64 removal | Blocker | Rect tile bounds can be emulated, but residual `uint64_t` helpers exist in strategy/MCMC code. | Any remaining int64 capability use will fail compilation or pipeline creation. | Turn on int64 emulation and statically audit SPIR-V/source for residual int64. |
| Radix/Morton sorting | High | Radix sort workgroup size is 512, but uses subgroup arithmetic/ballot and subgroup-derived masks. | May compile but still produce incorrect offsets with 64-lane subgroup assumptions if untested. | Validate radix sort independently on device before using full training. |
| Runtime/API packaging | Medium | OHOS CMake builds core without Python. `VkSplatTrainingSession` exists as a C++ layer. | Native integration needs explicit shader asset paths and C/C++ API ownership. | Treat Python as host-only and expose the C++ training/session core to OHOS app code. |
| Timing/query portability | Medium | Device selection requires a compute queue with `timestampValidBits != 0`; runtime creates timestamp query pools. | Some OHOS Vulkan stacks may omit timestamp support on otherwise valid compute queues. | Make timestamps optional for non-benchmark builds. |

## Detailed issues and solution alternatives

### 1. Vulkan device creation requires subgroup-size control

Issue:

`VulkanGSPipeline::createDevice()` currently enables
`VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME` unconditionally and initializes
`VkPhysicalDeviceSubgroupSizeControlFeaturesEXT` with subgroup-size-control
features enabled. `createComputePipeline()` can attach
`VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT` when the device
subgroup size differs from `VKSPLAT_SUBGROUP_SIZE`.

Maleoon impact:

The target has fixed subgroup size 64 and no subgroup control capability.
Requiring the extension is incompatible with the device, even if shaders are
compiled for the native 64-lane subgroup size.

Alternatives:

- Optionalize subgroup-size control:
  - Query extension and feature support before adding it to the device extension
    list.
  - Only attach required subgroup-size metadata when the extension is present
    and the selected profile explicitly needs it.
  - For Maleoon, compile `VKSPLAT_SUBGROUP_SIZE=64` and avoid requesting any
    subgroup override.
- Add a dedicated Maleoon profile:
  - Use a single CMake/profile flag to disable subgroup-size control, set
    subgroup size 64, and enable required emulations.
  - This is less flexible but safer for a known target.
- Keep separate 32-lane and 64-lane shader bundles:
  - Select the matching bundle after device enumeration.
  - This is useful long term if multiple OHOS GPUs are supported, but it adds
    packaging and validation overhead.

Recommended path:

Use a dedicated Maleoon profile first, implemented on top of optional
subgroup-size-control probing. The profile should never request subgroup size
control because the target's native subgroup size already matches the intended
64-lane compile profile.

### 2. 1024-thread kernels exceed max workgroup size

Issue:

The current profile uses 1024-thread kernels in multiple places:

- `VKSPLAT_CUMSUM_BLOCK_SIZE=1024`
- `VKSPLAT_SUM_BLOCK_SIZE=1024`
- `VKSPLAT_MORTON_APPLY_THREADS=1024`
- `VKSPLAT_MORTON_STATS_THREADS=SUBGROUP_SIZE*SUBGROUP_SIZE`

Changing only `VKSPLAT_SUBGROUP_SIZE` from 32 to 64 would make
`VKSPLAT_MORTON_STATS_THREADS` become 4096, worsening the mismatch.

Maleoon impact:

Any generated shader with `local_size_x=1024` or larger cannot be used on a
device limited to 512 invocations per workgroup.

Alternatives:

- Cap constants in the Maleoon profile:
  - Set cumsum, sum, Morton stats, and Morton apply threads to 512 or lower.
  - Recompile shaders and adjust C++ dispatch validation accordingly.
- Retune scan/reduction algorithms:
  - Rewrite prefix sum and reductions around 512-thread blocks and 64-lane
    subgroup scans.
  - This preserves functionality with predictable performance.
- Temporarily disable optional Morton reorder:
  - Use this only for first bring-up if Morton kernels remain problematic.
  - Training should still run, but memory locality and performance may regress.

Recommended path:

Cap constants to 512 in the first profile, then specifically validate cumsum,
sum, Morton stats, and Morton apply kernels. Do not rely on the current
`SUBGROUP_SIZE*SUBGROUP_SIZE` formula for Morton stats under the Maleoon
profile.

### 3. Fixed subgroup size 64 conflicts with default 32-lane assumptions

Issue:

The default generated shader config records subgroup size 32. Several kernels
use `SUBGROUP_SIZE` in indexing, shared-memory layout, scan logic, SH reorder,
subtile culling, and raster backward scheduling.

Maleoon impact:

Running 32-lane assumptions on a 64-lane device without subgroup control can
break correctness. The risk is highest in code using `WaveReadLaneAt`,
`WavePrefixSum`, `WaveActiveSum`, `WaveShuffle`, subgroup ballot, and explicit
lane/subgroup indexing.

Alternatives:

- Compile native 64-lane shaders:
  - Set `VKSPLAT_SUBGROUP_SIZE=64` and regenerate all shader config fragments.
  - Audit kernels with subgroup-specific indexing.
- Prefer non-subgroup raster backward variants at first:
  - Tensor backward config entries include variants with
    `USE_SUBGROUP_OPERATIONS=0`.
  - Scheduler can be constrained until correctness/performance is measured.
- Maintain per-GPU profiles:
  - Keep the default 32-lane profile for desktop GPUs that support it.
  - Add a fixed 64-lane profile for Maleoon.

Recommended path:

Compile a native 64-lane Maleoon profile and initially prefer shader variants
that minimize subgroup-specific raster backward behavior. Re-enable and
benchmark subgroup-heavy variants only after correctness tests pass.

### 4. Float32 atomic add is unavailable

Issue:

Float atomic add is used in raster backward gradient accumulation and Morton
statistics. The Slang code already contains a `USE_EMULATED_F32_ATOMIC` path
that implements float add with integer compare-exchange, and CMake can set
`VKSPLAT_EMULATE_F32_ATOMIC=ON`.

Maleoon impact:

Without emulation or algorithm replacement, shaders requiring float atomic add
cannot be compiled or executed on the target.

Alternatives:

- Use existing CAS-based emulation:
  - Fastest implementation path.
  - Expected to be slower under contention, especially in raster backward.
- Reduce before atomics:
  - Increase group-level reduction before atomic writes where variants support
    it.
  - This reduces contention but may increase shared-memory use.
- Replace atomics with sort/reduce accumulation:
  - Emit gradient contributions, sort/group by splat id, then reduce.
  - More deterministic and avoids float atomics, but adds memory traffic and
    kernels.

Recommended path:

Use CAS emulation for bring-up. Measure raster backward and Morton stats
contention after correctness is established, then decide whether a grouped
reduction or sort/reduce path is necessary.

### 5. Shader int64 is unavailable

Issue:

The repo already emulates `rect_tile_space` by storing two 32-bit words when
`VKSPLAT_USE_EMULATED_INT64=1`. However, shader sources still contain some
`uint64_t` helpers, especially in strategy/MCMC random sampling and optional
64-bit Morton code paths. The current JSON field `shader_requires_int64` depends
on the emulation and sorting key configuration.

Maleoon impact:

Any residual shader `Int64` capability use is a blocker. The target requires
all 64-bit integer logic to be removed, avoided, or represented with 32-bit
pairs.

Alternatives:

- Use existing `rect_tile_space` emulation and audit residual int64:
  - Set `VKSPLAT_EMULATE_INT64=ON`.
  - Compile and inspect SPIR-V for `Int64` capability.
- Fully replace random-sampling helpers with 32-bit pair arithmetic:
  - Prefer the existing `uint32_t2` helper style where possible.
  - Ensure helper declarations themselves do not force int64 compilation.
- Disable MCMC until verified:
  - Default densification has less residual int64 risk.
  - MCMC can be validated after baseline training is stable.

Recommended path:

Enable int64 emulation and make the static SPIR-V audit mandatory. If any
generated shader still declares `Int64`, first remove/guard unused `uint64_t`
helpers, then defer MCMC-specific paths until default training passes.

### 6. Radix and Morton sorting need target-specific validation

Issue:

The vendored radix sort GLSL uses a 512-thread workgroup, which matches the
Maleoon limit, but it depends heavily on subgroup arithmetic and ballot logic.
Morton sorting uses float atomics for stats and 1024-thread apply/update kernels
in the default profile.

Maleoon impact:

Radix sort may compile while still having hidden correctness issues with a
64-lane subgroup. Morton sorting has both workgroup-size and float-atomic
issues under the default profile.

Alternatives:

- Adapt and validate the current sort:
  - Compile with subgroup size 64 and run a standalone sorting correctness
    test for random keys, duplicate keys, and boundary sizes.
- Replace with a known portable GPU sort:
  - Higher implementation cost, but lower long-term maintenance risk if
    multiple mobile GPUs are targeted.
- Temporarily limit Morton reorder:
  - Keep tile sorting mandatory for rendering/training.
  - Disable or reduce Morton reorder frequency during first bring-up if needed.

Recommended path:

Validate radix sort independently before full pipeline tests. Keep tile sort in
scope for the first port because it is central to rendering. Treat Morton
reorder as optimizable and temporarily deferrable if it blocks bring-up.

### 7. OHOS build and runtime integration

Issue:

The OHOS CMake preset builds the native core with Python disabled. This is the
right direction for OHOS, but the runtime still needs an app-facing integration
path for shader assets, dataset/image inputs, model outputs, and native object
lifetime.

Maleoon impact:

The core can compile without producing a usable OHOS workflow unless native
callers can supply shader paths, input data, and training/render commands.

Alternatives:

- Use `VkSplatTrainingSession` as the first native integration layer:
  - It already wraps trainer state, buffers, uniforms, SPIR-V path setup, and
    training/render entry points.
  - Keep Python bindings host-only.
- Add a small C ABI wrapper later:
  - Useful for ArkTS/NAPI or app boundary integration.
  - Keep it thin and avoid duplicating trainer logic.
- Keep training orchestration on host:
  - Use OHOS only as an execution target for selected kernels or render tests.
  - Good for debugging, not sufficient for standalone device training.

Recommended path:

Use `VkSplatTrainingSession` for native core bring-up. Add an OHOS app/API
wrapper only after device creation, shader loading, forward render, and one
training step are validated.

### 8. Timestamp queries and diagnostics may be too strict

Issue:

Device selection currently requires a compute queue with nonzero
`timestampValidBits`, and the runtime creates a timestamp query pool during
initialization.

Maleoon impact:

If the OHOS Vulkan driver exposes a usable compute queue without timestamp
support, the current selection logic rejects it. Even if timestamps are
available, query behavior should not be a prerequisite for non-benchmark runs.

Alternatives:

- Make timestamps optional:
  - Accept compute queues without timestamps.
  - Disable `PerfTimer` GPU markers when query support is absent.
- Keep strict timestamps for benchmark builds:
  - Useful for desktop and controlled performance work.
  - Too strict for first device bring-up.
- Provide a fallback host timer:
  - Lower precision but enough for smoke tests and coarse regression tracking.

Recommended path:

Make timestamp queries optional in the Maleoon profile and keep strict query
requirements only for benchmark-oriented builds.

## Proposed bring-up roadmap

1. Add the Maleoon profile plumbing.
   - CMake option or preset value for the target profile.
   - Compile definitions for subgroup 64, int64 emulation, f32 atomic
     emulation, and workgroup caps.
   - Regenerated Slang/GLSL/JSON shader config.

2. Fix runtime capability negotiation.
   - Query subgroup-size-control support before enabling the extension.
   - Do not request subgroup-size control on Maleoon.
   - Do not require float atomic extension when f32 atomic emulation is enabled.
   - Do not require shader int64 when the generated shader manifest says int64
     is not required.

3. Enforce workgroup-size compatibility.
   - Cap all configured workgroup sizes to 512.
   - Audit generated shader files or SPIR-V reflection output for local sizes.
   - Fix cumsum/sum/Morton logic if the smaller block size exposes assumptions.

4. Validate shader capabilities statically.
   - Confirm no generated SPIR-V declares `Int64`.
   - Confirm no generated SPIR-V declares float atomic capabilities.
   - Confirm all compute local sizes are `<=512`.
   - Confirm shader config JSON matches the C++ build profile.

5. Bring up minimal runtime on device.
   - Device enumeration.
   - Logical device creation.
   - Shader module and compute pipeline creation.
   - Buffer allocation and host/device transfers.

6. Validate kernels in dependency order.
   - Cumsum and sum.
   - Radix sort.
   - Projection forward.
   - Generate keys and compute tile ranges.
   - Rasterize forward.
   - SSIM loss.
   - Raster backward with emulated atomics.
   - Fused projection backward optimizer.
   - Default densification.
   - MCMC and Morton reorder after the baseline path is stable.

7. Run end-to-end smoke tests.
   - One forward render on a tiny scene.
   - One full training step.
   - A short training run with default densification.
   - Optional MCMC run after int64 and sort/reorder audits pass.

## Validation checklist

### Static shader checks

- Generated `shader_config.json` reports:
  - `subgroup_size=64`
  - `use_emulated_int64=1`
  - `use_emulated_f32_atomic=1`
  - `shader_requires_int64=0`
  - all workgroup-related constants `<=512`
- SPIR-V audit confirms:
  - no `OpCapability Int64`
  - no float atomic capability requirement
  - no local size above 512
- Generated GLSL/Slang intermediates do not contain unguarded `int64_t` or
  `uint64_t` in compiled Maleoon paths.

### Build checks

- `cmake --preset ohos-release` succeeds with the Maleoon profile enabled.
- `cmake --build --preset ohos-release` builds `vksplat_core`.
- Shader config check passes after regeneration.
- Host build still succeeds with the default profile.
- Host build also succeeds with the Maleoon profile where toolchain support is
  available.

### On-device smoke checks

- Device enumeration prints Maleoon 910 as accepted or explicitly soft-accepted
  with emulation enabled.
- Logical device creation succeeds without subgroup-size-control extension.
- All compute pipelines load from the packaged shader bundle.
- A tiny forward render produces finite pixel values.
- One training step completes without validation errors or device loss.
- Short default-strategy training run shows stable memory use and no NaN
  parameter growth.

## Acceptance criteria for the native core port

The OHOS native compute core should be considered ported when:

- The Maleoon profile builds the native core and matching shaders.
- Runtime initialization succeeds on Maleoon 910 without requesting unsupported
  subgroup-size-control, float-atomic, or int64 features.
- Static shader audits pass for int64, float atomics, and local size.
- Forward render and one full training step execute on device.
- The desktop default profile remains usable.

## Open risks

- CAS-based float atomic emulation may be correct but slow under raster backward
  contention.
- 64-lane subgroup behavior may change performance ranking of raster backward
  variants, so the existing scheduler should not be trusted until profiled.
- MCMC has additional random sampling and residual 64-bit helper risk; default
  densification should be the first correctness target.
- If OHOS Vulkan timestamp support is incomplete, performance reporting must
  fall back to host timing until GPU timestamps are optional.

