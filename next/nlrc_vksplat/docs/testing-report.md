# Testing Report

This document is the testing reference for the rewrite in `next/nlrc_vksplat/`.
It records what the current tests validate, what is intentionally outside the
current test surface, and where future hardening should focus.

## Purpose And Scope

The current suite is a host-first regression harness for the rewrite. It focuses
on deterministic fixture data, shader build/profile consistency, Vulkan compute
wrapper behavior, and isolated utility shader dispatches. It does not claim
end-to-end training, rendering, image quality, or full reference parity coverage.

Use this report when adding tests, reviewing gaps, or deciding whether a behavior
is already protected by an automated check.

## Test Execution Workflow

Primary Windows workflow:

```powershell
cd next\nlrc_vksplat
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1 -Preset windows-debug -RunTests
```

Direct CMake workflow from a configured compiler environment:

```powershell
cd next\nlrc_vksplat
cmake --preset windows-debug
cmake --build --preset windows-debug
ctest --preset windows-debug --output-on-failure
```

Fixture drift check:

```powershell
python test_data\generate_fixtures.py --check
```

CMake also registers `nlrc_vksplat_fixture_generation_check`, so fixture drift is
validated by CTest when `BUILD_TESTING` is enabled. If `python` is not on `PATH`,
use `scripts/build-windows.ps1 -PythonExe <path>` or configure CMake with
`-DPython_EXECUTABLE=<path>`.

## Coverage Matrix

| Area | What is tested | Test entrypoints | Covered behavior |
|------|----------------|------------------|------------------|
| Test harness | Catch2 executable and CTest discovery | `test_main.cpp`, `tests/CMakeLists.txt` | Test binary links, Catch2 test cases are discovered, generator drift is registered as a separate CTest. |
| Fixture generator | Synthetic fixture reproducibility | `nlrc_vksplat_fixture_generation_check`, `test_data/generate_fixtures.py --check` | Generated fixture/golden manifests and `.bin` payloads match checked-in files byte-for-byte. |
| Fixture catalog | Whole catalog validity | `test_fixture_catalog.cpp` | Every fixture/golden manifest loads, stage names match directories, build profile matches, bindings reference buffers, files exist, and file sizes match dtype/shape. |
| Manifest parser | Manifest fields and rejection paths | `test_fixture_loader.cpp`, `fixture_manifest.cpp` | Supported dtype parsing, dtype sizes, invalid dtype rejection, malformed JSON rejection, empty or invalid shape rejection. |
| Fixture loader | Typed raw-buffer loading | `test_fixture_loader.cpp`, `fixture_loader.cpp` | Typed float loading, missing buffer names, missing files, dtype mismatch, size mismatch, empty shape guards, little-endian `int32` payloads. |
| Golden comparison | Float result comparison helpers | `test_golden_compare.cpp`, `golden_compare.cpp` | Epsilon pass/fail behavior, size mismatch diagnostics, NaN and infinity rejection. |
| Build profile | Fixture and shader config profile checks | `test_manifest_profile.cpp`, `test_shader_config_profile.cpp` | `harness_smoke` fixture profile matches the build; generated shader config matches C++ constants and CMake profile toggles. |
| GPU availability policy | Optional/required GPU behavior | `gpu_available.cpp`, GPU-tagged tests | `AUTO` skips when no Vulkan compute device is available, `REQUIRE` fails when unavailable, `OFF` skips GPU tests. |
| Compute wrapper | Basic Vulkan compute wrapper validation | `test_gpu_smoke.cpp` | Embedded smoke shader dispatch; invalid SPIR-V; invalid push constant size; descriptor count, null binding, and unbound descriptor rejection. |
| Utility shaders | Isolated GPU utility dispatches | `test_utility_shaders.cpp`, generated fixture data | `cumsum_single_pass`, `sum`, and `where` results match synthetic goldens, including cumsum near/exact block size, sum multi-block atomic accumulation, where no-write, first/last, and block-boundary cases. |

## Fixture And Golden Data Coverage

Synthetic fixture values are defined in `test_data/generate_fixtures.py`; checked-in
`.bin` files and manifests are generated outputs. This keeps small binary fixtures
available to CTest while making the source values reviewable and reproducible.

| Fixture stage | Covered behavior | Oracle type |
|---------------|------------------|-------------|
| `harness_smoke` | Host manifest, loader, and compare plumbing | Synthetic invariant |
| `D_cumsum_single_pass` | Small signed inclusive prefix sum | Synthetic invariant |
| `D_cumsum_single_pass_near_block` | Single-pass cumsum just below `VKSPLAT_CUMSUM_BLOCK_SIZE` | Synthetic invariant |
| `D_cumsum_single_pass_exact_block` | Single-pass cumsum exactly at `VKSPLAT_CUMSUM_BLOCK_SIZE` | Synthetic invariant |
| `D_sum` | Small integer reduction | Synthetic invariant |
| `D_sum_multi_block` | Multi-workgroup sum with atomic accumulation | Synthetic invariant |
| `D_where` | Basic mask compaction | Synthetic invariant |
| `D_where_no_true` | No-write mask behavior preserves sentinel output | Synthetic invariant |
| `D_where_first_last` | First-element and last-element output indices | Synthetic invariant |
| `D_where_block_boundary` | Workgroup boundary behavior around `VKSPLAT_WHERE_BLOCK_SIZE` | Synthetic invariant |

Current fixture data is intentionally small. Large training dumps are avoided in
git; curated reference-derived buffers should be minimal and should either be
encoded in the generator or documented with source metadata.

## GPU Policy And Reliability

Host tests are the reliable baseline and should pass on normal developer machines
with a configured compiler, Python, Vulkan SDK, `slangc`, and `glslc`. GPU tests
are controlled by `NLRC_VKSPLAT_GPU_TESTS`:

| Policy | Behavior |
|--------|----------|
| `AUTO` | Run GPU tests when a Vulkan compute device is available; otherwise skip. |
| `REQUIRE` | Fail if no Vulkan compute device is available. Use this for dedicated GPU validation. |
| `OFF` | Skip GPU tests even when a device is available. Use this for constrained environments. |

GPU tests validate correctness for small deterministic utility shader cases, not
cross-device numerical stability, performance, or full rendering/training behavior.

## Not Covered

| Area not covered | Status | Reason | Future plan |
|------------------|--------|--------|-------------|
| Full training pipeline parity with `ref/` | Future plan | Current rewrite tests target infrastructure and isolated utility shaders, not full model training. | Add curated reference parity fixtures after stable buffer dump contracts exist for representative training stages. |
| Renderer/image-quality regression tests | Future plan | No rewrite renderer output oracle or image comparison harness is present. | Define small scene fixtures, image output format, and tolerated image metrics before adding render regression tests. |
| Large real-world buffer dumps | Avoided | Large dumps are expensive, hard to review, and unsuitable for git. | Keep large dumps under `outputs/` or external storage; commit only minimal curated payloads with checksums and source metadata. |
| Multi-phase cumsum pipeline | Future plan | Current tests cover `cumsum_single_pass`; block scan, block-sum scan, and offset phases are not dispatched together. | Add generated multi-block cumsum fixtures and a multi-dispatch harness when those phases are wired as a public utility path. |
| Radix sort shader stages | Future plan | Current rewrite tests do not dispatch radix sort shaders. | Add key/value fixtures covering empty-like minimum sizes, single block, multi-block, duplicate keys, and sorted/reverse inputs after dispatch APIs are available. |
| Remaining rewrite shader stages | Future plan | The suite covers only smoke, cumsum, sum, and where utility shaders. | Add one synthetic invariant fixture per new shader stage before adding ref-parity fixtures. |
| OHOS runtime execution | Blocked | OHOS presets are cross-compile oriented; local CTest execution is disabled by default for OHOS. | Add device/emulator execution instructions and a CI lane before claiming runtime coverage. |
| Performance/regression benchmarking | Avoided | Correctness tests are intentionally fast and deterministic; timing assertions would be flaky on developer machines. | Keep performance in dedicated benchmark tooling with explicit hardware/environment metadata. |
| Memory lifetime, stress, and fault injection beyond wrapper checks | Future plan | Current wrapper tests cover argument validation, not repeated allocation stress, device loss, or allocation failures. | Add targeted stress tests behind an opt-in CTest label once resource limits and expected failure modes are defined. |
| Cross-GPU numerical stability | Blocked | Requires multiple Vulkan devices/drivers and a tolerance policy beyond current exact integer fixtures. | Add hardware matrix reporting and float-heavy fixtures when rewrite kernels produce float outputs. |
| Reference-derived binary conversion tool coverage | Future plan | `vkbd_to_bins.py` is used for curated ref data but is not covered by automated tests. | Add small synthetic `.vkbd` samples or pure parser unit tests before relying on it for parity fixture ingestion. |

## Hardening Backlog

- Add labels or presets that separate host-only, GPU-optional, GPU-required, and slow tests.
- Expand manifest profile checks from `harness_smoke` to every generated fixture if build-profile variants are introduced.
- Add generator unit tests for drift diagnostics, missing files, extra files, and malformed fixture specs.
- Add reference-parity fixtures only after the buffer source, baseline tag, and expected tolerance are documented.
- Add shader-stage coverage as new rewrite kernels become callable from tests.
- Add CI documentation describing which jobs run host-only tests and which jobs require GPU hardware.

## Gap Detection Rules

- New public test helper behavior should have at least one host test.
- New generated fixture data must come from `test_data/generate_fixtures.py` or document why it is curated separately.
- New shader dispatch contracts should include descriptor order, push constants, fixture inputs, and expected outputs.
- New floating-point comparisons must state epsilon and whether NaN/inf are valid or rejected.
- New reference-derived tests must record the reference baseline tag and source of the dump.
- A behavior should not be marked covered in this document until it is asserted by CTest or by a documented dedicated validation workflow.
