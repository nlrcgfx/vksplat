# Testing Report

This document is the testing reference for the rewrite in `next/nlrc_vksplat/`.
It records what the current tests validate, what is intentionally outside the
current test surface, and where future hardening should focus.

## At a glance

| Item | Current state |
|------|---------------|
| Test binary | `nlrc_vksplat_tests` (Catch2 + CTest) |
| Host tests | `[host]` cases; reliable baseline on normal dev machines |
| GPU tests | `[gpu]` cases; controlled by `NLRC_VKSPLAT_GPU_TESTS` (`AUTO`, `REQUIRE`, `OFF`) |
| Covered dispatches | Smoke compute wrapper; Subgraph D utilities: cumsum, sum, where; isolated 32-bit radix sort |
| Fixture source | [`test_data/generate_fixtures.py`](../../../test_data/generate_fixtures.py) |
| Per-fixture values | [`test_data/README.md`](../../../test_data/README.md) fixture catalog |

## How to use this document

| Section | Read when |
|---------|-----------|
| [Test execution workflow](#test-execution-workflow) | Running or debugging CTest |
| [Coverage matrix](#coverage-matrix) | Checking what is already tested |
| [Fixture and golden data coverage](#fixture-and-golden-data-coverage) | Adding or changing fixtures |
| [GPU policy and reliability](#gpu-policy-and-reliability) | Configuring optional GPU validation |
| [Not covered](#not-covered) | Planning next work |
| [Hardening backlog](#hardening-backlog) | Optional future improvements |
| [Gap detection rules](#gap-detection-rules) | Required checks when changing tests or fixtures |

## Related documentation

- [`test_data/README.md`](../../../test_data/README.md): fixture catalog, exact buffer values, regeneration workflow
- [`.cursor/rules/rewrite-testing.mdc`](../../../.cursor/rules/rewrite-testing.mdc): oracle types, Catch2 tags, porting workflow
- [`.cursor/rules/rewrite-buffer-contract.mdc`](../../../.cursor/rules/rewrite-buffer-contract.mdc): descriptor binding order for new shader tests

## Purpose and scope

The current suite is a host-first regression harness for the rewrite. It focuses
on deterministic fixture data, shader build/profile consistency, Vulkan compute
wrapper behavior, isolated utility shader dispatches, and isolated radix sort.
It does not claim
end-to-end training, rendering, image quality, or full reference parity coverage.

Use this report when adding tests, reviewing gaps, or deciding whether a behavior
is already protected by an automated check.

## Test execution workflow

**Prerequisites:** configured C++ compiler, Python, Vulkan SDK, `slangc`, and `glslc`.

Paths below use PowerShell from the repo root.

| Goal | Command |
|------|---------|
| Full Windows build and tests | `powershell -ExecutionPolicy Bypass -File next\nlrc_vksplat\scripts\build-windows.ps1 -Preset windows-debug -RunTests` |
| Configure, build, and CTest | `cmake --preset windows-debug`, `cmake --build --preset windows-debug`, `ctest --preset windows-debug --output-on-failure` (from `next/nlrc_vksplat`) |
| Fixture drift check | `python test_data\generate_fixtures.py --check` |

Primary Windows workflow:

```powershell
cd next\nlrc_vksplat
powershell -ExecutionPolicy Bypass -File scripts\build-windows.ps1 -Preset windows-debug -RunTests
```

CMake also registers `nlrc_vksplat_fixture_generation_check`, so fixture drift is
validated by CTest when `BUILD_TESTING` is enabled. If `python` is not on `PATH`,
use `scripts/build-windows.ps1 -PythonExe <path>` or configure CMake with
`-DPython_EXECUTABLE=<path>`.

## Coverage matrix

| Area | Tag | What is tested | Test entrypoints | Covered behavior |
|------|-----|----------------|------------------|------------------|
| Test harness | `[host]` | Catch2 executable and CTest discovery | `test_main.cpp`, `tests/CMakeLists.txt` | Test binary links, Catch2 test cases are discovered, generator drift is registered as a separate CTest. |
| Fixture generator | `[host]` | Synthetic fixture reproducibility | `nlrc_vksplat_fixture_generation_check`, `test_data/generate_fixtures.py --check` | Generated fixture/golden manifests and `.bin` payloads match checked-in files byte-for-byte. |
| Fixture catalog | `[host]` | Whole catalog validity | `test_fixture_catalog.cpp` | Every fixture/golden manifest loads, stage names match directories, build profile matches, bindings reference buffers, files exist, and file sizes match dtype/shape. |
| Manifest parser | `[host]` | Manifest fields and rejection paths | `test_fixture_loader.cpp`, `fixture_manifest.cpp` | Supported dtype parsing, dtype sizes, invalid dtype rejection, malformed JSON rejection, empty or invalid shape rejection. |
| Fixture loader | `[host]` | Typed raw-buffer loading | `test_fixture_loader.cpp`, `fixture_loader.cpp` | Typed float loading, missing buffer names, missing files, dtype mismatch, size mismatch, empty shape guards, little-endian `int32` payloads. |
| Golden comparison | `[host]` | Float result comparison helpers | `test_golden_compare.cpp`, `golden_compare.cpp` | Epsilon pass/fail behavior, size mismatch diagnostics, NaN and infinity rejection. |
| Build profile | `[host]` | Fixture and shader config profile checks | `test_manifest_profile.cpp`, `test_shader_config_profile.cpp` | `harness_smoke` fixture profile matches the build; generated shader config matches C++ constants and CMake profile toggles. |
| GPU availability policy | `[host]` | Optional/required GPU behavior | `gpu_available.cpp`, GPU-tagged tests | `AUTO` skips when no Vulkan compute device is available, `REQUIRE` fails when unavailable, `OFF` skips GPU tests. See [GPU policy](#gpu-policy-and-reliability). |
| Compute wrapper | `[gpu]` | Basic Vulkan compute wrapper validation | `test_gpu_smoke.cpp` | Embedded smoke shader dispatch; invalid SPIR-V; invalid push constant size; descriptor count, null binding, and unbound descriptor rejection. |
| Utility shaders | `[gpu]` | Isolated Subgraph D utility dispatches | `test_utility_shaders.cpp`, generated fixture data | `cumsum_single_pass`, multi-phase cumsum, `sum`, and `where` match synthetic goldens; see [fixture catalog](#fixture-catalog) below. |
| Radix sort | `[gpu]` | Isolated 32-bit sort pipeline | `test_radix_sort.cpp`, generated fixture data | Four 8-bit passes dispatch `radix_sort/upsweep`, `radix_sort/spine`, and `radix_sort/downsweep`; output keys match CPU stable-sort goldens, indices remain a permutation, and duplicate-key order is stable. |

## Fixture and golden data coverage

Synthetic fixture values are defined in [`test_data/generate_fixtures.py`](../../../test_data/generate_fixtures.py);
checked-in `.bin` files and manifests are generated outputs. This keeps small binary
fixtures available to CTest while making the source values reviewable and reproducible.

Per-buffer values and shader binding names live in [`test_data/README.md`](../../../test_data/README.md).
All listed fixtures use synthetic invariant oracles unless noted otherwise in the catalog.

### Generated data assumptions

The Python fixture generator mirrors shader sizing constants from
[`next/nlrc_vksplat/src/nlrc_vksplat_config.hpp`](../src/nlrc_vksplat_config.hpp).

#### Block-size mirrors

| Utility | Python constant (`generate_fixtures.py`) | C++/shader constant (`nlrc_vksplat_config.hpp`) | Value |
|---------|------------------------------------------|-------------------------------------------------|-------|
| Cumsum | `CUMSUM_BLOCK_SIZE` | `VKSPLAT_CUMSUM_BLOCK_SIZE` | 512 |
| Sum | `SUM_BLOCK_SIZE` | `VKSPLAT_SUM_BLOCK_SIZE` | 512 |
| Where | `WHERE_BLOCK_SIZE` | `VKSPLAT_WHERE_BLOCK_SIZE` | 256 |

#### Radix sort mirrors

| Python constant (`generate_fixtures.py`) | C++/shader constant (`nlrc_vksplat_config.hpp`) | Value |
|------------------------------------------|-------------------------------------------------|-------|
| `SORTING_KEY_BITS` | `VKSPLAT_SORTING_KEY_BITS` | 32 |
| `RADIX_SORT_RADIX` | `VKSPLAT_RADIX_SORT_RADIX` | 256 |
| `RADIX_BITS_PER_PASS` | `VKSPLAT_RADIX_BITS_PER_PASS` | 8 |
| `RADIX_WORKGROUP_SIZE` | `VKSPLAT_RADIX_WORKGROUP_SIZE` | 512 |
| `RADIX_PARTITION_DIVISION` | `VKSPLAT_RADIX_PARTITION_DIVISION` | 8 |
| `RADIX_PARTITION_SIZE` | `VKSPLAT_RADIX_PARTITION_SIZE` | 4096 |

#### Boundary input sizes

Boundary coverage uses input lengths (or mask indices) derived from the block-size
constants above.

| Utility | Block symbol | Input-size formulas |
|---------|--------------|---------------------|
| Cumsum | `B` = `CUMSUM_BLOCK_SIZE` | `B - 1`, `B`, `B + 1`, `B * B + 1` |
| Sum | `S` = `SUM_BLOCK_SIZE` | `S + 1` |
| Where | `W` = `WHERE_BLOCK_SIZE` | mask length `W + 1`, true at `W - 1` and `W` |
| Radix sort | `P` = `RADIX_PARTITION_SIZE` | `1`, `P`, `P + 1`, duplicate, sorted, and reverse-sorted key lists |

#### Fixture catalog

| Fixture stage | Subgraph | Tag | Input size | Covered behavior |
|---------------|----------|-----|------------|------------------|
| `harness_smoke` | harness | `[host]` | 4 floats | Host manifest, loader, and compare plumbing |
| `D_cumsum_single_pass` | D | `[gpu]` | 5 | Small signed inclusive prefix sum (`cumsum_single_pass`) |
| `D_cumsum_single_pass_near_block` | D | `[gpu]` | `B - 1` | Single-pass cumsum just below `VKSPLAT_CUMSUM_BLOCK_SIZE` |
| `D_cumsum_single_pass_exact_block` | D | `[gpu]` | `B` | Single-pass cumsum at `VKSPLAT_CUMSUM_BLOCK_SIZE` |
| `D_cumsum_multi_block` | D | `[gpu]` | `B + 1` | One-level `block_scan` -> `scan_block_sums` -> `add_block_offsets` cumsum |
| `D_cumsum_multi_block_two_level` | D | `[gpu]` | `B * B + 1` | Two-level cumsum path using `_cumsum_blockSums2` |
| `D_sum` | D | `[gpu]` | 4 | Small integer reduction |
| `D_sum_multi_block` | D | `[gpu]` | `S + 1` | Multi-workgroup sum with atomic accumulation |
| `D_where` | D | `[gpu]` | 5 | Basic mask compaction |
| `D_where_no_true` | D | `[gpu]` | 4 | No-write mask behavior preserves sentinel output |
| `D_where_first_last` | D | `[gpu]` | 5 | First-element and last-element output indices |
| `D_where_block_boundary` | D | `[gpu]` | `W + 1` | Workgroup boundary behavior around `VKSPLAT_WHERE_BLOCK_SIZE` |
| `radix_sort_minimum_one` | radix sort | `[gpu]` | 1 | Minimum non-empty sort dispatch |
| `radix_sort_single_partition` | radix sort | `[gpu]` | 12 | Single-partition mixed-key sorting |
| `radix_sort_partition_boundary` | radix sort | `[gpu]` | `P` | Exact `VKSPLAT_RADIX_PARTITION_SIZE` boundary |
| `radix_sort_multi_partition` | radix sort | `[gpu]` | `P + 1` | Multi-partition upsweep/downsweep dispatch |
| `radix_sort_duplicates` | radix sort | `[gpu]` | 12 | Duplicate-key stable ordering |
| `radix_sort_sorted` | radix sort | `[gpu]` | 64 | Already-sorted input regression guard |
| `radix_sort_reverse` | radix sort | `[gpu]` | 64 | Reverse-sorted input regression guard |

Current fixture data is intentionally small, except for
`D_cumsum_multi_block_two_level`, which uses the `B * B + 1` input size (`B` =
`CUMSUM_BLOCK_SIZE`); this is the minimum that crosses the second cumsum
block-sums level.
Large training dumps are avoided in git; curated reference-derived buffers should
be minimal and should either be encoded in the generator or documented with
source metadata.

### When block sizes change

Treat any change to `VKSPLAT_*_BLOCK_SIZE`, `VKSPLAT_RADIX_*`, or
`VKSPLAT_SORTING_KEY_BITS` as a fixture catalog change. Follow the regeneration
checklist in [`test_data/README.md#when-block-sizes-change`](../../../test_data/README.md#when-block-sizes-change).

Additional review items for this report:

- Re-check generated binary sizes, especially `D_cumsum_multi_block_two_level`.
- Revisit `.pre-commit-config.yaml` large-file exceptions if the two-level fixture grows.
- Update this document if boundary formulas or covered fixture stages change.
- For radix changes, re-check pass count, partition histogram size, and whether 64-bit keys are now in scope.

## GPU policy and reliability

Host tests are the reliable baseline. GPU tests are controlled by
`NLRC_VKSPLAT_GPU_TESTS`:

| Policy | Behavior |
|--------|----------|
| `AUTO` | Run GPU tests when a Vulkan compute device is available; otherwise skip. |
| `REQUIRE` | Fail if no Vulkan compute device is available. Use this for dedicated GPU validation. |
| `OFF` | Skip GPU tests even when a device is available. Use this for constrained environments. |

GPU tests validate correctness for small deterministic utility shader cases, not
cross-device numerical stability, performance, or full rendering/training behavior.
Radix sort GPU tests are isolated 32-bit integer checks; they do not wire sorted
keys into Subgraph A or G.

## Not covered

Status meanings: **Future plan** = planned work; **Blocked** = external dependency;
**Avoided** = intentional non-goal.

### Future plan

| Area | Notes |
|------|-------|
| Full training pipeline parity with `ref/` | Suite targets infrastructure and isolated utility shaders, not full model training. Add curated reference parity fixtures after stable buffer dump contracts exist. |
| Renderer/image-quality regression tests | No rewrite renderer output oracle or image comparison harness. Define small scene fixtures, output format, and tolerated metrics first. |
| Radix sort integration with Subgraph A/G | Isolated radix sort is covered, but `generate_keys`, `compute_tile_ranges`, rasterization, and Morton sort producers/consumers are not wired through it yet. Wire integration only after the relevant subgraph stages pass isolated tests. |
| Radix sort zero-element no-op | Current fixture schema and `StorageBuffer` require non-empty buffers, and the ref pipeline skips sort when `num_indices == 0`. Add this as a wrapper/subgraph no-op test when Subgraph A/G wiring exists. |
| Radix sort 64-bit key path | Current rewrite config uses `VKSPLAT_SORTING_KEY_BITS = 32`. Add 64-bit fixtures and shader-profile validation only if the rewrite enables 64-bit sorting keys. |
| Radix sort large spine-loop path | Current fixtures cover `num_parts = 1` and `num_parts = 2`, not `num_parts > VKSPLAT_RADIX_WORKGROUP_SIZE`. Add a generated-at-test-time or slow fixture if that path becomes important. |
| Remaining rewrite shader stages | Smoke, Subgraph D utilities, and isolated radix sort are covered. Add one synthetic invariant fixture per new stage before ref-parity fixtures. |
| Memory lifetime, stress, and fault injection | Wrapper tests cover argument validation only. Add opt-in stress tests once resource limits and failure modes are defined. |
| Reference-derived binary conversion tool coverage | `vkbd_to_bins.py` is not covered by automated tests. Add synthetic `.vkbd` samples or parser unit tests before parity ingestion. |

### Blocked

| Area | Notes |
|------|-------|
| OHOS runtime execution | OHOS presets are cross-compile oriented; local CTest is disabled by default. Needs device/emulator instructions and a CI lane. |
| Cross-GPU numerical stability | Needs multiple Vulkan devices/drivers and a tolerance policy beyond exact integer fixtures. |
| Radix sort subgroup-size matrix | Current validation uses whatever subgroup behavior the available Vulkan device exposes. Requires hardware/profile matrix support before claiming portability across subgroup sizes. |

### Avoided

| Area | Notes |
|------|-------|
| Large real-world buffer dumps | Expensive, hard to review, unsuitable for git. Keep under `outputs/` or external storage; commit only minimal curated payloads. |
| Performance/regression benchmarking | Correctness tests stay fast and deterministic. Keep timing in dedicated benchmark tooling with hardware metadata. |

## Hardening backlog

Optional improvements; not required for current test changes.

- Add labels or presets that separate host-only, GPU-optional, GPU-required, and slow tests.
- Expand manifest profile checks from `harness_smoke` to every generated fixture if build-profile variants are introduced.
- Add generator unit tests for drift diagnostics, missing files, extra files, and malformed fixture specs.
- If future cumsum block sizes make `B * B + 1` too large to commit, move the two-level cumsum case to generated-at-test-time data or introduce a dedicated small test shader profile.
- Add a generator or CTest guard that verifies Python radix constants mirror `nlrc_vksplat_config.hpp`.
- Add reference-parity fixtures only after the buffer source, baseline tag, and expected tolerance are documented.
- Add shader-stage coverage as new rewrite kernels become callable from tests.
- Add CI documentation describing which jobs run host-only tests and which jobs require GPU hardware.

## Gap detection rules

Required checks when changing tests, fixtures, or documented coverage.

- New public test helper behavior should have at least one host test.
- New generated fixture data must come from `test_data/generate_fixtures.py` or document why it is curated separately.
- Any change to `VKSPLAT_*_BLOCK_SIZE`, `VKSPLAT_RADIX_*`, or `VKSPLAT_SORTING_KEY_BITS` must be reviewed as a fixture catalog change and followed by fixture regeneration plus drift checking.
- Add a generator or CTest guard that parses `next/nlrc_vksplat/src/nlrc_vksplat_config.hpp` and fails if Python generator constants drift from C++ shader-size defines.
- New shader dispatch contracts should include descriptor order, push constants, fixture inputs, and expected outputs.
- New floating-point comparisons must state epsilon and whether NaN/inf are valid or rejected.
- New reference-derived tests must record the reference baseline tag and source of the dump.
- A behavior should not be marked covered in this document until it is asserted by CTest or by a documented dedicated validation workflow.
