#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "nlrc_vksplat_config.hpp"
#include "span.hpp"

namespace nlrc::vksplat::gpu {

enum class DescriptorAccess : std::uint8_t {
  Read,
  Write,
  ReadWrite,
};

enum class ShaderId : std::uint8_t {
  CumsumSinglePass,
  CumsumBlockScan,
  CumsumScanBlockSums,
  CumsumAddBlockOffsets,
  Sum,
  Where,
  ProjectionForward,
  GenerateKeys,
  ComputeTileRanges,
  RasterizeForward,
  RadixSortUpsweep,
  RadixSortSpine,
  RadixSortDownsweep,
};

struct ElementCountPushConstants final {
  std::uint32_t num_elements{};
};

static_assert(sizeof(ElementCountPushConstants) == 4);

struct RadixSortPushConstants final {
  std::uint32_t pass{};
  std::uint32_t element_count{};
};

static_assert(sizeof(RadixSortPushConstants) == 8);

struct RendererUniforms final {
  std::uint32_t image_height{};
  std::uint32_t image_width{};
  std::uint32_t grid_height{};
  std::uint32_t grid_width{};
  std::uint32_t num_splats{};
  std::uint32_t active_sh{};
  std::uint32_t step{};
  std::uint32_t camera_model{};
  float fx{};
  float fy{};
  float cx{};
  float cy{};
  std::array<float, 4> dist_coeffs{};
  std::array<float, 16> world_view_transform{};
};

static_assert(sizeof(RendererUniforms) == 128);
static_assert(sizeof(RendererUniforms) <= kMaxPushConstantBytes);
static_assert(offsetof(RendererUniforms, fx) == 32);
static_assert(offsetof(RendererUniforms, dist_coeffs) == 48);
static_assert(offsetof(RendererUniforms, world_view_transform) == 64);

struct ShaderBindingDescriptor final {
  const char *name{};
  DescriptorAccess access{DescriptorAccess::Read};
};

struct ShaderDefineDescriptor final {
  const char *name{};
  std::int32_t value{};
};

struct ShaderSourceDescriptor final {
  const char *path{};
  const char *language{};
  Span<const ShaderDefineDescriptor> defines;
};

struct ShaderDispatchDescriptor final {
  std::uint32_t block_size{};
  const char *formula{};
};

struct ShaderInterface final {
  ShaderId id{};
  const char *logical_name{};
  Span<const ShaderBindingDescriptor> bindings;
  std::uint32_t binding_count{};
  const char *push_constant_type{};
  std::size_t push_constant_size{};
  ShaderDispatchDescriptor dispatch{};
  const char *profile_notes{};
  ShaderSourceDescriptor source{};
};

struct FixtureBindingContract final {
  const char *name{};
  Span<const char *const> bindings;
};

// Kernel port checklist:
// 1. Add the logical shader registry entry and binding table here.
// 2. Add a checked *_storage_bindings() helper in shader_binding_resolver.hpp.
// 3. Route fixture/golden manifests through shader_fixture_mapping.cpp or an explicit untracked path.
// 4. Keep test_data/shader_binding_contracts.json and descriptor/generator checks green.

namespace detail {

inline constexpr std::array<ShaderBindingDescriptor, 3> kCumsumBindings = {{
    {"input", DescriptorAccess::Read},
    {"output", DescriptorAccess::ReadWrite},
    {"block_sums", DescriptorAccess::ReadWrite},
}};

inline constexpr std::array<ShaderBindingDescriptor, 2> kSumBindings = {{
    {"input", DescriptorAccess::Read},
    {"output", DescriptorAccess::ReadWrite},
}};

inline constexpr std::array<ShaderBindingDescriptor, 3> kWhereBindings = {{
    {"mask", DescriptorAccess::Read},
    {"mask_cumsum", DescriptorAccess::Read},
    {"out_indices", DescriptorAccess::Write},
}};

inline constexpr std::array<ShaderBindingDescriptor, 11> kProjectionForwardBindings = {{
    {"xyz_ws", DescriptorAccess::Read},
    {"sh_coeffs", DescriptorAccess::Read},
    {"rotations", DescriptorAccess::Read},
    {"scales_opacs", DescriptorAccess::Read},
    {"tiles_touched", DescriptorAccess::Write},
    {"rect_tile_space", DescriptorAccess::Write},
    {"radii", DescriptorAccess::Write},
    {"xy_vs", DescriptorAccess::Write},
    {"depths", DescriptorAccess::Write},
    {"inv_cov_vs_opacity", DescriptorAccess::Write},
    {"rgb", DescriptorAccess::Write},
}};

inline constexpr std::array<ShaderBindingDescriptor, 7> kGenerateKeysBindings = {{
    {"xy_vs", DescriptorAccess::Read},
    {"inv_cov_vs_opacity", DescriptorAccess::Read},
    {"depths", DescriptorAccess::Read},
    {"rect_tile_space", DescriptorAccess::Read},
    {"index_buffer_offset", DescriptorAccess::Read},
    {"unsorted_keys", DescriptorAccess::Write},
    {"unsorted_gauss_idx", DescriptorAccess::Write},
}};

inline constexpr std::array<ShaderBindingDescriptor, 2> kComputeTileRangesBindings = {{
    {"sorted_keys", DescriptorAccess::Read},
    {"tile_ranges", DescriptorAccess::Write},
}};

inline constexpr std::array<ShaderBindingDescriptor, 7> kRasterizeForwardBindings = {{
    {"sorted_gauss_idx", DescriptorAccess::Read},
    {"tile_ranges", DescriptorAccess::Read},
    {"xy_vs", DescriptorAccess::Read},
    {"inv_cov_vs_opacity", DescriptorAccess::Read},
    {"rgb", DescriptorAccess::Read},
    {"pixel_state", DescriptorAccess::Write},
    {"n_contributors", DescriptorAccess::Write},
}};

inline constexpr std::array<ShaderBindingDescriptor, 3> kRadixSortUpsweepBindings = {{
    {"unsorted_keys", DescriptorAccess::Read},
    {"_sorting_histogram", DescriptorAccess::ReadWrite},
    {"_sorting_histogram_cumsum", DescriptorAccess::Write},
}};

inline constexpr std::array<ShaderBindingDescriptor, 2> kRadixSortSpineBindings = {{
    {"_sorting_histogram", DescriptorAccess::ReadWrite},
    {"_sorting_histogram_cumsum", DescriptorAccess::ReadWrite},
}};

inline constexpr std::array<ShaderBindingDescriptor, 6> kRadixSortDownsweepBindings = {{
    {"_sorting_histogram", DescriptorAccess::Read},
    {"_sorting_histogram_cumsum", DescriptorAccess::Read},
    {"unsorted_keys", DescriptorAccess::Read},
    {"unsorted_gauss_idx", DescriptorAccess::Read},
    {"sorted_keys", DescriptorAccess::Write},
    {"sorted_gauss_idx", DescriptorAccess::Write},
}};

inline constexpr std::array<ShaderDefineDescriptor, 0> kNoDefines{};
inline constexpr std::array<ShaderDefineDescriptor, 1> kCumsumSinglePassDefines = {{{"CUMSUM_PHASE", 0}}};
inline constexpr std::array<ShaderDefineDescriptor, 1> kCumsumBlockScanDefines = {{{"CUMSUM_PHASE", 1}}};
inline constexpr std::array<ShaderDefineDescriptor, 1> kCumsumScanBlockSumsDefines = {{{"CUMSUM_PHASE", 2}}};
inline constexpr std::array<ShaderDefineDescriptor, 1> kCumsumAddBlockOffsetsDefines = {{{"CUMSUM_PHASE", 3}}};
inline constexpr std::array<ShaderDefineDescriptor, 1> kProjectionForwardDefines = {{{"EXPORT_MODE", 0}}};
inline constexpr std::array<ShaderDefineDescriptor, 1> kGenerateKeysDefines = {{{"ENTRY", 1}}};
inline constexpr std::array<ShaderDefineDescriptor, 1> kComputeTileRangesDefines = {{{"ENTRY", 2}}};
inline constexpr std::array<ShaderDefineDescriptor, 1> kRasterizeForwardDefines = {{{"EXPORT_MODE", 0}}};

inline constexpr const char *kNoProfileNotes = "";
inline constexpr const char *kRectTileSpaceProfileNotes =
    "rect_tile_space is int64[1] in native-int64 builds and int32[2] in emulate-int64 builds";
inline constexpr const char *kRadixSortProfileNotes =
    "32-bit sorting keys; update descriptors and execute_sort before enabling 64-bit keys";

} // namespace detail

template <ShaderId Id>
struct ShaderBindingTraits;

// clang-format off
#define NLRC_VKSPLAT_SHADER_BINDING_TRAITS(shader_id, binding_array) \
  template <> \
  struct ShaderBindingTraits<ShaderId::shader_id> final { \
    static constexpr std::size_t binding_count = detail::binding_array.size(); \
 \
    [[nodiscard]] static constexpr Span<const ShaderBindingDescriptor> bindings() { \
      return make_span(detail::binding_array); \
    } \
  }

NLRC_VKSPLAT_SHADER_BINDING_TRAITS(CumsumSinglePass, kCumsumBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(CumsumBlockScan, kCumsumBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(CumsumScanBlockSums, kCumsumBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(CumsumAddBlockOffsets, kCumsumBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(Sum, kSumBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(Where, kWhereBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(ProjectionForward, kProjectionForwardBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(GenerateKeys, kGenerateKeysBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(ComputeTileRanges, kComputeTileRangesBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(RasterizeForward, kRasterizeForwardBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(RadixSortUpsweep, kRadixSortUpsweepBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(RadixSortSpine, kRadixSortSpineBindings);
NLRC_VKSPLAT_SHADER_BINDING_TRAITS(RadixSortDownsweep, kRadixSortDownsweepBindings);

#undef NLRC_VKSPLAT_SHADER_BINDING_TRAITS
// clang-format on

template <ShaderId Id>
// NOLINTNEXTLINE(readability-identifier-naming)
inline constexpr std::size_t shader_binding_count_v = ShaderBindingTraits<Id>::binding_count;

template <ShaderId Id, std::size_t Index>
[[nodiscard]] constexpr const char *shader_binding_name() {
  static_assert(Index < shader_binding_count_v<Id>, "Shader binding index is out of range");
  return ShaderBindingTraits<Id>::bindings()[Index].name;
}

[[nodiscard]] constexpr bool storage_binding_name_equal(const char *left, std::string_view right) noexcept {
  if (left == nullptr) {
    return right.empty();
  }

  std::size_t index = 0;
  while (left[index] != '\0') {
    if (index >= right.size() || left[index] != right[index]) {
      return false;
    }
    ++index;
  }
  return index == right.size();
}

inline constexpr std::uint32_t kCumsumDescriptorBindingCount =
    static_cast<std::uint32_t>(detail::kCumsumBindings.size());
inline constexpr std::uint32_t kSumDescriptorBindingCount = static_cast<std::uint32_t>(detail::kSumBindings.size());
inline constexpr std::uint32_t kWhereDescriptorBindingCount = static_cast<std::uint32_t>(detail::kWhereBindings.size());
inline constexpr std::uint32_t kProjectionForwardDescriptorBindingCount =
    static_cast<std::uint32_t>(detail::kProjectionForwardBindings.size());
inline constexpr std::uint32_t kGenerateKeysDescriptorBindingCount =
    static_cast<std::uint32_t>(detail::kGenerateKeysBindings.size());
inline constexpr std::uint32_t kComputeTileRangesDescriptorBindingCount =
    static_cast<std::uint32_t>(detail::kComputeTileRangesBindings.size());
inline constexpr std::uint32_t kRasterizeForwardDescriptorBindingCount =
    static_cast<std::uint32_t>(detail::kRasterizeForwardBindings.size());
inline constexpr std::uint32_t kRadixSortUpsweepDescriptorBindingCount =
    static_cast<std::uint32_t>(detail::kRadixSortUpsweepBindings.size());
inline constexpr std::uint32_t kRadixSortSpineDescriptorBindingCount =
    static_cast<std::uint32_t>(detail::kRadixSortSpineBindings.size());
inline constexpr std::uint32_t kRadixSortDownsweepDescriptorBindingCount =
    static_cast<std::uint32_t>(detail::kRadixSortDownsweepBindings.size());

namespace detail {

// clang-format off
inline constexpr std::array<ShaderInterface, 13> kShaderInterfaces = {{
    {
        ShaderId::CumsumSinglePass,
        "cumsum_single_pass",
        make_span(kCumsumBindings),
        kCumsumDescriptorBindingCount,
        "ElementCountPushConstants",
        sizeof(ElementCountPushConstants),
        {VKSPLAT_CUMSUM_BLOCK_SIZE, "ceil(element_count / VKSPLAT_CUMSUM_BLOCK_SIZE),1,1"},
        kNoProfileNotes,
        {"slang/cumsum.slang", "slang", make_span(kCumsumSinglePassDefines)},
    },
    {
        ShaderId::CumsumBlockScan,
        "cumsum_block_scan",
        make_span(kCumsumBindings),
        kCumsumDescriptorBindingCount,
        "ElementCountPushConstants",
        sizeof(ElementCountPushConstants),
        {VKSPLAT_CUMSUM_BLOCK_SIZE, "ceil(phase_element_count / VKSPLAT_CUMSUM_BLOCK_SIZE),1,1"},
        kNoProfileNotes,
        {"slang/cumsum.slang", "slang", make_span(kCumsumBlockScanDefines)},
    },
    {
        ShaderId::CumsumScanBlockSums,
        "cumsum_scan_block_sums",
        make_span(kCumsumBindings),
        kCumsumDescriptorBindingCount,
        "ElementCountPushConstants",
        sizeof(ElementCountPushConstants),
        {VKSPLAT_CUMSUM_BLOCK_SIZE, "ceil(phase_element_count / VKSPLAT_CUMSUM_BLOCK_SIZE),1,1"},
        kNoProfileNotes,
        {"slang/cumsum.slang", "slang", make_span(kCumsumScanBlockSumsDefines)},
    },
    {
        ShaderId::CumsumAddBlockOffsets,
        "cumsum_add_block_offsets",
        make_span(kCumsumBindings),
        kCumsumDescriptorBindingCount,
        "ElementCountPushConstants",
        sizeof(ElementCountPushConstants),
        {VKSPLAT_CUMSUM_BLOCK_SIZE, "ceil(element_count / VKSPLAT_CUMSUM_BLOCK_SIZE),1,1"},
        kNoProfileNotes,
        {"slang/cumsum.slang", "slang", make_span(kCumsumAddBlockOffsetsDefines)},
    },
    {
        ShaderId::Sum,
        "sum",
        make_span(kSumBindings),
        kSumDescriptorBindingCount,
        "ElementCountPushConstants",
        sizeof(ElementCountPushConstants),
        {VKSPLAT_SUM_BLOCK_SIZE, "ceil(element_count / VKSPLAT_SUM_BLOCK_SIZE),1,1"},
        kNoProfileNotes,
        {"slang/sum.slang", "slang", make_span(kNoDefines)},
    },
    {
        ShaderId::Where,
        "where",
        make_span(kWhereBindings),
        kWhereDescriptorBindingCount,
        "ElementCountPushConstants",
        sizeof(ElementCountPushConstants),
        {VKSPLAT_WHERE_BLOCK_SIZE, "ceil(element_count / VKSPLAT_WHERE_BLOCK_SIZE),1,1"},
        kNoProfileNotes,
        {"slang/where.slang", "slang", make_span(kNoDefines)},
    },
    {
        ShaderId::ProjectionForward,
        "projection_forward",
        make_span(kProjectionForwardBindings),
        kProjectionForwardDescriptorBindingCount,
        "RendererUniforms",
        sizeof(RendererUniforms),
        {VKSPLAT_SUBGROUP_SIZE, "ceil(num_splats / VKSPLAT_SUBGROUP_SIZE),1,1"},
        kRectTileSpaceProfileNotes,
        {"slang/vertex_shader.slang", "slang", make_span(kProjectionForwardDefines)},
    },
    {
        ShaderId::GenerateKeys,
        "generate_keys",
        make_span(kGenerateKeysBindings),
        kGenerateKeysDescriptorBindingCount,
        "RendererUniforms",
        sizeof(RendererUniforms),
        {VKSPLAT_TILE_SHADER_GENERATE_KEYS_BLOCK_SIZE,
         "ceil(num_splats / VKSPLAT_TILE_SHADER_GENERATE_KEYS_BLOCK_SIZE),1,1"},
        kRectTileSpaceProfileNotes,
        {"slang/tile_shader.slang", "slang", make_span(kGenerateKeysDefines)},
    },
    {
        ShaderId::ComputeTileRanges,
        "compute_tile_ranges",
        make_span(kComputeTileRangesBindings),
        kComputeTileRangesDescriptorBindingCount,
        "RendererUniforms",
        sizeof(RendererUniforms),
        {VKSPLAT_TILE_SHADER_TILE_RANGES_THREADS,
         "ceil((num_indices + 1) / VKSPLAT_TILE_SHADER_TILE_RANGES_THREADS),1,1"},
        kNoProfileNotes,
        {"slang/tile_shader.slang", "slang", make_span(kComputeTileRangesDefines)},
    },
    {
        ShaderId::RasterizeForward,
        "rasterize_forward",
        make_span(kRasterizeForwardBindings),
        kRasterizeForwardDescriptorBindingCount,
        "RendererUniforms",
        sizeof(RendererUniforms),
        {VKSPLAT_TILE_SIZE, "ceil(image_width / VKSPLAT_TILE_WIDTH),ceil(image_height / VKSPLAT_TILE_HEIGHT),1"},
        kNoProfileNotes,
        {"slang/alphablend_shader.slang", "slang", make_span(kRasterizeForwardDefines)},
    },
    {
        ShaderId::RadixSortUpsweep,
        "radix_sort_upsweep",
        make_span(kRadixSortUpsweepBindings),
        kRadixSortUpsweepDescriptorBindingCount,
        "RadixSortPushConstants",
        sizeof(RadixSortPushConstants),
        {VKSPLAT_RADIX_PARTITION_SIZE, "ceil(element_count / VKSPLAT_RADIX_PARTITION_SIZE),1,1"},
        kRadixSortProfileNotes,
        {"shader/radix_sort/upsweep.comp", "glsl", make_span(kNoDefines)},
    },
    {
        ShaderId::RadixSortSpine,
        "radix_sort_spine",
        make_span(kRadixSortSpineBindings),
        kRadixSortSpineDescriptorBindingCount,
        "RadixSortPushConstants",
        sizeof(RadixSortPushConstants),
        {VKSPLAT_RADIX_SORT_RADIX, "VKSPLAT_RADIX_SORT_RADIX,1,1"},
        kRadixSortProfileNotes,
        {"shader/radix_sort/spine.comp", "glsl", make_span(kNoDefines)},
    },
    {
        ShaderId::RadixSortDownsweep,
        "radix_sort_downsweep",
        make_span(kRadixSortDownsweepBindings),
        kRadixSortDownsweepDescriptorBindingCount,
        "RadixSortPushConstants",
        sizeof(RadixSortPushConstants),
        {VKSPLAT_RADIX_PARTITION_SIZE, "ceil(element_count / VKSPLAT_RADIX_PARTITION_SIZE),1,1"},
        kRadixSortProfileNotes,
        {"shader/radix_sort/downsweep.comp", "glsl", make_span(kNoDefines)},
    },
}};
// clang-format on

// clang-format off
inline constexpr std::array<const char *, 6> kRadixSortPipelineFixtureBindings = {{
    "sorting_keys_1",
    "sorting_gauss_idx_1",
    "sorting_keys_2",
    "sorting_gauss_idx_2",
    "_sorting_histogram",
    "_sorting_histogram_cumsum",
}};

inline constexpr std::array<const char *, 3> kCumsumMultiBlockFixtureBindings = {{
    "input",
    "output",
    "block_sums",
}};

inline constexpr std::array<const char *, 4> kCumsumTwoLevelFixtureBindings = {{
    "input",
    "output",
    "block_sums",
    "block_sums2",
}};

inline constexpr std::array<const char *, 1> kOutputFixtureBindings = {{
    "output",
}};

inline constexpr std::array<const char *, 1> kWhereOutputFixtureBindings = {{
    "out_indices",
}};

inline constexpr std::array<const char *, 2> kGenerateKeysOutputFixtureBindings = {{
    "unsorted_keys",
    "unsorted_gauss_idx",
}};

inline constexpr std::array<const char *, 1> kComputeTileRangesOutputFixtureBindings = {{
    "tile_ranges",
}};

inline constexpr std::array<const char *, 2> kRadixSortOutputFixtureBindings = {{
    "sorted_keys",
    "sorted_gauss_idx",
}};

inline constexpr std::array<const char *, 0> kNoExactGoldenOutputFixtureBindings = {};

inline constexpr std::array<FixtureBindingContract, 10> kFixtureBindingContracts = {{
    {"radix_sort_pipeline", make_span(kRadixSortPipelineFixtureBindings)},
    {"cumsum_multi_block", make_span(kCumsumMultiBlockFixtureBindings)},
    {"cumsum_multi_block_two_level", make_span(kCumsumTwoLevelFixtureBindings)},
    {"cumsum_output", make_span(kOutputFixtureBindings)},
    {"sum_output", make_span(kOutputFixtureBindings)},
    {"where_output", make_span(kWhereOutputFixtureBindings)},
    {"generate_keys_output", make_span(kGenerateKeysOutputFixtureBindings)},
    {"compute_tile_ranges_output", make_span(kComputeTileRangesOutputFixtureBindings)},
    {"radix_sort_output", make_span(kRadixSortOutputFixtureBindings)},
    {"no_exact_golden_outputs", make_span(kNoExactGoldenOutputFixtureBindings)},
}};
// clang-format on

} // namespace detail

[[nodiscard]] constexpr Span<const ShaderBindingDescriptor> shader_binding_descriptors(ShaderId id) noexcept {
  switch (id) {
    case ShaderId::CumsumSinglePass:
    case ShaderId::CumsumBlockScan:
    case ShaderId::CumsumScanBlockSums:
    case ShaderId::CumsumAddBlockOffsets:
      return make_span(detail::kCumsumBindings);
    case ShaderId::Sum:
      return make_span(detail::kSumBindings);
    case ShaderId::Where:
      return make_span(detail::kWhereBindings);
    case ShaderId::ProjectionForward:
      return make_span(detail::kProjectionForwardBindings);
    case ShaderId::GenerateKeys:
      return make_span(detail::kGenerateKeysBindings);
    case ShaderId::ComputeTileRanges:
      return make_span(detail::kComputeTileRangesBindings);
    case ShaderId::RasterizeForward:
      return make_span(detail::kRasterizeForwardBindings);
    case ShaderId::RadixSortUpsweep:
      return make_span(detail::kRadixSortUpsweepBindings);
    case ShaderId::RadixSortSpine:
      return make_span(detail::kRadixSortSpineBindings);
    case ShaderId::RadixSortDownsweep:
      return make_span(detail::kRadixSortDownsweepBindings);
  }
  return {};
}

[[nodiscard]] inline Span<const ShaderInterface> shader_interfaces() noexcept {
  return make_span(detail::kShaderInterfaces);
}

[[nodiscard]] inline const ShaderInterface &shader_interface(ShaderId id) {
  for (const auto &shader : detail::kShaderInterfaces) {
    if (shader.id == id) {
      return shader;
    }
  }
  throw std::invalid_argument("Unknown shader id");
}

[[nodiscard]] inline const ShaderInterface &shader_interface(std::string_view logical_name) {
  for (const auto &shader : detail::kShaderInterfaces) {
    if (shader.logical_name == logical_name) {
      return shader;
    }
  }
  throw std::invalid_argument("Unknown shader interface: " + std::string(logical_name));
}

[[nodiscard]] inline Span<const FixtureBindingContract> fixture_binding_contracts() noexcept {
  return make_span(detail::kFixtureBindingContracts);
}

[[nodiscard]] inline const FixtureBindingContract &fixture_binding_contract(std::string_view name) {
  for (const auto &contract : detail::kFixtureBindingContracts) {
    if (contract.name == name) {
      return contract;
    }
  }
  throw std::invalid_argument("Unknown fixture binding contract: " + std::string(name));
}

[[nodiscard]] inline std::vector<std::string> binding_names(const ShaderInterface &shader) {
  std::vector<std::string> names;
  names.reserve(shader.bindings.size());
  for (std::size_t index = 0; index < shader.bindings.size(); ++index) {
    names.emplace_back(shader.bindings[index].name);
  }
  return names;
}

[[nodiscard]] inline std::vector<std::string> binding_names(const FixtureBindingContract &contract) {
  std::vector<std::string> names;
  names.reserve(contract.bindings.size());
  for (std::size_t index = 0; index < contract.bindings.size(); ++index) {
    names.emplace_back(contract.bindings[index]);
  }
  return names;
}

} // namespace nlrc::vksplat::gpu
