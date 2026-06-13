#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gpu/shader_execution.hpp"
#include "span.hpp"

namespace nlrc::vksplat::gpu {

struct NamedStorageBinding final {
  std::string_view name;
  const StorageBuffer *buffer{};
};

[[nodiscard]] inline auto storage_buffer_by_name(Span<const NamedStorageBinding> bindings, std::string_view name)
    -> const StorageBuffer * {
  for (std::size_t index = 0; index < bindings.size(); ++index) {
    if (bindings[index].name == name) {
      return bindings[index].buffer;
    }
  }
  throw std::invalid_argument("Unknown storage binding: " + std::string(name));
}

[[nodiscard]] inline std::vector<const StorageBuffer *>
resolve_storage_bindings(const ShaderInterface &shader, Span<const NamedStorageBinding> named_bindings) {
  std::vector<const StorageBuffer *> buffers;
  buffers.reserve(shader.binding_count);
  for (std::size_t index = 0; index < shader.bindings.size(); ++index) {
    buffers.push_back(storage_buffer_by_name(named_bindings, shader.bindings[index].name));
  }
  return buffers;
}

namespace detail {

template <ShaderId Id, std::size_t... Indices>
// NOLINTNEXTLINE(readability-named-parameter)
[[nodiscard]] constexpr bool storage_binding_names_match_registry(std::index_sequence<Indices...>) noexcept {
  const auto descriptors = shader_binding_descriptors(Id);
  return descriptors.size() == shader_binding_count_v<Id> &&
         (storage_binding_name_equal(descriptors[Indices].name, shader_binding_name<Id, Indices>()) && ...);
}

template <ShaderId Id, std::size_t... Indices, typename... Buffers>
// NOLINTNEXTLINE(readability-named-parameter)
[[nodiscard]] constexpr auto make_storage_bindings_impl(std::index_sequence<Indices...>, Buffers *...buffers) {
  return std::array<NamedStorageBinding, sizeof...(Indices)>{{
      {shader_binding_name<Id, Indices>(), buffers}...,
  }};
}

} // namespace detail

#define NLRC_VKSPLAT_SHADER_STORAGE_BINDING(shader_id, index, binding_name, buffer_expr)                               \
  ([&]() -> ::nlrc::vksplat::gpu::NamedStorageBinding {                                                                \
    static_assert(::nlrc::vksplat::gpu::storage_binding_name_equal(                                                    \
                      #binding_name,                                                                                   \
                      ::nlrc::vksplat::gpu::shader_binding_name<::nlrc::vksplat::gpu::ShaderId::shader_id, index>()),  \
                  "Storage binding helper token must match the shader registry binding name");                         \
    return {::nlrc::vksplat::gpu::shader_binding_name<::nlrc::vksplat::gpu::ShaderId::shader_id, index>(),             \
            (buffer_expr)};                                                                                            \
  }())

template <ShaderId Id>
[[nodiscard]] constexpr bool storage_binding_names_match_registry() noexcept {
  return detail::storage_binding_names_match_registry<Id>(std::make_index_sequence<shader_binding_count_v<Id>>{});
}

template <ShaderId Id, typename... Buffers>
[[nodiscard]] constexpr auto make_storage_bindings(Buffers *...buffers) {
  static_assert(sizeof...(Buffers) == shader_binding_count_v<Id>,
                "Storage binding helper argument count must match the shader registry");
  return detail::make_storage_bindings_impl<Id>(std::make_index_sequence<shader_binding_count_v<Id>>{}, buffers...);
}

[[nodiscard]] inline std::array<NamedStorageBinding, kCumsumBindingCount>
cumsum_storage_bindings(const StorageBuffer &input, const StorageBuffer &output, const StorageBuffer &block_sums) {
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(CumsumSinglePass, 0, input, &input),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(CumsumSinglePass, 1, output, &output),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(CumsumSinglePass, 2, block_sums, &block_sums),
  }};
}

[[nodiscard]] inline std::array<NamedStorageBinding, kSumBindingCount>
sum_storage_bindings(const StorageBuffer &input, const StorageBuffer &output) {
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(Sum, 0, input, &input),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(Sum, 1, output, &output),
  }};
}

[[nodiscard]] inline std::array<NamedStorageBinding, kWhereBindingCount>
where_storage_bindings(const StorageBuffer &mask, const StorageBuffer &mask_cumsum, const StorageBuffer &out_indices) {
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(Where, 0, mask, &mask),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(Where, 1, mask_cumsum, &mask_cumsum),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(Where, 2, out_indices, &out_indices),
  }};
}

[[nodiscard]] inline std::array<NamedStorageBinding, kProjectionForwardBindingCount>
projection_forward_storage_bindings(const ProjectionForwardBindings &bindings) {
  // clang-format off
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 0, xyz_ws, bindings.xyz_ws),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 1, sh_coeffs, bindings.sh_coeffs),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 2, rotations, bindings.rotations),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 3, scales_opacs, bindings.scales_opacs),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 4, tiles_touched, bindings.tiles_touched),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 5, rect_tile_space, bindings.rect_tile_space),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 6, radii, bindings.radii),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 7, xy_vs, bindings.xy_vs),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 8, depths, bindings.depths),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 9, inv_cov_vs_opacity, bindings.inv_cov_vs_opacity),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ProjectionForward, 10, rgb, bindings.rgb),
  }};
  // clang-format on
}

[[nodiscard]] inline std::array<NamedStorageBinding, kGenerateKeysBindingCount>
generate_keys_storage_bindings(const GenerateKeysBindings &bindings) {
  // clang-format off
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(GenerateKeys, 0, xy_vs, bindings.xy_vs),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(GenerateKeys, 1, inv_cov_vs_opacity, bindings.inv_cov_vs_opacity),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(GenerateKeys, 2, depths, bindings.depths),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(GenerateKeys, 3, rect_tile_space, bindings.rect_tile_space),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(GenerateKeys, 4, index_buffer_offset, bindings.index_buffer_offset),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(GenerateKeys, 5, unsorted_keys, bindings.unsorted_keys),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(GenerateKeys, 6, unsorted_gauss_idx, bindings.unsorted_gauss_idx),
  }};
  // clang-format on
}

[[nodiscard]] inline std::array<NamedStorageBinding, kComputeTileRangesBindingCount>
compute_tile_ranges_storage_bindings(const ComputeTileRangesBindings &bindings) {
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ComputeTileRanges, 0, sorted_keys, bindings.sorted_keys),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(ComputeTileRanges, 1, tile_ranges, bindings.tile_ranges),
  }};
}

[[nodiscard]] inline std::array<NamedStorageBinding, kRasterizeForwardBindingCount>
rasterize_forward_storage_bindings(const RasterizeForwardBindings &bindings) {
  // clang-format off
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RasterizeForward, 0, sorted_gauss_idx, bindings.sorted_gauss_idx),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RasterizeForward, 1, tile_ranges, bindings.tile_ranges),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RasterizeForward, 2, xy_vs, bindings.xy_vs),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RasterizeForward, 3, inv_cov_vs_opacity, bindings.inv_cov_vs_opacity),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RasterizeForward, 4, rgb, bindings.rgb),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RasterizeForward, 5, pixel_state, bindings.pixel_state),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RasterizeForward, 6, n_contributors, bindings.n_contributors),
  }};
  // clang-format on
}

[[nodiscard]] inline std::array<NamedStorageBinding, kRadixSortUpsweepBindingCount>
radix_sort_upsweep_storage_bindings(const StorageBuffer &unsorted_keys,
                                    const StorageBuffer &sorting_histogram,
                                    const StorageBuffer &sorting_histogram_cumsum) {
  // clang-format off
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortUpsweep, 0, unsorted_keys, &unsorted_keys),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortUpsweep, 1, _sorting_histogram, &sorting_histogram),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortUpsweep, 2, _sorting_histogram_cumsum,
                                          &sorting_histogram_cumsum),
  }};
  // clang-format on
}

[[nodiscard]] inline std::array<NamedStorageBinding, kRadixSortSpineBindingCount>
radix_sort_spine_storage_bindings(const StorageBuffer &sorting_histogram,
                                  const StorageBuffer &sorting_histogram_cumsum) {
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortSpine, 0, _sorting_histogram, &sorting_histogram),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortSpine, 1, _sorting_histogram_cumsum, &sorting_histogram_cumsum),
  }};
}

[[nodiscard]] inline std::array<NamedStorageBinding, kRadixSortDownsweepBindingCount>
radix_sort_downsweep_storage_bindings(const StorageBuffer &sorting_histogram,
                                      const StorageBuffer &sorting_histogram_cumsum,
                                      const StorageBuffer &unsorted_keys,
                                      const StorageBuffer &unsorted_gauss_idx,
                                      const StorageBuffer &sorted_keys,
                                      const StorageBuffer &sorted_gauss_idx) {
  // clang-format off
  return {{
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortDownsweep, 0, _sorting_histogram, &sorting_histogram),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortDownsweep, 1, _sorting_histogram_cumsum,
                                          &sorting_histogram_cumsum),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortDownsweep, 2, unsorted_keys, &unsorted_keys),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortDownsweep, 3, unsorted_gauss_idx, &unsorted_gauss_idx),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortDownsweep, 4, sorted_keys, &sorted_keys),
      NLRC_VKSPLAT_SHADER_STORAGE_BINDING(RadixSortDownsweep, 5, sorted_gauss_idx, &sorted_gauss_idx),
  }};
  // clang-format on
}

} // namespace nlrc::vksplat::gpu
