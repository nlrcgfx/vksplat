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
  return make_storage_bindings<ShaderId::CumsumSinglePass>(&input, &output, &block_sums);
}

[[nodiscard]] inline std::array<NamedStorageBinding, kSumBindingCount>
sum_storage_bindings(const StorageBuffer &input, const StorageBuffer &output) {
  return make_storage_bindings<ShaderId::Sum>(&input, &output);
}

[[nodiscard]] inline std::array<NamedStorageBinding, kWhereBindingCount>
where_storage_bindings(const StorageBuffer &mask, const StorageBuffer &mask_cumsum, const StorageBuffer &out_indices) {
  return make_storage_bindings<ShaderId::Where>(&mask, &mask_cumsum, &out_indices);
}

[[nodiscard]] inline std::array<NamedStorageBinding, kProjectionForwardBindingCount>
projection_forward_storage_bindings(const ProjectionForwardBindings &bindings) {
  // clang-format off
  return make_storage_bindings<ShaderId::ProjectionForward>(
      bindings.xyz_ws,
      bindings.sh_coeffs,
      bindings.rotations,
      bindings.scales_opacs,
      bindings.tiles_touched,
      bindings.rect_tile_space,
      bindings.radii,
      bindings.xy_vs,
      bindings.depths,
      bindings.inv_cov_vs_opacity,
      bindings.rgb);
  // clang-format on
}

[[nodiscard]] inline std::array<NamedStorageBinding, kGenerateKeysBindingCount>
generate_keys_storage_bindings(const GenerateKeysBindings &bindings) {
  // clang-format off
  return make_storage_bindings<ShaderId::GenerateKeys>(
      bindings.xy_vs,
      bindings.inv_cov_vs_opacity,
      bindings.depths,
      bindings.rect_tile_space,
      bindings.index_buffer_offset,
      bindings.unsorted_keys,
      bindings.unsorted_gauss_idx);
  // clang-format on
}

[[nodiscard]] inline std::array<NamedStorageBinding, kComputeTileRangesBindingCount>
compute_tile_ranges_storage_bindings(const ComputeTileRangesBindings &bindings) {
  return make_storage_bindings<ShaderId::ComputeTileRanges>(bindings.sorted_keys, bindings.tile_ranges);
}

[[nodiscard]] inline std::array<NamedStorageBinding, kRasterizeForwardBindingCount>
rasterize_forward_storage_bindings(const RasterizeForwardBindings &bindings) {
  // clang-format off
  return make_storage_bindings<ShaderId::RasterizeForward>(
      bindings.sorted_gauss_idx,
      bindings.tile_ranges,
      bindings.xy_vs,
      bindings.inv_cov_vs_opacity,
      bindings.rgb,
      bindings.pixel_state,
      bindings.n_contributors);
  // clang-format on
}

[[nodiscard]] inline std::array<NamedStorageBinding, kRadixSortUpsweepBindingCount>
radix_sort_upsweep_storage_bindings(const StorageBuffer &unsorted_keys,
                                    const StorageBuffer &sorting_histogram,
                                    const StorageBuffer &sorting_histogram_cumsum) {
  // clang-format off
  return make_storage_bindings<ShaderId::RadixSortUpsweep>(
      &unsorted_keys,
      &sorting_histogram,
      &sorting_histogram_cumsum);
  // clang-format on
}

[[nodiscard]] inline std::array<NamedStorageBinding, kRadixSortSpineBindingCount>
radix_sort_spine_storage_bindings(const StorageBuffer &sorting_histogram,
                                  const StorageBuffer &sorting_histogram_cumsum) {
  return make_storage_bindings<ShaderId::RadixSortSpine>(&sorting_histogram, &sorting_histogram_cumsum);
}

[[nodiscard]] inline std::array<NamedStorageBinding, kRadixSortDownsweepBindingCount>
radix_sort_downsweep_storage_bindings(const StorageBuffer &sorting_histogram,
                                      const StorageBuffer &sorting_histogram_cumsum,
                                      const StorageBuffer &unsorted_keys,
                                      const StorageBuffer &unsorted_gauss_idx,
                                      const StorageBuffer &sorted_keys,
                                      const StorageBuffer &sorted_gauss_idx) {
  // clang-format off
  return make_storage_bindings<ShaderId::RadixSortDownsweep>(
      &sorting_histogram,
      &sorting_histogram_cumsum,
      &unsorted_keys,
      &unsorted_gauss_idx,
      &sorted_keys,
      &sorted_gauss_idx);
  // clang-format on
}

} // namespace nlrc::vksplat::gpu
