#pragma once

#include <cstddef>
#include <cstdint>

#include "gpu/compute_pipeline.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/shader_descriptors.hpp"
#include "gpu/storage_buffer.hpp"
#include "nlrc_vksplat_config.hpp"

namespace nlrc::vksplat::gpu {

inline constexpr std::uint32_t kCumsumBindingCount = kCumsumDescriptorBindingCount;
inline constexpr std::uint32_t kSumBindingCount = kSumDescriptorBindingCount;
inline constexpr std::uint32_t kWhereBindingCount = kWhereDescriptorBindingCount;
inline constexpr std::uint32_t kGenerateKeysBindingCount = kGenerateKeysDescriptorBindingCount;
inline constexpr std::uint32_t kComputeTileRangesBindingCount = kComputeTileRangesDescriptorBindingCount;
inline constexpr std::uint32_t kRasterizeForwardBindingCount = kRasterizeForwardDescriptorBindingCount;
inline constexpr std::uint32_t kProjectionForwardBindingCount = kProjectionForwardDescriptorBindingCount;
inline constexpr std::uint32_t kRadixSortUpsweepBindingCount = kRadixSortUpsweepDescriptorBindingCount;
inline constexpr std::uint32_t kRadixSortSpineBindingCount = kRadixSortSpineDescriptorBindingCount;
inline constexpr std::uint32_t kRadixSortDownsweepBindingCount = kRadixSortDownsweepDescriptorBindingCount;
inline constexpr std::uint32_t kRadixSortPasses = VKSPLAT_SORTING_KEY_BITS / VKSPLAT_RADIX_BITS_PER_PASS;

static_assert(VKSPLAT_SORTING_KEY_BITS % VKSPLAT_RADIX_BITS_PER_PASS == 0);

struct ProjectionForwardBindings final {
  const StorageBuffer *xyz_ws{};
  const StorageBuffer *sh_coeffs{};
  const StorageBuffer *rotations{};
  const StorageBuffer *scales_opacs{};
  StorageBuffer *tiles_touched{};
  StorageBuffer *rect_tile_space{};
  StorageBuffer *radii{};
  StorageBuffer *xy_vs{};
  StorageBuffer *depths{};
  StorageBuffer *inv_cov_vs_opacity{};
  StorageBuffer *rgb{};
};

struct GenerateKeysBindings final {
  const StorageBuffer *xy_vs{};
  const StorageBuffer *inv_cov_vs_opacity{};
  const StorageBuffer *depths{};
  const StorageBuffer *rect_tile_space{};
  const StorageBuffer *index_buffer_offset{};
  StorageBuffer *unsorted_keys{};
  StorageBuffer *unsorted_gauss_idx{};
};

struct ComputeTileRangesBindings final {
  const StorageBuffer *sorted_keys{};
  StorageBuffer *tile_ranges{};
};

struct RasterizeForwardBindings final {
  const StorageBuffer *sorted_gauss_idx{};
  const StorageBuffer *tile_ranges{};
  const StorageBuffer *xy_vs{};
  const StorageBuffer *inv_cov_vs_opacity{};
  const StorageBuffer *rgb{};
  StorageBuffer *pixel_state{};
  StorageBuffer *n_contributors{};
};

struct RadixSortBindings final {
  StorageBuffer *keys_1{};
  StorageBuffer *indices_1{};
  StorageBuffer *keys_2{};
  StorageBuffer *indices_2{};
};

struct RadixSortResult final {
  const StorageBuffer *keys{};
  const StorageBuffer *indices{};
};

struct ForwardBindings final {
  ProjectionForwardBindings projection{};
  StorageBuffer *index_buffer_offset{};
  StorageBuffer *sorting_keys_1{};
  StorageBuffer *sorting_gauss_idx_1{};
  StorageBuffer *sorting_keys_2{};
  StorageBuffer *sorting_gauss_idx_2{};
  StorageBuffer *tile_ranges{};
  StorageBuffer *pixel_state{};
  StorageBuffer *n_contributors{};
};

struct ForwardResult final {
  std::size_t num_indices{};
  bool rasterized{};
  const StorageBuffer *sorted_keys{};
  const StorageBuffer *sorted_gauss_idx{};
};

[[nodiscard]] std::size_t ceil_div(std::size_t value, std::size_t divisor);
[[nodiscard]] DispatchShape dispatch_groups_for(std::size_t element_count, std::uint32_t block_size);

void execute_cumsum(const HeadlessContext &context,
                    const StorageBuffer &input,
                    StorageBuffer &output,
                    std::size_t element_count);

void execute_sum(const HeadlessContext &context,
                 const StorageBuffer &input,
                 StorageBuffer &output,
                 std::size_t element_count);

void execute_where(const HeadlessContext &context,
                   const StorageBuffer &mask,
                   const StorageBuffer &mask_cumsum,
                   StorageBuffer &out_indices,
                   std::size_t element_count);

void execute_projection_forward(const HeadlessContext &context,
                                const ProjectionForwardBindings &bindings,
                                const RendererUniforms &uniforms);

void execute_generate_keys(const HeadlessContext &context,
                           const GenerateKeysBindings &bindings,
                           const RendererUniforms &uniforms,
                           std::size_t output_count);

void execute_compute_tile_ranges(const HeadlessContext &context,
                                 const ComputeTileRangesBindings &bindings,
                                 const RendererUniforms &uniforms,
                                 std::size_t num_indices);

void execute_rasterize_forward(const HeadlessContext &context,
                               const RasterizeForwardBindings &bindings,
                               const RendererUniforms &uniforms,
                               std::size_t num_indices);

[[nodiscard]] RadixSortResult
execute_sort(const HeadlessContext &context, const RadixSortBindings &bindings, std::size_t element_count);

[[nodiscard]] ForwardResult
execute_forward(const HeadlessContext &context, const ForwardBindings &bindings, const RendererUniforms &uniforms);

} // namespace nlrc::vksplat::gpu
