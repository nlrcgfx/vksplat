#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

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

#define NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(struct_type, field_name, shader_id, index)                            \
  static_assert(storage_binding_name_equal(#field_name, shader_binding_name<ShaderId::shader_id, index>()),            \
                "Binding struct field must match the shader registry binding name")

#define NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(struct_type, left_field, right_field)                                   \
  static_assert(offsetof(struct_type, left_field) < offsetof(struct_type, right_field),                                \
                "Binding struct fields must stay in shader registry order")

static_assert(std::is_standard_layout_v<ProjectionForwardBindings>);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, xyz_ws, ProjectionForward, 0);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, sh_coeffs, ProjectionForward, 1);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, rotations, ProjectionForward, 2);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, scales_opacs, ProjectionForward, 3);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, tiles_touched, ProjectionForward, 4);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, rect_tile_space, ProjectionForward, 5);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, radii, ProjectionForward, 6);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, xy_vs, ProjectionForward, 7);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, depths, ProjectionForward, 8);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, inv_cov_vs_opacity, ProjectionForward, 9);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ProjectionForwardBindings, rgb, ProjectionForward, 10);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, xyz_ws, sh_coeffs);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, sh_coeffs, rotations);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, rotations, scales_opacs);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, scales_opacs, tiles_touched);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, tiles_touched, rect_tile_space);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, rect_tile_space, radii);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, radii, xy_vs);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, xy_vs, depths);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, depths, inv_cov_vs_opacity);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ProjectionForwardBindings, inv_cov_vs_opacity, rgb);

static_assert(std::is_standard_layout_v<GenerateKeysBindings>);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(GenerateKeysBindings, xy_vs, GenerateKeys, 0);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(GenerateKeysBindings, inv_cov_vs_opacity, GenerateKeys, 1);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(GenerateKeysBindings, depths, GenerateKeys, 2);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(GenerateKeysBindings, rect_tile_space, GenerateKeys, 3);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(GenerateKeysBindings, index_buffer_offset, GenerateKeys, 4);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(GenerateKeysBindings, unsorted_keys, GenerateKeys, 5);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(GenerateKeysBindings, unsorted_gauss_idx, GenerateKeys, 6);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(GenerateKeysBindings, xy_vs, inv_cov_vs_opacity);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(GenerateKeysBindings, inv_cov_vs_opacity, depths);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(GenerateKeysBindings, depths, rect_tile_space);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(GenerateKeysBindings, rect_tile_space, index_buffer_offset);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(GenerateKeysBindings, index_buffer_offset, unsorted_keys);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(GenerateKeysBindings, unsorted_keys, unsorted_gauss_idx);

static_assert(std::is_standard_layout_v<ComputeTileRangesBindings>);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ComputeTileRangesBindings, sorted_keys, ComputeTileRanges, 0);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(ComputeTileRangesBindings, tile_ranges, ComputeTileRanges, 1);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(ComputeTileRangesBindings, sorted_keys, tile_ranges);

static_assert(std::is_standard_layout_v<RasterizeForwardBindings>);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(RasterizeForwardBindings, sorted_gauss_idx, RasterizeForward, 0);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(RasterizeForwardBindings, tile_ranges, RasterizeForward, 1);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(RasterizeForwardBindings, xy_vs, RasterizeForward, 2);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(RasterizeForwardBindings, inv_cov_vs_opacity, RasterizeForward, 3);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(RasterizeForwardBindings, rgb, RasterizeForward, 4);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(RasterizeForwardBindings, pixel_state, RasterizeForward, 5);
NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD(RasterizeForwardBindings, n_contributors, RasterizeForward, 6);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(RasterizeForwardBindings, sorted_gauss_idx, tile_ranges);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(RasterizeForwardBindings, tile_ranges, xy_vs);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(RasterizeForwardBindings, xy_vs, inv_cov_vs_opacity);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(RasterizeForwardBindings, inv_cov_vs_opacity, rgb);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(RasterizeForwardBindings, rgb, pixel_state);
NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER(RasterizeForwardBindings, pixel_state, n_contributors);

#undef NLRC_VKSPLAT_ASSERT_STRUCT_BINDING_FIELD
#undef NLRC_VKSPLAT_ASSERT_STRUCT_FIELD_ORDER

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
