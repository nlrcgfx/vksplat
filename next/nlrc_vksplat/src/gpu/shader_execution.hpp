#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "gpu/compute_pipeline.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/storage_buffer.hpp"
#include "nlrc_vksplat_config.hpp"

namespace nlrc::vksplat::gpu {

inline constexpr std::uint32_t kCumsumBindingCount = 3;
inline constexpr std::uint32_t kSumBindingCount = 2;
inline constexpr std::uint32_t kWhereBindingCount = 3;
inline constexpr std::uint32_t kGenerateKeysBindingCount = 7;
inline constexpr std::uint32_t kProjectionForwardBindingCount = 11;
inline constexpr std::uint32_t kRadixSortUpsweepBindingCount = 3;
inline constexpr std::uint32_t kRadixSortSpineBindingCount = 2;
inline constexpr std::uint32_t kRadixSortDownsweepBindingCount = 6;
inline constexpr std::uint32_t kRadixSortPasses = VKSPLAT_SORTING_KEY_BITS / VKSPLAT_RADIX_BITS_PER_PASS;

static_assert(VKSPLAT_SORTING_KEY_BITS % VKSPLAT_RADIX_BITS_PER_PASS == 0);

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

[[nodiscard]] RadixSortResult
execute_sort(const HeadlessContext &context, const RadixSortBindings &bindings, std::size_t element_count);

} // namespace nlrc::vksplat::gpu
