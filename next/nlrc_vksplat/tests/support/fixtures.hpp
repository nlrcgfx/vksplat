#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/shader_execution.hpp"
#include "nlrc_vksplat_config.hpp"

namespace nlrc::vksplat::tests {

inline constexpr std::uint32_t kDefaultImageHeight = 32;
inline constexpr std::uint32_t kDefaultImageWidth = 32;
inline constexpr std::uint32_t kDefaultGridHeight = 2;
inline constexpr std::uint32_t kDefaultGridWidth = 2;
inline constexpr std::size_t kPixelChannels = 4;

struct ProjectionFixtureData final {
  std::vector<float> xyz_ws;
  std::vector<float> sh_coeffs;
  std::vector<float> rotations;
  std::vector<float> scales_opacs;
  std::vector<std::int32_t> tiles_touched;
  std::vector<RectTileSpace> rect_tile_space;
  std::vector<std::int32_t> radii;
  std::vector<float> xy_vs;
  std::vector<float> depths;
  std::vector<float> inv_cov_vs_opacity;
  std::vector<float> rgb;
};

struct ProjectionFixtureBuffers final {
  gpu::StorageBuffer xyz_ws;
  gpu::StorageBuffer sh_coeffs;
  gpu::StorageBuffer rotations;
  gpu::StorageBuffer scales_opacs;
  gpu::StorageBuffer tiles_touched;
  gpu::StorageBuffer rect_tile_space;
  gpu::StorageBuffer radii;
  gpu::StorageBuffer xy_vs;
  gpu::StorageBuffer depths;
  gpu::StorageBuffer inv_cov_vs_opacity;
  gpu::StorageBuffer rgb;
  gpu::ProjectionForwardBindings bindings{};

  ProjectionFixtureBuffers(gpu::StorageBuffer xyz_ws_buffer,
                           gpu::StorageBuffer sh_coeffs_buffer,
                           gpu::StorageBuffer rotations_buffer,
                           gpu::StorageBuffer scales_opacs_buffer,
                           gpu::StorageBuffer tiles_touched_buffer,
                           gpu::StorageBuffer rect_tile_space_buffer,
                           gpu::StorageBuffer radii_buffer,
                           gpu::StorageBuffer xy_vs_buffer,
                           gpu::StorageBuffer depths_buffer,
                           gpu::StorageBuffer inv_cov_vs_opacity_buffer,
                           gpu::StorageBuffer rgb_buffer);

  ProjectionFixtureBuffers(const ProjectionFixtureBuffers &) = delete;
  ProjectionFixtureBuffers &operator=(const ProjectionFixtureBuffers &) = delete;

  ProjectionFixtureBuffers(ProjectionFixtureBuffers &&other) noexcept;
  ProjectionFixtureBuffers &operator=(ProjectionFixtureBuffers &&other) noexcept;

private:
  void refresh_bindings() noexcept;
};

struct PixelState final {
  float r;
  float g;
  float b;
  float transmittance;
};

[[nodiscard]] gpu::push_constants::Renderer default_renderer_uniforms(std::uint32_t num_splats);

[[nodiscard]] std::string int64_profile_stage_name(std::string_view base);

[[nodiscard]] ProjectionFixtureData load_projection_fixture(const std::string &stage_name);

[[nodiscard]] ProjectionFixtureBuffers upload_projection_fixture(const gpu::HeadlessContext &context,
                                                                 const ProjectionFixtureData &fixture);

void require_valid_tile_ranges(const std::vector<std::int32_t> &tile_ranges,
                               const std::vector<std::uint32_t> &sorted_keys,
                               std::uint32_t grid_width,
                               std::uint32_t grid_height);

void require_valid_tile_ranges(const std::vector<std::int32_t> &tile_ranges,
                               const std::vector<std::uint32_t> &sorted_keys);

void require_pixel_invariants(const std::vector<float> &pixel_state,
                              const std::vector<std::int32_t> &n_contributors,
                              std::uint32_t image_width,
                              std::uint32_t image_height,
                              std::size_t num_indices);

void require_pixel_invariants(const std::vector<float> &pixel_state,
                              const std::vector<std::int32_t> &n_contributors,
                              std::size_t num_indices);

[[nodiscard]] PixelState
pixel_at(const std::vector<float> &pixel_state, std::uint32_t image_width, std::uint32_t x, std::uint32_t y);

[[nodiscard]] std::int32_t contributor_at(const std::vector<std::int32_t> &n_contributors,
                                          std::uint32_t image_width,
                                          std::uint32_t x,
                                          std::uint32_t y);

} // namespace nlrc::vksplat::tests
