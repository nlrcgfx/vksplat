#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "gpu/shader_descriptors.hpp"
#include "nlrc_vksplat_config.hpp"

namespace nlrc::vksplat::tests {

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

struct PixelState final {
  float r;
  float g;
  float b;
  float transmittance;
};

[[nodiscard]] gpu::push_constants::Renderer default_renderer_uniforms(std::uint32_t num_splats);

[[nodiscard]] std::string int64_profile_stage_name(std::string_view base);

[[nodiscard]] ProjectionFixtureData load_projection_fixture(const std::string &stage_name);

void require_valid_tile_ranges(const std::vector<std::int32_t> &tile_ranges,
                               const std::vector<std::uint32_t> &sorted_keys,
                               std::uint32_t grid_width,
                               std::uint32_t grid_height);

void require_valid_tile_ranges(const std::vector<std::int32_t> &tile_ranges,
                               const std::vector<std::uint32_t> &sorted_keys);

[[nodiscard]] PixelState
pixel_at(const std::vector<float> &pixel_state, std::uint32_t image_width, std::uint32_t x, std::uint32_t y);

[[nodiscard]] std::int32_t contributor_at(const std::vector<std::int32_t> &n_contributors,
                                          std::uint32_t image_width,
                                          std::uint32_t x,
                                          std::uint32_t y);

} // namespace nlrc::vksplat::tests
