#include "fixtures.hpp"

#include <algorithm>
#include <cstddef>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"

namespace nlrc::vksplat::tests {

namespace {

constexpr std::uint32_t kDefaultImageHeight = 32;
constexpr std::uint32_t kDefaultImageWidth = 32;
constexpr std::uint32_t kDefaultGridHeight = 2;
constexpr std::uint32_t kDefaultGridWidth = 2;
constexpr std::uint32_t kDepthBits = 23;
constexpr std::size_t kPixelChannels = 4;

} // namespace

[[nodiscard]] gpu::push_constants::Renderer default_renderer_uniforms(std::uint32_t num_splats) {
  return {
      kDefaultImageHeight,
      kDefaultImageWidth,
      kDefaultGridHeight,
      kDefaultGridWidth,
      num_splats,
      0U,
      0U,
      0U,
      16.0F,
      16.0F,
      16.0F,
      16.0F,
      {0.0F, 0.0F, 0.0F, 0.0F},
      {
          1.0F,
          0.0F,
          0.0F,
          0.0F,
          0.0F,
          1.0F,
          0.0F,
          0.0F,
          0.0F,
          0.0F,
          1.0F,
          0.0F,
          0.0F,
          0.0F,
          0.0F,
          1.0F,
      },
  };
}

[[nodiscard]] std::string int64_profile_stage_name(std::string_view base) {
  std::string stage_name{base};
  if constexpr (kUseEmulatedInt64) {
    stage_name += "_emulated_int64";
  } else {
    stage_name += "_native_int64";
  }
  return stage_name;
}

[[nodiscard]] ProjectionFixtureData load_projection_fixture(const std::string &stage_name) {
  const auto fixture_root = fixture_dir(stage_name);
  const auto manifest = load_fixture_manifest(fixture_root / "manifest.json");

  return {
      load_fixture_buffer<float>(fixture_root, manifest, "xyz_ws"),
      load_fixture_buffer<float>(fixture_root, manifest, "sh_coeffs"),
      load_fixture_buffer<float>(fixture_root, manifest, "rotations"),
      load_fixture_buffer<float>(fixture_root, manifest, "scales_opacs"),
      load_fixture_buffer<std::int32_t>(fixture_root, manifest, "tiles_touched"),
      load_fixture_buffer<RectTileSpace>(fixture_root, manifest, "rect_tile_space"),
      load_fixture_buffer<std::int32_t>(fixture_root, manifest, "radii"),
      load_fixture_buffer<float>(fixture_root, manifest, "xy_vs"),
      load_fixture_buffer<float>(fixture_root, manifest, "depths"),
      load_fixture_buffer<float>(fixture_root, manifest, "inv_cov_vs_opacity"),
      load_fixture_buffer<float>(fixture_root, manifest, "rgb"),
  };
}

void require_valid_tile_ranges(const std::vector<std::int32_t> &tile_ranges,
                               const std::vector<std::uint32_t> &sorted_keys,
                               std::uint32_t grid_width,
                               std::uint32_t grid_height) {
  const auto num_indices = static_cast<std::int32_t>(sorted_keys.size());
  const auto num_tiles = static_cast<std::size_t>(grid_width) * grid_height;

  REQUIRE(tile_ranges.size() == num_tiles + 1U);
  REQUIRE(std::is_sorted(tile_ranges.begin(), tile_ranges.end()));
  REQUIRE(tile_ranges.back() == num_indices);

  for (const auto key : sorted_keys) {
    REQUIRE((key >> kDepthBits) < num_tiles);
  }

  for (std::size_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
    INFO("tile: " << tile_id);
    const auto start = tile_ranges[tile_id];
    const auto end = tile_ranges[tile_id + 1U];
    REQUIRE(start >= 0);
    REQUIRE(end >= start);
    REQUIRE(end <= num_indices);

    for (std::int32_t index = start; index < end; ++index) {
      INFO("range index: " << index);
      REQUIRE((sorted_keys[static_cast<std::size_t>(index)] >> kDepthBits) == tile_id);
    }
  }
}

void require_valid_tile_ranges(const std::vector<std::int32_t> &tile_ranges,
                               const std::vector<std::uint32_t> &sorted_keys) {
  require_valid_tile_ranges(tile_ranges, sorted_keys, kDefaultGridWidth, kDefaultGridHeight);
}

[[nodiscard]] PixelState
pixel_at(const std::vector<float> &pixel_state, std::uint32_t image_width, std::uint32_t x, std::uint32_t y) {
  const auto base = ((static_cast<std::size_t>(y) * image_width) + x) * kPixelChannels;
  REQUIRE(base + 3U < pixel_state.size());
  return {pixel_state[base], pixel_state[base + 1U], pixel_state[base + 2U], pixel_state[base + 3U]};
}

[[nodiscard]] std::int32_t contributor_at(const std::vector<std::int32_t> &n_contributors,
                                          std::uint32_t image_width,
                                          std::uint32_t x,
                                          std::uint32_t y) {
  const auto index = (static_cast<std::size_t>(y) * image_width) + x;
  REQUIRE(index < n_contributors.size());
  return n_contributors[index];
}

} // namespace nlrc::vksplat::tests
