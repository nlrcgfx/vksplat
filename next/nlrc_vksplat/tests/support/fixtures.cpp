#include "fixtures.hpp"

#include <algorithm>
#include <cstddef>
#include <utility>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "golden_compare.hpp"
#include "span.hpp"

namespace nlrc::vksplat::tests {

namespace {

constexpr std::uint32_t kDepthBits = 23;

} // namespace

ProjectionFixtureBuffers::ProjectionFixtureBuffers(gpu::StorageBuffer xyz_ws_buffer,
                                                   gpu::StorageBuffer sh_coeffs_buffer,
                                                   gpu::StorageBuffer rotations_buffer,
                                                   gpu::StorageBuffer scales_opacs_buffer,
                                                   gpu::StorageBuffer tiles_touched_buffer,
                                                   gpu::StorageBuffer rect_tile_space_buffer,
                                                   gpu::StorageBuffer radii_buffer,
                                                   gpu::StorageBuffer xy_vs_buffer,
                                                   gpu::StorageBuffer depths_buffer,
                                                   gpu::StorageBuffer inv_cov_vs_opacity_buffer,
                                                   gpu::StorageBuffer rgb_buffer)
  : xyz_ws(std::move(xyz_ws_buffer)), sh_coeffs(std::move(sh_coeffs_buffer)), rotations(std::move(rotations_buffer)),
    scales_opacs(std::move(scales_opacs_buffer)), tiles_touched(std::move(tiles_touched_buffer)),
    rect_tile_space(std::move(rect_tile_space_buffer)), radii(std::move(radii_buffer)), xy_vs(std::move(xy_vs_buffer)),
    depths(std::move(depths_buffer)), inv_cov_vs_opacity(std::move(inv_cov_vs_opacity_buffer)),
    rgb(std::move(rgb_buffer)) {
  refresh_bindings();
}

ProjectionFixtureBuffers::ProjectionFixtureBuffers(ProjectionFixtureBuffers &&other) noexcept
  : xyz_ws(std::move(other.xyz_ws)), sh_coeffs(std::move(other.sh_coeffs)), rotations(std::move(other.rotations)),
    scales_opacs(std::move(other.scales_opacs)), tiles_touched(std::move(other.tiles_touched)),
    rect_tile_space(std::move(other.rect_tile_space)), radii(std::move(other.radii)), xy_vs(std::move(other.xy_vs)),
    depths(std::move(other.depths)), inv_cov_vs_opacity(std::move(other.inv_cov_vs_opacity)),
    rgb(std::move(other.rgb)) {
  refresh_bindings();
}

ProjectionFixtureBuffers &ProjectionFixtureBuffers::operator=(ProjectionFixtureBuffers &&other) noexcept {
  if (this != &other) {
    xyz_ws = std::move(other.xyz_ws);
    sh_coeffs = std::move(other.sh_coeffs);
    rotations = std::move(other.rotations);
    scales_opacs = std::move(other.scales_opacs);
    tiles_touched = std::move(other.tiles_touched);
    rect_tile_space = std::move(other.rect_tile_space);
    radii = std::move(other.radii);
    xy_vs = std::move(other.xy_vs);
    depths = std::move(other.depths);
    inv_cov_vs_opacity = std::move(other.inv_cov_vs_opacity);
    rgb = std::move(other.rgb);
    refresh_bindings();
  }
  return *this;
}

void ProjectionFixtureBuffers::refresh_bindings() noexcept {
  bindings.xyz_ws = &xyz_ws;
  bindings.sh_coeffs = &sh_coeffs;
  bindings.rotations = &rotations;
  bindings.scales_opacs = &scales_opacs;
  bindings.tiles_touched = &tiles_touched;
  bindings.rect_tile_space = &rect_tile_space;
  bindings.radii = &radii;
  bindings.xy_vs = &xy_vs;
  bindings.depths = &depths;
  bindings.inv_cov_vs_opacity = &inv_cov_vs_opacity;
  bindings.rgb = &rgb;
}

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

[[nodiscard]] ProjectionFixtureBuffers upload_projection_fixture(const gpu::HeadlessContext &context,
                                                                 const ProjectionFixtureData &fixture) {
  return {
      gpu::make_storage_buffer(context, fixture.xyz_ws),
      gpu::make_storage_buffer(context, fixture.sh_coeffs),
      gpu::make_storage_buffer(context, fixture.rotations),
      gpu::make_storage_buffer(context, fixture.scales_opacs),
      gpu::make_storage_buffer(context, fixture.tiles_touched),
      gpu::make_storage_buffer(context, fixture.rect_tile_space),
      gpu::make_storage_buffer(context, fixture.radii),
      gpu::make_storage_buffer(context, fixture.xy_vs),
      gpu::make_storage_buffer(context, fixture.depths),
      gpu::make_storage_buffer(context, fixture.inv_cov_vs_opacity),
      gpu::make_storage_buffer(context, fixture.rgb),
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

void require_pixel_invariants(const std::vector<float> &pixel_state,
                              const std::vector<std::int32_t> &n_contributors,
                              std::uint32_t image_width,
                              std::uint32_t image_height,
                              std::size_t num_indices) {
  REQUIRE_NOTHROW(assert_no_nan_inf(make_span(pixel_state)));

  const auto num_pixels = static_cast<std::size_t>(image_width) * image_height;
  REQUIRE(pixel_state.size() == num_pixels * kPixelChannels);
  REQUIRE(n_contributors.size() == num_pixels);

  for (std::size_t index = 3; index < pixel_state.size(); index += kPixelChannels) {
    INFO("pixel_state index: " << index);
    REQUIRE(pixel_state[index] >= 0.0F);
    REQUIRE(pixel_state[index] <= 1.0F);
  }

  for (const auto contributors : n_contributors) {
    REQUIRE(contributors >= 0);
    REQUIRE(contributors <= static_cast<std::int32_t>(num_indices));
  }
}

void require_pixel_invariants(const std::vector<float> &pixel_state,
                              const std::vector<std::int32_t> &n_contributors,
                              std::size_t num_indices) {
  require_pixel_invariants(pixel_state, n_contributors, kDefaultImageWidth, kDefaultImageHeight, num_indices);
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
