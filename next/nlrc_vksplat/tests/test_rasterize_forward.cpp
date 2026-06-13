#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "golden_compare.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/shader_execution.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "nlrc_vksplat_config.hpp"
#include "span.hpp"

using namespace nlrc::vksplat;

namespace {

constexpr std::uint32_t kDepthBits = 23;
constexpr std::size_t kPixelChannels = 4;

struct ImageGrid final {
  std::uint32_t image_height;
  std::uint32_t image_width;
  std::uint32_t grid_height;
  std::uint32_t grid_width;
};

struct PixelState final {
  float r;
  float g;
  float b;
  float transmittance;
};

[[nodiscard]] ImageGrid image_grid_from_manifest(const tests::FixtureManifest &manifest) {
  const auto &pixel_state_spec = tests::buffer_spec(manifest, "pixel_state");
  REQUIRE(pixel_state_spec.shape.size() == 3U);
  REQUIRE(pixel_state_spec.shape[2] == kPixelChannels);
  REQUIRE(pixel_state_spec.shape[0] % VKSPLAT_TILE_HEIGHT == 0U);
  REQUIRE(pixel_state_spec.shape[1] % VKSPLAT_TILE_WIDTH == 0U);

  return {
      static_cast<std::uint32_t>(pixel_state_spec.shape[0]),
      static_cast<std::uint32_t>(pixel_state_spec.shape[1]),
      static_cast<std::uint32_t>(pixel_state_spec.shape[0] / VKSPLAT_TILE_HEIGHT),
      static_cast<std::uint32_t>(pixel_state_spec.shape[1] / VKSPLAT_TILE_WIDTH),
  };
}

[[nodiscard]] gpu::RendererUniforms make_uniforms(const ImageGrid &image_grid, std::uint32_t num_splats) {
  return {
      image_grid.image_height,
      image_grid.image_width,
      image_grid.grid_height,
      image_grid.grid_width,
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

void require_valid_sorted_ranges(const std::vector<std::uint32_t> &sorted_keys,
                                 const std::vector<std::int32_t> &sorted_gauss_idx,
                                 const std::vector<std::int32_t> &tile_ranges,
                                 const ImageGrid &image_grid,
                                 std::size_t num_splats) {
  REQUIRE(sorted_keys.size() == sorted_gauss_idx.size());
  REQUIRE(std::is_sorted(sorted_keys.begin(), sorted_keys.end()));

  const auto num_indices = static_cast<std::int32_t>(sorted_keys.size());
  const auto num_tiles = static_cast<std::size_t>(image_grid.grid_width) * image_grid.grid_height;
  REQUIRE(tile_ranges.size() == num_tiles + 1U);
  REQUIRE(std::is_sorted(tile_ranges.begin(), tile_ranges.end()));
  REQUIRE(tile_ranges.back() == num_indices);

  for (const auto splat_index : sorted_gauss_idx) {
    REQUIRE(splat_index >= 0);
    REQUIRE(static_cast<std::size_t>(splat_index) < num_splats);
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

void require_output_invariants(const std::vector<float> &pixel_state,
                               const std::vector<std::int32_t> &n_contributors,
                               std::size_t num_indices) {
  REQUIRE_NOTHROW(tests::assert_no_nan_inf(make_span(pixel_state)));

  REQUIRE(pixel_state.size() % kPixelChannels == 0U);
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

void require_empty_tile_centers_are_baseline(const std::vector<float> &pixel_state,
                                             const std::vector<std::int32_t> &n_contributors,
                                             const std::vector<std::int32_t> &tile_ranges,
                                             const ImageGrid &image_grid) {
  const auto num_tiles = static_cast<std::size_t>(image_grid.grid_width) * image_grid.grid_height;
  for (std::size_t tile_id = 0; tile_id < num_tiles; ++tile_id) {
    if (tile_ranges[tile_id] != tile_ranges[tile_id + 1U]) {
      continue;
    }

    INFO("empty tile: " << tile_id);
    const auto tile_x = static_cast<std::uint32_t>(tile_id % image_grid.grid_width);
    const auto tile_y = static_cast<std::uint32_t>(tile_id / image_grid.grid_width);
    const auto pixel_x = (tile_x * VKSPLAT_TILE_WIDTH) + (VKSPLAT_TILE_WIDTH / 2U);
    const auto pixel_y = (tile_y * VKSPLAT_TILE_HEIGHT) + (VKSPLAT_TILE_HEIGHT / 2U);

    const auto pixel = pixel_at(pixel_state, image_grid.image_width, pixel_x, pixel_y);
    REQUIRE(pixel.r == 0.0F);
    REQUIRE(pixel.g == 0.0F);
    REQUIRE(pixel.b == 0.0F);
    REQUIRE(pixel.transmittance == 1.0F);
    REQUIRE(contributor_at(n_contributors, image_grid.image_width, pixel_x, pixel_y) == 0);
  }
}

void require_splat_centers_have_contribution(const std::vector<float> &pixel_state,
                                             const std::vector<std::int32_t> &n_contributors,
                                             const std::vector<float> &xy_vs,
                                             const std::vector<std::int32_t> &sorted_gauss_idx,
                                             const ImageGrid &image_grid) {
  for (const auto splat_index : sorted_gauss_idx) {
    INFO("splat index: " << splat_index);
    const auto xy_base = static_cast<std::size_t>(splat_index) * 2U;
    REQUIRE(xy_base + 1U < xy_vs.size());

    const auto pixel_x = static_cast<std::uint32_t>(xy_vs[xy_base]);
    const auto pixel_y = static_cast<std::uint32_t>(xy_vs[xy_base + 1U]);
    REQUIRE(pixel_x < image_grid.image_width);
    REQUIRE(pixel_y < image_grid.image_height);

    const auto pixel = pixel_at(pixel_state, image_grid.image_width, pixel_x, pixel_y);
    REQUIRE(contributor_at(n_contributors, image_grid.image_width, pixel_x, pixel_y) > 0);
    REQUIRE(pixel.transmittance < 1.0F);
    REQUIRE((pixel.r > 0.0F || pixel.g > 0.0F || pixel.b > 0.0F));
  }
}

} // namespace

TEST_CASE("Dispatch rasterize_forward shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  constexpr std::array kStageNames = {
      "rasterize_forward_single_splat",
      "rasterize_forward_multi_tile",
  };

  for (const auto *stage_name : kStageNames) {
    SECTION(stage_name) {
      const auto fixture_root = tests::fixture_dir(stage_name);
      const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");

      const std::vector<std::string> expected_bindings = {
          "sorted_gauss_idx", "tile_ranges", "xy_vs", "inv_cov_vs_opacity", "rgb", "pixel_state", "n_contributors",
      };
      REQUIRE(manifest.bindings == expected_bindings);
      REQUIRE(manifest.bindings.size() == gpu::kRasterizeForwardBindingCount);
      REQUIRE(manifest.profile_agnostic);

      const auto sorted_keys = tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorted_keys");
      const auto sorted_gauss_idx =
          tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "sorted_gauss_idx");
      const auto tile_ranges = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "tile_ranges");
      const auto xy_vs = tests::load_fixture_buffer<float>(fixture_root, manifest, "xy_vs");
      const auto inv_cov_vs_opacity = tests::load_fixture_buffer<float>(fixture_root, manifest, "inv_cov_vs_opacity");
      const auto rgb = tests::load_fixture_buffer<float>(fixture_root, manifest, "rgb");
      const auto initial_pixel_state = tests::load_fixture_buffer<float>(fixture_root, manifest, "pixel_state");
      const auto initial_n_contributors =
          tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "n_contributors");

      const auto image_grid = image_grid_from_manifest(manifest);
      const auto num_splats = xy_vs.size() / 2U;
      const auto num_pixels = static_cast<std::size_t>(image_grid.image_width) * image_grid.image_height;

      REQUIRE_FALSE(sorted_keys.empty());
      REQUIRE(sorted_keys.size() == sorted_gauss_idx.size());
      REQUIRE(xy_vs.size() == num_splats * 2U);
      REQUIRE(inv_cov_vs_opacity.size() == num_splats * 4U);
      REQUIRE(rgb.size() == num_splats * 3U);
      REQUIRE(initial_pixel_state.size() == num_pixels * kPixelChannels);
      REQUIRE(initial_n_contributors.size() == num_pixels);
      REQUIRE(std::all_of(initial_pixel_state.begin(), initial_pixel_state.end(), [](float value) {
        return value == -1.0F;
      }));
      REQUIRE(std::all_of(initial_n_contributors.begin(), initial_n_contributors.end(), [](std::int32_t value) {
        return value == -1;
      }));

      require_valid_sorted_ranges(sorted_keys, sorted_gauss_idx, tile_ranges, image_grid, num_splats);

      const gpu::HeadlessContext context;

      auto sorted_gauss_idx_buffer = gpu::make_storage_buffer(context, sorted_gauss_idx);
      auto tile_ranges_buffer = gpu::make_storage_buffer(context, tile_ranges);
      auto xy_vs_buffer = gpu::make_storage_buffer(context, xy_vs);
      auto inv_cov_vs_opacity_buffer = gpu::make_storage_buffer(context, inv_cov_vs_opacity);
      auto rgb_buffer = gpu::make_storage_buffer(context, rgb);
      auto pixel_state_buffer = gpu::make_storage_buffer(context, initial_pixel_state);
      auto n_contributors_buffer = gpu::make_storage_buffer(context, initial_n_contributors);

      gpu::RasterizeForwardBindings bindings{};
      bindings.sorted_gauss_idx = &sorted_gauss_idx_buffer;
      bindings.tile_ranges = &tile_ranges_buffer;
      bindings.xy_vs = &xy_vs_buffer;
      bindings.inv_cov_vs_opacity = &inv_cov_vs_opacity_buffer;
      bindings.rgb = &rgb_buffer;
      bindings.pixel_state = &pixel_state_buffer;
      bindings.n_contributors = &n_contributors_buffer;

      const auto uniforms = make_uniforms(image_grid, static_cast<std::uint32_t>(num_splats));
      gpu::execute_rasterize_forward(context, bindings, uniforms, sorted_gauss_idx.size());

      const auto actual_pixel_state = pixel_state_buffer.read_back<float>(initial_pixel_state.size());
      const auto actual_n_contributors = n_contributors_buffer.read_back<std::int32_t>(initial_n_contributors.size());

      require_output_invariants(actual_pixel_state, actual_n_contributors, sorted_gauss_idx.size());
      require_empty_tile_centers_are_baseline(actual_pixel_state, actual_n_contributors, tile_ranges, image_grid);
      require_splat_centers_have_contribution(actual_pixel_state, actual_n_contributors, xy_vs, sorted_gauss_idx,
                                              image_grid);
    }
  }
}
