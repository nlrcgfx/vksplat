#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/shader_execution.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "nlrc_vksplat_config.hpp"

using namespace nlrc::vksplat;

namespace {

constexpr std::uint32_t kImageHeight = 32;
constexpr std::uint32_t kImageWidth = 32;
constexpr std::uint32_t kGridHeight = 2;
constexpr std::uint32_t kGridWidth = 2;
constexpr std::uint32_t kDepthBits = 23;

static_assert(VKSPLAT_SORTING_KEY_BITS == 32, "Update generate_keys tests when 64-bit keys are enabled.");

[[nodiscard]] gpu::RendererUniforms make_uniforms(std::uint32_t num_splats) {
  return {
      kImageHeight,
      kImageWidth,
      kGridHeight,
      kGridWidth,
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

void require_valid_generate_keys_output(const std::vector<std::uint32_t> &keys,
                                        const std::vector<std::int32_t> &indices,
                                        std::uint32_t num_splats) {
  REQUIRE(keys.size() == indices.size());
  for (std::size_t index = 0; index < keys.size(); ++index) {
    INFO("output index: " << index);
    const auto tile_id = keys[index] >> kDepthBits;
    REQUIRE(tile_id < kGridWidth * kGridHeight);
    REQUIRE(indices[index] >= 0);
    REQUIRE(indices[index] < static_cast<std::int32_t>(num_splats));
  }
}

} // namespace

TEST_CASE("Dispatch generate_keys shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

#if VKSPLAT_USE_EMULATED_INT64
  constexpr const char *kStageName = "generate_keys_emulated_int64";
#else
  constexpr const char *kStageName = "generate_keys_native_int64";
#endif

  const auto fixture_root = tests::fixture_dir(kStageName);
  const auto golden_root = tests::golden_dir(kStageName);

  const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
  const auto golden_manifest = tests::load_fixture_manifest(golden_root / "manifest.json");

  const std::vector<std::string> expected_bindings = {
      "xy_vs",         "inv_cov_vs_opacity", "depths", "rect_tile_space", "index_buffer_offset",
      "unsorted_keys", "unsorted_gauss_idx",
  };
  REQUIRE(manifest.bindings == expected_bindings);
  REQUIRE(manifest.bindings.size() == gpu::kGenerateKeysBindingCount);
  REQUIRE_FALSE(manifest.profile_agnostic);

  const auto xy_vs = tests::load_fixture_buffer<float>(fixture_root, manifest, "xy_vs");
  const auto inv_cov_vs_opacity = tests::load_fixture_buffer<float>(fixture_root, manifest, "inv_cov_vs_opacity");
  const auto depths = tests::load_fixture_buffer<float>(fixture_root, manifest, "depths");
  const auto rect_tile_space = tests::load_fixture_buffer<RectTileSpace>(fixture_root, manifest, "rect_tile_space");
  const auto tiles_touched = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "tiles_touched");
  const auto index_buffer_offset =
      tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "index_buffer_offset");
  const auto initial_keys = tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "unsorted_keys");
  const auto initial_indices = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "unsorted_gauss_idx");

  REQUIRE(index_buffer_offset.size() == tiles_touched.size());
  REQUIRE_FALSE(index_buffer_offset.empty());
  REQUIRE(index_buffer_offset.back() == static_cast<std::int32_t>(initial_keys.size()));
  REQUIRE(initial_keys.size() == initial_indices.size());
  REQUIRE(xy_vs.size() == tiles_touched.size() * 2U);
  REQUIRE(inv_cov_vs_opacity.size() == tiles_touched.size() * 4U);
  REQUIRE(depths.size() == tiles_touched.size());
  REQUIRE(rect_tile_space.size() == tiles_touched.size() * kRectTileSpaceWords);

  std::int32_t running_total = 0;
  for (std::size_t index = 0; index < tiles_touched.size(); ++index) {
    running_total += tiles_touched[index];
    REQUIRE(index_buffer_offset[index] == running_total);
  }

  const gpu::HeadlessContext context;

  auto xy_vs_buffer = gpu::make_storage_buffer(context, xy_vs);
  auto inv_cov_vs_opacity_buffer = gpu::make_storage_buffer(context, inv_cov_vs_opacity);
  auto depths_buffer = gpu::make_storage_buffer(context, depths);
  auto rect_tile_space_buffer = gpu::make_storage_buffer(context, rect_tile_space);
  auto index_buffer_offset_buffer = gpu::make_storage_buffer(context, index_buffer_offset);
  auto keys_buffer = gpu::make_storage_buffer(context, initial_keys);
  auto indices_buffer = gpu::make_storage_buffer(context, initial_indices);

  gpu::GenerateKeysBindings bindings{};
  bindings.xy_vs = &xy_vs_buffer;
  bindings.inv_cov_vs_opacity = &inv_cov_vs_opacity_buffer;
  bindings.depths = &depths_buffer;
  bindings.rect_tile_space = &rect_tile_space_buffer;
  bindings.index_buffer_offset = &index_buffer_offset_buffer;
  bindings.unsorted_keys = &keys_buffer;
  bindings.unsorted_gauss_idx = &indices_buffer;

  const auto uniforms = make_uniforms(static_cast<std::uint32_t>(tiles_touched.size()));
  gpu::execute_generate_keys(context, bindings, uniforms, initial_keys.size());

  const auto expected_keys = tests::load_fixture_buffer<std::uint32_t>(golden_root, golden_manifest, "unsorted_keys");
  const auto expected_indices =
      tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "unsorted_gauss_idx");

  const auto actual_keys = keys_buffer.read_back<std::uint32_t>(expected_keys.size());
  const auto actual_indices = indices_buffer.read_back<std::int32_t>(expected_indices.size());

  REQUIRE(actual_keys.size() == static_cast<std::size_t>(index_buffer_offset.back()));
  require_valid_generate_keys_output(actual_keys, actual_indices, static_cast<std::uint32_t>(tiles_touched.size()));

  for (std::size_t splat_index = 0; splat_index < tiles_touched.size(); ++splat_index) {
    if (tiles_touched[splat_index] == 0) {
      const auto zero_touch_output = static_cast<std::int32_t>(splat_index);
      REQUIRE(std::none_of(actual_indices.begin(), actual_indices.end(), [zero_touch_output](std::int32_t value) {
        return value == zero_touch_output;
      }));
    }
  }

  REQUIRE(actual_keys == expected_keys);
  REQUIRE(actual_indices == expected_indices);
}

TEST_CASE("Chain projection_forward cumsum generate_keys", "[gpu]") {
  NLRC_REQUIRE_GPU();

#if VKSPLAT_USE_EMULATED_INT64
  constexpr const char *kProjectionStageName = "projection_forward_emulated_int64";
#else
  constexpr const char *kProjectionStageName = "projection_forward_native_int64";
#endif

  const auto fixture_root = tests::fixture_dir(kProjectionStageName);
  const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");

  const auto xyz_ws = tests::load_fixture_buffer<float>(fixture_root, manifest, "xyz_ws");
  const auto sh_coeffs = tests::load_fixture_buffer<float>(fixture_root, manifest, "sh_coeffs");
  const auto rotations = tests::load_fixture_buffer<float>(fixture_root, manifest, "rotations");
  const auto scales_opacs = tests::load_fixture_buffer<float>(fixture_root, manifest, "scales_opacs");
  const auto tiles_touched = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "tiles_touched");
  const auto rect_tile_space = tests::load_fixture_buffer<RectTileSpace>(fixture_root, manifest, "rect_tile_space");
  const auto radii = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "radii");
  const auto xy_vs = tests::load_fixture_buffer<float>(fixture_root, manifest, "xy_vs");
  const auto depths = tests::load_fixture_buffer<float>(fixture_root, manifest, "depths");
  const auto inv_cov_vs_opacity = tests::load_fixture_buffer<float>(fixture_root, manifest, "inv_cov_vs_opacity");
  const auto rgb = tests::load_fixture_buffer<float>(fixture_root, manifest, "rgb");

  REQUIRE(tiles_touched.size() == 1);

  const gpu::HeadlessContext context;

  auto xyz_ws_buffer = gpu::make_storage_buffer(context, xyz_ws);
  auto sh_coeffs_buffer = gpu::make_storage_buffer(context, sh_coeffs);
  auto rotations_buffer = gpu::make_storage_buffer(context, rotations);
  auto scales_opacs_buffer = gpu::make_storage_buffer(context, scales_opacs);
  auto tiles_touched_buffer = gpu::make_storage_buffer(context, tiles_touched);
  auto rect_tile_space_buffer = gpu::make_storage_buffer(context, rect_tile_space);
  auto radii_buffer = gpu::make_storage_buffer(context, radii);
  auto xy_vs_buffer = gpu::make_storage_buffer(context, xy_vs);
  auto depths_buffer = gpu::make_storage_buffer(context, depths);
  auto inv_cov_vs_opacity_buffer = gpu::make_storage_buffer(context, inv_cov_vs_opacity);
  auto rgb_buffer = gpu::make_storage_buffer(context, rgb);

  gpu::ProjectionForwardBindings projection_bindings{};
  projection_bindings.xyz_ws = &xyz_ws_buffer;
  projection_bindings.sh_coeffs = &sh_coeffs_buffer;
  projection_bindings.rotations = &rotations_buffer;
  projection_bindings.scales_opacs = &scales_opacs_buffer;
  projection_bindings.tiles_touched = &tiles_touched_buffer;
  projection_bindings.rect_tile_space = &rect_tile_space_buffer;
  projection_bindings.radii = &radii_buffer;
  projection_bindings.xy_vs = &xy_vs_buffer;
  projection_bindings.depths = &depths_buffer;
  projection_bindings.inv_cov_vs_opacity = &inv_cov_vs_opacity_buffer;
  projection_bindings.rgb = &rgb_buffer;

  const auto uniforms = make_uniforms(1U);
  gpu::execute_projection_forward(context, projection_bindings, uniforms);

  std::vector<std::int32_t> index_buffer_offset(tiles_touched.size(), 0);
  auto index_buffer_offset_buffer = gpu::make_storage_buffer(context, index_buffer_offset);

  // The cumsum helper produces the inclusive offsets consumed by generate_keys.
  gpu::execute_cumsum(context, tiles_touched_buffer, index_buffer_offset_buffer, tiles_touched.size());

  index_buffer_offset = index_buffer_offset_buffer.read_back<std::int32_t>(tiles_touched.size());
  REQUIRE_FALSE(index_buffer_offset.empty());
  const auto num_indices = index_buffer_offset.back();
  REQUIRE(num_indices > 0);

  std::vector<std::uint32_t> initial_keys(static_cast<std::size_t>(num_indices), 0U);
  std::vector<std::int32_t> initial_indices(static_cast<std::size_t>(num_indices), 0);
  auto keys_buffer = gpu::make_storage_buffer(context, initial_keys);
  auto indices_buffer = gpu::make_storage_buffer(context, initial_indices);

  gpu::GenerateKeysBindings generate_keys_bindings{};
  generate_keys_bindings.xy_vs = &xy_vs_buffer;
  generate_keys_bindings.inv_cov_vs_opacity = &inv_cov_vs_opacity_buffer;
  generate_keys_bindings.depths = &depths_buffer;
  generate_keys_bindings.rect_tile_space = &rect_tile_space_buffer;
  generate_keys_bindings.index_buffer_offset = &index_buffer_offset_buffer;
  generate_keys_bindings.unsorted_keys = &keys_buffer;
  generate_keys_bindings.unsorted_gauss_idx = &indices_buffer;

  gpu::execute_generate_keys(context, generate_keys_bindings, uniforms, static_cast<std::size_t>(num_indices));

  const auto actual_keys = keys_buffer.read_back<std::uint32_t>(static_cast<std::size_t>(num_indices));
  const auto actual_indices = indices_buffer.read_back<std::int32_t>(static_cast<std::size_t>(num_indices));

  REQUIRE(actual_keys.size() == static_cast<std::size_t>(num_indices));
  require_valid_generate_keys_output(actual_keys, actual_indices, uniforms.num_splats);
}
