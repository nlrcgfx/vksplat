#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "fixtures.hpp"
#include "golden_compare.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/shader_execution.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "nlrc_vksplat_config.hpp"
#include "span.hpp"

using namespace nlrc::vksplat;

namespace {

struct RectBounds final {
  std::uint32_t min_x;
  std::uint32_t min_y;
  std::uint32_t max_x;
  std::uint32_t max_y;
};

[[nodiscard]] RectBounds decode_rect_tile_space(const std::vector<RectTileSpace> &rect_words) {
  if constexpr (kUseEmulatedInt64) {
    REQUIRE(rect_words.size() == 2);
    const auto min_word = static_cast<std::uint32_t>(rect_words[0]);
    const auto max_word = static_cast<std::uint32_t>(rect_words[1]);
    return {
        min_word & 0xFFFFU,
        (min_word >> 16U) & 0xFFFFU,
        max_word & 0xFFFFU,
        (max_word >> 16U) & 0xFFFFU,
    };
  } else {
    REQUIRE(rect_words.size() == 1);
    const auto packed = static_cast<std::uint64_t>(rect_words[0]);
    return {
        static_cast<std::uint32_t>(packed & 0xFFFFULL),
        static_cast<std::uint32_t>((packed >> 16U) & 0xFFFFULL),
        static_cast<std::uint32_t>((packed >> 32U) & 0xFFFFULL),
        static_cast<std::uint32_t>((packed >> 48U) & 0xFFFFULL),
    };
  }
}

} // namespace

TEST_CASE("Dispatch projection_forward shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const auto stage_name = tests::int64_profile_stage_name("projection_forward");
  const auto fixture_root = tests::fixture_dir(stage_name);
  const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
  const auto fixture = tests::load_projection_fixture(stage_name);

  REQUIRE_FALSE(manifest.profile_agnostic);

  REQUIRE(fixture.sh_coeffs.size() == static_cast<std::size_t>(12U * VKSPLAT_SH_REORDER_SIZE * 4U));
  REQUIRE(fixture.tiles_touched.size() == 1);
  REQUIRE(fixture.radii.size() == 1);
  REQUIRE(fixture.xy_vs.size() == 2);
  REQUIRE(fixture.depths.size() == 1);
  REQUIRE(fixture.inv_cov_vs_opacity.size() == 4);
  REQUIRE(fixture.rgb.size() == 3);

  const gpu::HeadlessContext context;

  auto xyz_ws_buffer = gpu::make_storage_buffer(context, fixture.xyz_ws);
  auto sh_coeffs_buffer = gpu::make_storage_buffer(context, fixture.sh_coeffs);
  auto rotations_buffer = gpu::make_storage_buffer(context, fixture.rotations);
  auto scales_opacs_buffer = gpu::make_storage_buffer(context, fixture.scales_opacs);
  auto tiles_touched_buffer = gpu::make_storage_buffer(context, fixture.tiles_touched);
  auto rect_tile_space_buffer = gpu::make_storage_buffer(context, fixture.rect_tile_space);
  auto radii_buffer = gpu::make_storage_buffer(context, fixture.radii);
  auto xy_vs_buffer = gpu::make_storage_buffer(context, fixture.xy_vs);
  auto depths_buffer = gpu::make_storage_buffer(context, fixture.depths);
  auto inv_cov_vs_opacity_buffer = gpu::make_storage_buffer(context, fixture.inv_cov_vs_opacity);
  auto rgb_buffer = gpu::make_storage_buffer(context, fixture.rgb);

  gpu::ProjectionForwardBindings bindings{};
  bindings.xyz_ws = &xyz_ws_buffer;
  bindings.sh_coeffs = &sh_coeffs_buffer;
  bindings.rotations = &rotations_buffer;
  bindings.scales_opacs = &scales_opacs_buffer;
  bindings.tiles_touched = &tiles_touched_buffer;
  bindings.rect_tile_space = &rect_tile_space_buffer;
  bindings.radii = &radii_buffer;
  bindings.xy_vs = &xy_vs_buffer;
  bindings.depths = &depths_buffer;
  bindings.inv_cov_vs_opacity = &inv_cov_vs_opacity_buffer;
  bindings.rgb = &rgb_buffer;

  const auto uniforms = tests::default_renderer_uniforms(1U);
  gpu::execute_projection_forward(context, bindings, uniforms);

  const auto actual_tiles_touched = tiles_touched_buffer.read_back<std::int32_t>(1);
  const auto actual_rect_tile_space = rect_tile_space_buffer.read_back<RectTileSpace>(fixture.rect_tile_space.size());
  const auto actual_radii = radii_buffer.read_back<std::int32_t>(1);
  const auto actual_xy_vs = xy_vs_buffer.read_back<float>(2);
  const auto actual_depths = depths_buffer.read_back<float>(1);
  const auto actual_inv_cov_vs_opacity = inv_cov_vs_opacity_buffer.read_back<float>(4);
  const auto actual_rgb = rgb_buffer.read_back<float>(3);

  REQUIRE_NOTHROW(tests::assert_no_nan_inf(make_span(actual_xy_vs)));
  REQUIRE_NOTHROW(tests::assert_no_nan_inf(make_span(actual_depths)));
  REQUIRE_NOTHROW(tests::assert_no_nan_inf(make_span(actual_inv_cov_vs_opacity)));
  REQUIRE_NOTHROW(tests::assert_no_nan_inf(make_span(actual_rgb)));

  REQUIRE(actual_tiles_touched[0] > 0);
  REQUIRE(actual_tiles_touched[0] <= static_cast<std::int32_t>(uniforms.grid_width * uniforms.grid_height));
  REQUIRE(actual_radii[0] > 0);

  REQUIRE(actual_xy_vs[0] >= 0.0F);
  REQUIRE(actual_xy_vs[0] < static_cast<float>(uniforms.image_width));

  REQUIRE(actual_xy_vs[1] >= 0.0F);
  REQUIRE(actual_xy_vs[1] < static_cast<float>(uniforms.image_height));

  REQUIRE(actual_depths[0] > 0.0F);
  REQUIRE(actual_inv_cov_vs_opacity[3] >= 0.0F);
  REQUIRE(actual_inv_cov_vs_opacity[3] <= 1.0F);

  for (const float channel : actual_rgb) {
    REQUIRE(channel >= 0.0F);
    REQUIRE(channel <= 1.0F);
  }

  const RectBounds rect = decode_rect_tile_space(actual_rect_tile_space);
  REQUIRE(rect.min_x <= rect.max_x);
  REQUIRE(rect.min_y <= rect.max_y);
  REQUIRE(rect.max_x <= uniforms.grid_width);
  REQUIRE(rect.max_y <= uniforms.grid_height);
  REQUIRE(rect.min_x < rect.max_x);
  REQUIRE(rect.min_y < rect.max_y);

  const auto rect_area = static_cast<std::int32_t>((rect.max_x - rect.min_x) * (rect.max_y - rect.min_y));
  REQUIRE(actual_tiles_touched[0] <= rect_area);
}
