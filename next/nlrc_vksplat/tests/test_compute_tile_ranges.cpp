#include <algorithm>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "fixtures.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/shader_execution.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "nlrc_vksplat_config.hpp"

using namespace nlrc::vksplat;

namespace {

constexpr std::uint32_t kImageWidth = 48;
constexpr std::uint32_t kGridWidth = 3;

static_assert(VKSPLAT_SORTING_KEY_BITS == 32, "Update compute_tile_ranges tests when 64-bit keys are enabled.");

} // namespace

TEST_CASE("Dispatch compute_tile_ranges shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  constexpr const char *kStageName = "compute_tile_ranges";
  const auto fixture_root = tests::fixture_dir(kStageName);
  const auto golden_root = tests::golden_dir(kStageName);

  const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
  const auto golden_manifest = tests::load_fixture_manifest(golden_root / "manifest.json");

  const auto expected_bindings = gpu::binding_names(gpu::shader_interface("compute_tile_ranges"));
  REQUIRE(manifest.bindings == expected_bindings);
  REQUIRE(manifest.bindings.size() == gpu::kComputeTileRangesBindingCount);
  REQUIRE(manifest.profile_agnostic);

  const auto sorted_keys = tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorted_keys");
  const auto initial_tile_ranges = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "tile_ranges");
  const auto expected_tile_ranges =
      tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "tile_ranges");

  REQUIRE_FALSE(sorted_keys.empty());
  REQUIRE(initial_tile_ranges.size() == static_cast<std::size_t>(tests::kDefaultGridHeight * kGridWidth) + 1U);
  REQUIRE(expected_tile_ranges.size() == initial_tile_ranges.size());
  REQUIRE(std::all_of(initial_tile_ranges.begin(), initial_tile_ranges.end(), [](std::int32_t value) {
    return value == -1;
  }));

  const gpu::HeadlessContext context;

  auto sorted_keys_buffer = gpu::make_storage_buffer(context, sorted_keys);
  auto tile_ranges_buffer = gpu::make_storage_buffer(context, initial_tile_ranges);

  gpu::ComputeTileRangesBindings bindings{};
  bindings.sorted_keys = &sorted_keys_buffer;
  bindings.tile_ranges = &tile_ranges_buffer;

  // active_sh is intentionally zero; execute_compute_tile_ranges must alias it to num_indices.
  auto uniforms = tests::default_renderer_uniforms(0U);
  uniforms.image_height = tests::kDefaultImageHeight;
  uniforms.image_width = kImageWidth;
  uniforms.grid_height = tests::kDefaultGridHeight;
  uniforms.grid_width = kGridWidth;
  gpu::execute_compute_tile_ranges(context, bindings, uniforms, sorted_keys.size());

  const auto actual_tile_ranges = tile_ranges_buffer.read_back<std::int32_t>(expected_tile_ranges.size());

  tests::require_valid_tile_ranges(actual_tile_ranges, sorted_keys, kGridWidth, tests::kDefaultGridHeight);
  REQUIRE(actual_tile_ranges == expected_tile_ranges);
}
