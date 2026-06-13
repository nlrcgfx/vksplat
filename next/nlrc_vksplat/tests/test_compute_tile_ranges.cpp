#include <algorithm>
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
constexpr std::uint32_t kImageWidth = 48;
constexpr std::uint32_t kGridHeight = 2;
constexpr std::uint32_t kGridWidth = 3;
constexpr std::uint32_t kDepthBits = 23;

static_assert(VKSPLAT_SORTING_KEY_BITS == 32, "Update compute_tile_ranges tests when 64-bit keys are enabled.");

[[nodiscard]] gpu::RendererUniforms make_uniforms() {
  return {
      kImageHeight,
      kImageWidth,
      kGridHeight,
      kGridWidth,
      0U,
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

void require_valid_tile_ranges(const std::vector<std::int32_t> &tile_ranges,
                               const std::vector<std::uint32_t> &sorted_keys) {
  const auto num_indices = static_cast<std::int32_t>(sorted_keys.size());
  const auto num_tiles = static_cast<std::size_t>(kGridHeight) * kGridWidth;

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

} // namespace

TEST_CASE("Dispatch compute_tile_ranges shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  constexpr const char *kStageName = "compute_tile_ranges";
  const auto fixture_root = tests::fixture_dir(kStageName);
  const auto golden_root = tests::golden_dir(kStageName);

  const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
  const auto golden_manifest = tests::load_fixture_manifest(golden_root / "manifest.json");

  const std::vector<std::string> expected_bindings = {
      "sorted_keys",
      "tile_ranges",
  };
  REQUIRE(manifest.bindings == expected_bindings);
  REQUIRE(manifest.bindings.size() == gpu::kComputeTileRangesBindingCount);
  REQUIRE(manifest.profile_agnostic);

  const auto sorted_keys = tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorted_keys");
  const auto initial_tile_ranges = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "tile_ranges");
  const auto expected_tile_ranges =
      tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "tile_ranges");

  REQUIRE_FALSE(sorted_keys.empty());
  REQUIRE(initial_tile_ranges.size() == static_cast<std::size_t>(kGridHeight * kGridWidth) + 1U);
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
  const auto uniforms = make_uniforms();
  gpu::execute_compute_tile_ranges(context, bindings, uniforms, sorted_keys.size());

  const auto actual_tile_ranges = tile_ranges_buffer.read_back<std::int32_t>(expected_tile_ranges.size());

  require_valid_tile_ranges(actual_tile_ranges, sorted_keys);
  REQUIRE(actual_tile_ranges == expected_tile_ranges);
}
