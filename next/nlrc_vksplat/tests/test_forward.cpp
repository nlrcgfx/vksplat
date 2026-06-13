#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixtures.hpp"
#include "golden_compare.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/shader_execution.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "span.hpp"

using namespace nlrc::vksplat;

namespace {

template <typename T>
void require_all_equal(const std::vector<T> &values, T sentinel) {
  REQUIRE(std::all_of(values.begin(), values.end(), [sentinel](T value) {
    return value == sentinel;
  }));
}

} // namespace

TEST_CASE("Dispatch forward pipeline", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const auto visible_stage_name = tests::int64_profile_stage_name("projection_forward");
  const auto no_visible_stage_name = tests::int64_profile_stage_name("projection_forward_no_visible");

  SECTION("visible single splat") {
    const auto fixture = tests::load_projection_fixture(visible_stage_name);
    const auto num_splats = fixture.tiles_touched.size();
    const auto num_tiles = static_cast<std::size_t>(tests::kDefaultGridHeight) * tests::kDefaultGridWidth;
    const auto max_indices = num_splats * num_tiles;
    const auto num_pixels = static_cast<std::size_t>(tests::kDefaultImageHeight) * tests::kDefaultImageWidth;

    const gpu::HeadlessContext context;

    auto projection_buffers = tests::upload_projection_fixture(context, fixture);

    std::vector<std::int32_t> index_buffer_offset(num_splats, -1);
    std::vector<std::uint32_t> sorting_keys_1(max_indices, 0xFFFFFFFFU);
    std::vector<std::int32_t> sorting_gauss_idx_1(max_indices, -1);
    std::vector<std::uint32_t> sorting_keys_2(max_indices, 0xEEEEEEEEU);
    std::vector<std::int32_t> sorting_gauss_idx_2(max_indices, -1);
    std::vector<std::int32_t> tile_ranges(num_tiles + 1U, -1);
    std::vector<float> pixel_state(num_pixels * tests::kPixelChannels, -1.0F);
    std::vector<std::int32_t> n_contributors(num_pixels, -1);

    auto index_buffer_offset_buffer = gpu::make_storage_buffer(context, index_buffer_offset);
    auto sorting_keys_1_buffer = gpu::make_storage_buffer(context, sorting_keys_1);
    auto sorting_gauss_idx_1_buffer = gpu::make_storage_buffer(context, sorting_gauss_idx_1);
    auto sorting_keys_2_buffer = gpu::make_storage_buffer(context, sorting_keys_2);
    auto sorting_gauss_idx_2_buffer = gpu::make_storage_buffer(context, sorting_gauss_idx_2);
    auto tile_ranges_buffer = gpu::make_storage_buffer(context, tile_ranges);
    auto pixel_state_buffer = gpu::make_storage_buffer(context, pixel_state);
    auto n_contributors_buffer = gpu::make_storage_buffer(context, n_contributors);

    gpu::ForwardBindings bindings{};
    bindings.projection = projection_buffers.bindings;
    bindings.index_buffer_offset = &index_buffer_offset_buffer;
    bindings.sorting_keys_1 = &sorting_keys_1_buffer;
    bindings.sorting_gauss_idx_1 = &sorting_gauss_idx_1_buffer;
    bindings.sorting_keys_2 = &sorting_keys_2_buffer;
    bindings.sorting_gauss_idx_2 = &sorting_gauss_idx_2_buffer;
    bindings.tile_ranges = &tile_ranges_buffer;
    bindings.pixel_state = &pixel_state_buffer;
    bindings.n_contributors = &n_contributors_buffer;

    const auto result = gpu::execute_forward(context, bindings,
                                             tests::default_renderer_uniforms(static_cast<std::uint32_t>(num_splats)));

    REQUIRE(result.num_indices > 0U);
    REQUIRE(result.rasterized);
    REQUIRE(result.sorted_keys != nullptr);
    REQUIRE(result.sorted_gauss_idx != nullptr);

    const auto actual_index_buffer_offset = index_buffer_offset_buffer.read_back<std::int32_t>(num_splats);
    REQUIRE(actual_index_buffer_offset.back() == static_cast<std::int32_t>(result.num_indices));

    const auto sorted_keys = result.sorted_keys->read_back<std::uint32_t>(result.num_indices);
    const auto sorted_indices = result.sorted_gauss_idx->read_back<std::int32_t>(result.num_indices);
    REQUIRE(std::is_sorted(sorted_keys.begin(), sorted_keys.end()));
    for (const auto sorted_index : sorted_indices) {
      REQUIRE(sorted_index >= 0);
      REQUIRE(static_cast<std::size_t>(sorted_index) < num_splats);
    }

    const auto actual_tile_ranges = tile_ranges_buffer.read_back<std::int32_t>(num_tiles + 1U);
    tests::require_valid_tile_ranges(actual_tile_ranges, sorted_keys);

    const auto actual_pixel_state = pixel_state_buffer.read_back<float>(num_pixels * tests::kPixelChannels);
    const auto actual_n_contributors = n_contributors_buffer.read_back<std::int32_t>(num_pixels);
    tests::require_pixel_invariants(actual_pixel_state, actual_n_contributors, result.num_indices);

    const auto actual_xy_vs = projection_buffers.xy_vs.read_back<float>(num_splats * 2U);
    REQUIRE_NOTHROW(tests::assert_no_nan_inf(make_span(actual_xy_vs)));
    REQUIRE(actual_xy_vs[0] >= 0.0F);
    REQUIRE(actual_xy_vs[0] < static_cast<float>(tests::kDefaultImageWidth));
    REQUIRE(actual_xy_vs[1] >= 0.0F);
    REQUIRE(actual_xy_vs[1] < static_cast<float>(tests::kDefaultImageHeight));

    const auto center_x = static_cast<std::uint32_t>(std::lround(actual_xy_vs[0]));
    const auto center_y = static_cast<std::uint32_t>(std::lround(actual_xy_vs[1]));
    REQUIRE(center_x < tests::kDefaultImageWidth);
    REQUIRE(center_y < tests::kDefaultImageHeight);

    const auto center_pixel = tests::pixel_at(actual_pixel_state, tests::kDefaultImageWidth, center_x, center_y);
    REQUIRE(tests::contributor_at(actual_n_contributors, tests::kDefaultImageWidth, center_x, center_y) > 0);
    REQUIRE(center_pixel.transmittance < 1.0F);
    REQUIRE((center_pixel.r > 0.0F || center_pixel.g > 0.0F || center_pixel.b > 0.0F));
  }

  SECTION("no visible splats early-exits before tile processing and rasterization") {
    const auto fixture = tests::load_projection_fixture(no_visible_stage_name);
    const auto num_splats = fixture.tiles_touched.size();
    const auto num_tiles = static_cast<std::size_t>(tests::kDefaultGridHeight) * tests::kDefaultGridWidth;
    const auto max_indices = num_splats * num_tiles;
    const auto num_pixels = static_cast<std::size_t>(tests::kDefaultImageHeight) * tests::kDefaultImageWidth;

    const gpu::HeadlessContext context;

    auto projection_buffers = tests::upload_projection_fixture(context, fixture);

    std::vector<std::int32_t> index_buffer_offset(num_splats, -1);
    std::vector<std::uint32_t> sorting_keys_1(max_indices, 0xFFFFFFFFU);
    std::vector<std::int32_t> sorting_gauss_idx_1(max_indices, -1);
    std::vector<std::uint32_t> sorting_keys_2(max_indices, 0xEEEEEEEEU);
    std::vector<std::int32_t> sorting_gauss_idx_2(max_indices, -1);
    std::vector<std::int32_t> tile_ranges(num_tiles + 1U, -1);
    std::vector<float> pixel_state(num_pixels * tests::kPixelChannels, -1.0F);
    std::vector<std::int32_t> n_contributors(num_pixels, -1);

    auto index_buffer_offset_buffer = gpu::make_storage_buffer(context, index_buffer_offset);
    auto sorting_keys_1_buffer = gpu::make_storage_buffer(context, sorting_keys_1);
    auto sorting_gauss_idx_1_buffer = gpu::make_storage_buffer(context, sorting_gauss_idx_1);
    auto sorting_keys_2_buffer = gpu::make_storage_buffer(context, sorting_keys_2);
    auto sorting_gauss_idx_2_buffer = gpu::make_storage_buffer(context, sorting_gauss_idx_2);
    auto tile_ranges_buffer = gpu::make_storage_buffer(context, tile_ranges);
    auto pixel_state_buffer = gpu::make_storage_buffer(context, pixel_state);
    auto n_contributors_buffer = gpu::make_storage_buffer(context, n_contributors);

    gpu::ForwardBindings bindings{};
    bindings.projection = projection_buffers.bindings;
    bindings.index_buffer_offset = &index_buffer_offset_buffer;
    bindings.sorting_keys_1 = &sorting_keys_1_buffer;
    bindings.sorting_gauss_idx_1 = &sorting_gauss_idx_1_buffer;
    bindings.sorting_keys_2 = &sorting_keys_2_buffer;
    bindings.sorting_gauss_idx_2 = &sorting_gauss_idx_2_buffer;
    bindings.tile_ranges = &tile_ranges_buffer;
    bindings.pixel_state = &pixel_state_buffer;
    bindings.n_contributors = &n_contributors_buffer;

    const auto result = gpu::execute_forward(context, bindings,
                                             tests::default_renderer_uniforms(static_cast<std::uint32_t>(num_splats)));

    REQUIRE(result.num_indices == 0U);
    REQUIRE_FALSE(result.rasterized);
    REQUIRE(result.sorted_keys == nullptr);
    REQUIRE(result.sorted_gauss_idx == nullptr);

    const auto actual_tiles_touched = projection_buffers.tiles_touched.read_back<std::int32_t>(num_splats);
    const auto actual_index_buffer_offset = index_buffer_offset_buffer.read_back<std::int32_t>(num_splats);
    REQUIRE(actual_tiles_touched[0] == 0);
    REQUIRE(actual_index_buffer_offset[0] == 0);

    require_all_equal(sorting_keys_1_buffer.read_back<std::uint32_t>(max_indices), 0xFFFFFFFFU);
    require_all_equal(sorting_gauss_idx_1_buffer.read_back<std::int32_t>(max_indices), -1);
    require_all_equal(sorting_keys_2_buffer.read_back<std::uint32_t>(max_indices), 0xEEEEEEEEU);
    require_all_equal(sorting_gauss_idx_2_buffer.read_back<std::int32_t>(max_indices), -1);
    require_all_equal(tile_ranges_buffer.read_back<std::int32_t>(num_tiles + 1U), -1);
    require_all_equal(pixel_state_buffer.read_back<float>(num_pixels * tests::kPixelChannels), -1.0F);
    require_all_equal(n_contributors_buffer.read_back<std::int32_t>(num_pixels), -1);
  }
}
