#include <array>
#include <cstdint>
#include <numeric>
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

[[nodiscard]] std::vector<std::int32_t> inclusive_prefix_sum(const std::vector<std::int32_t> &input) {
  std::vector<std::int32_t> output(input.size());
  std::inclusive_scan(input.begin(), input.end(), output.begin());
  return output;
}

} // namespace

TEST_CASE("Dispatch cumsum single-pass utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 3> cases = {
      "cumsum_single_pass",
      "cumsum_single_pass_near_block",
      "cumsum_single_pass_exact_block",
  };

  for (const auto *stage_name : cases) {
    INFO("stage: " << stage_name);

    const auto fixture_root = tests::fixture_dir(stage_name);
    const auto golden_root = tests::golden_dir(stage_name);

    const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = tests::load_fixture_manifest(golden_root / "manifest.json");

    const auto input = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "input");
    const auto initial_output = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "output");

    REQUIRE(input.size() <= VKSPLAT_CUMSUM_BLOCK_SIZE);

    const gpu::HeadlessContext context;
    auto input_buffer = gpu::make_storage_buffer(context, input);
    auto output_buffer = gpu::make_storage_buffer(context, initial_output);

    gpu::execute_cumsum(context, input_buffer, output_buffer, input.size());

    const auto expected = tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "output");
    const auto actual = output_buffer.read_back<std::int32_t>(expected.size());

    REQUIRE(actual == expected);
  }
}

TEST_CASE("Dispatch cumsum multi-block utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 2> cases = {
      "cumsum_multi_block",
      "cumsum_multi_block_two_level",
  };

  for (const auto *stage_name : cases) {
    INFO("stage: " << stage_name);

    const auto fixture_root = tests::fixture_dir(stage_name);
    const auto golden_root = tests::golden_dir(stage_name);

    const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = tests::load_fixture_manifest(golden_root / "manifest.json");

    const auto input = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "input");
    const auto initial_output = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "output");

    REQUIRE(input.size() > VKSPLAT_CUMSUM_BLOCK_SIZE);

    const gpu::HeadlessContext context;
    auto input_buffer = gpu::make_storage_buffer(context, input);
    auto output_buffer = gpu::make_storage_buffer(context, initial_output);

    // The production helper owns the temporary block-sum buffers used by the multi-phase path.
    gpu::execute_cumsum(context, input_buffer, output_buffer, input.size());

    const auto expected = tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "output");
    const auto expected_from_input = inclusive_prefix_sum(input);
    const auto actual = output_buffer.read_back<std::int32_t>(expected.size());

    REQUIRE(expected == expected_from_input);
    REQUIRE(actual.size() == expected.size());
    REQUIRE(actual == expected);
  }
}

TEST_CASE("Dispatch sum utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 2> cases = {
      "sum",
      "sum_multi_block",
  };

  for (const auto *stage_name : cases) {
    INFO("stage: " << stage_name);

    const auto fixture_root = tests::fixture_dir(stage_name);
    const auto golden_root = tests::golden_dir(stage_name);

    const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = tests::load_fixture_manifest(golden_root / "manifest.json");

    const auto input = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "input");
    const auto initial_output = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "output");

    const gpu::HeadlessContext context;
    auto input_buffer = gpu::make_storage_buffer(context, input);
    auto output_buffer = gpu::make_storage_buffer(context, initial_output);

    gpu::execute_sum(context, input_buffer, output_buffer, input.size());

    const auto expected = tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "output");
    const auto actual = output_buffer.read_back<std::int32_t>(expected.size());

    REQUIRE(actual == expected);
  }
}

TEST_CASE("Dispatch where utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 4> cases = {
      "where",
      "where_no_true",
      "where_first_last",
      "where_block_boundary",
  };

  for (const auto *stage_name : cases) {
    INFO("stage: " << stage_name);

    const auto fixture_root = tests::fixture_dir(stage_name);
    const auto golden_root = tests::golden_dir(stage_name);

    const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = tests::load_fixture_manifest(golden_root / "manifest.json");

    const auto mask = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "mask");
    const auto mask_cumsum = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "mask_cumsum");
    const auto initial_out_indices = tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "out_indices");

    const gpu::HeadlessContext context;
    auto mask_buffer = gpu::make_storage_buffer(context, mask);
    auto mask_cumsum_buffer = gpu::make_storage_buffer(context, mask_cumsum);
    auto out_indices_buffer = gpu::make_storage_buffer(context, initial_out_indices);

    gpu::execute_where(context, mask_buffer, mask_cumsum_buffer, out_indices_buffer, mask.size());

    const auto expected = tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "out_indices");
    const auto actual = out_indices_buffer.read_back<std::int32_t>(expected.size());
    REQUIRE(actual == expected);
  }
}
