#include <array>
#include <cstdint>
#include <filesystem>
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

struct UtilityGoldenStage final {
  const std::filesystem::path &fixture_root;
  const tests::FixtureManifest &manifest;
  const std::filesystem::path &golden_root;
  const tests::FixtureManifest &golden_manifest;
  const gpu::HeadlessContext &context;
};

[[nodiscard]] std::vector<std::int32_t> inclusive_prefix_sum(const std::vector<std::int32_t> &input) {
  std::vector<std::int32_t> output(input.size());
  std::inclusive_scan(input.begin(), input.end(), output.begin());
  return output;
}

template <typename Dispatch, typename Validate>
void run_input_output_case(const UtilityGoldenStage &stage, Dispatch dispatch, Validate validate) {
  const auto input = tests::load_fixture_buffer<std::int32_t>(stage.fixture_root, stage.manifest, "input");
  const auto initial_output = tests::load_fixture_buffer<std::int32_t>(stage.fixture_root, stage.manifest, "output");

  auto input_buffer = gpu::make_storage_buffer(stage.context, input);
  auto output_buffer = gpu::make_storage_buffer(stage.context, initial_output);

  dispatch(input, input_buffer, output_buffer);

  const auto expected = tests::load_fixture_buffer<std::int32_t>(stage.golden_root, stage.golden_manifest, "output");
  const auto actual = output_buffer.read_back<std::int32_t>(expected.size());

  validate(input, expected, actual);
  REQUIRE(actual == expected);
}

template <typename Dispatch>
void run_where_case(const UtilityGoldenStage &stage, Dispatch dispatch) {
  const auto mask = tests::load_fixture_buffer<std::int32_t>(stage.fixture_root, stage.manifest, "mask");
  const auto mask_cumsum = tests::load_fixture_buffer<std::int32_t>(stage.fixture_root, stage.manifest, "mask_cumsum");
  const auto initial_out_indices =
      tests::load_fixture_buffer<std::int32_t>(stage.fixture_root, stage.manifest, "out_indices");

  auto mask_buffer = gpu::make_storage_buffer(stage.context, mask);
  auto mask_cumsum_buffer = gpu::make_storage_buffer(stage.context, mask_cumsum);
  auto out_indices_buffer = gpu::make_storage_buffer(stage.context, initial_out_indices);

  dispatch(mask, mask_buffer, mask_cumsum_buffer, out_indices_buffer);

  const auto expected =
      tests::load_fixture_buffer<std::int32_t>(stage.golden_root, stage.golden_manifest, "out_indices");
  const auto actual = out_indices_buffer.read_back<std::int32_t>(expected.size());
  REQUIRE(actual == expected);
}

template <std::size_t CaseCount, typename RunStage>
void run_utility_golden_cases(const std::array<const char *, CaseCount> &cases, RunStage run_stage) {
  for (const auto *stage_name : cases) {
    INFO("stage: " << stage_name);

    const auto fixture_root = tests::fixture_dir(stage_name);
    const auto golden_root = tests::golden_dir(stage_name);

    const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = tests::load_fixture_manifest(golden_root / "manifest.json");

    const gpu::HeadlessContext context;
    run_stage(UtilityGoldenStage{fixture_root, manifest, golden_root, golden_manifest, context});
  }
}

} // namespace

TEST_CASE("Dispatch cumsum single-pass utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 3> cases = {
      "cumsum_single_pass",
      "cumsum_single_pass_near_block",
      "cumsum_single_pass_exact_block",
  };

  run_utility_golden_cases(cases, [](const UtilityGoldenStage &stage) {
    run_input_output_case(
        stage,
        [&stage](const std::vector<std::int32_t> &input, const auto &input_buffer, auto &output_buffer) {
          REQUIRE(input.size() <= VKSPLAT_CUMSUM_BLOCK_SIZE);
          gpu::execute_cumsum(stage.context, input_buffer, output_buffer, input.size());
        },
        [](const auto &, const auto &, const auto &) {});
  });
}

TEST_CASE("Dispatch cumsum multi-block utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 2> cases = {
      "cumsum_multi_block",
      "cumsum_multi_block_two_level",
  };

  run_utility_golden_cases(cases, [](const UtilityGoldenStage &stage) {
    run_input_output_case(
        stage,
        [&stage](const std::vector<std::int32_t> &input, const auto &input_buffer, auto &output_buffer) {
          REQUIRE(input.size() > VKSPLAT_CUMSUM_BLOCK_SIZE);

          // The production helper owns the temporary block-sum buffers used by the multi-phase path.
          gpu::execute_cumsum(stage.context, input_buffer, output_buffer, input.size());
        },
        [](const std::vector<std::int32_t> &input, const std::vector<std::int32_t> &expected,
           const std::vector<std::int32_t> &actual) {
          const auto expected_from_input = inclusive_prefix_sum(input);
          REQUIRE(expected == expected_from_input);
          REQUIRE(actual.size() == expected.size());
        });
  });
}

TEST_CASE("Dispatch sum utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 2> cases = {
      "sum",
      "sum_multi_block",
  };

  run_utility_golden_cases(cases, [](const UtilityGoldenStage &stage) {
    run_input_output_case(
        stage,
        [&stage](const std::vector<std::int32_t> &input, const auto &input_buffer, auto &output_buffer) {
          gpu::execute_sum(stage.context, input_buffer, output_buffer, input.size());
        },
        [](const auto &, const auto &, const auto &) {});
  });
}

TEST_CASE("Dispatch where utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 4> cases = {
      "where",
      "where_no_true",
      "where_first_last",
      "where_block_boundary",
  };

  run_utility_golden_cases(cases, [](const UtilityGoldenStage &stage) {
    run_where_case(stage, [&stage](const std::vector<std::int32_t> &mask, const auto &mask_buffer,
                                   const auto &mask_cumsum_buffer, auto &out_indices_buffer) {
      gpu::execute_where(stage.context, mask_buffer, mask_cumsum_buffer, out_indices_buffer, mask.size());
    });
  });
}
