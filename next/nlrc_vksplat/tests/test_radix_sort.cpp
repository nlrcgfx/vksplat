#include <algorithm>
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

static_assert(VKSPLAT_SORTING_KEY_BITS == 32, "Update radix sort tests when 64-bit keys are enabled.");

// contract: indices is a permutation of 0..indices.size() - 1
void require_index_permutation(const std::vector<std::uint32_t> &indices) {
  std::vector<std::uint32_t> actual = indices;
  std::sort(actual.begin(), actual.end());

  std::vector<std::uint32_t> expected(indices.size());
  std::iota(expected.begin(), expected.end(), 0U);

  REQUIRE(actual == expected);
}

void require_stable_duplicate_order(const std::vector<std::uint32_t> &keys, const std::vector<std::uint32_t> &indices) {
  REQUIRE(keys.size() == indices.size());

  for (std::size_t index = 1; index < keys.size(); ++index) {
    if (keys[index - 1] == keys[index]) {
      REQUIRE(indices[index - 1] < indices[index]);
    }
  }
}

} // namespace

TEST_CASE("Dispatch radix sort pipeline", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 7> cases = {
      "radix_sort_minimum_one",     "radix_sort_single_partition", "radix_sort_partition_boundary",
      "radix_sort_multi_partition", "radix_sort_duplicates",       "radix_sort_sorted",
      "radix_sort_reverse",
  };

  for (const auto *stage_name : cases) {
    INFO("stage: " << stage_name);

    const auto fixture_root = tests::fixture_dir(stage_name);
    const auto golden_root = tests::golden_dir(stage_name);

    const auto manifest = tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = tests::load_fixture_manifest(golden_root / "manifest.json");

    const auto sorting_keys_1 = tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorting_keys_1");
    const auto sorting_gauss_idx_1 =
        tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorting_gauss_idx_1");
    const auto sorting_keys_2 = tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorting_keys_2");
    const auto sorting_gauss_idx_2 =
        tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorting_gauss_idx_2");

    REQUIRE_FALSE(sorting_keys_1.empty());
    REQUIRE(sorting_keys_1.size() == sorting_gauss_idx_1.size());
    REQUIRE(sorting_keys_2.size() == sorting_keys_1.size());
    REQUIRE(sorting_gauss_idx_2.size() == sorting_keys_1.size());

    const gpu::HeadlessContext context;

    auto sorting_keys_1_buffer = gpu::make_storage_buffer(context, sorting_keys_1);
    auto sorting_gauss_idx_1_buffer = gpu::make_storage_buffer(context, sorting_gauss_idx_1);
    auto sorting_keys_2_buffer = gpu::make_storage_buffer(context, sorting_keys_2);
    auto sorting_gauss_idx_2_buffer = gpu::make_storage_buffer(context, sorting_gauss_idx_2);

    gpu::RadixSortBindings bindings{};
    bindings.keys_1 = &sorting_keys_1_buffer;
    bindings.indices_1 = &sorting_gauss_idx_1_buffer;
    bindings.keys_2 = &sorting_keys_2_buffer;
    bindings.indices_2 = &sorting_gauss_idx_2_buffer;

    // Use the returned ping-pong buffers instead of assuming where the final pass lands.
    const auto result = gpu::execute_sort(context, bindings, sorting_keys_1.size());

    const auto expected_keys = tests::load_fixture_buffer<std::uint32_t>(golden_root, golden_manifest, "sorted_keys");
    const auto expected_indices =
        tests::load_fixture_buffer<std::uint32_t>(golden_root, golden_manifest, "sorted_gauss_idx");

    const auto actual_keys = result.keys->read_back<std::uint32_t>(expected_keys.size());
    const auto actual_indices = result.indices->read_back<std::uint32_t>(expected_indices.size());

    REQUIRE(std::is_sorted(actual_keys.begin(), actual_keys.end()));

    require_index_permutation(actual_indices);
    require_stable_duplicate_order(actual_keys, actual_indices);

    REQUIRE(actual_keys == expected_keys);
    REQUIRE(actual_indices == expected_indices);
  }
}
