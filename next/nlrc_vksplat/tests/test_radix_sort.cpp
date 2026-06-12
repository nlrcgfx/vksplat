#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "gpu/compute_pipeline.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "nlrc_vksplat_config.hpp"
#include "radix_sort_downsweep_spirv.hpp"
#include "radix_sort_spine_spirv.hpp"
#include "radix_sort_upsweep_spirv.hpp"
#include "span.hpp"

namespace {

constexpr std::uint32_t kRadixSortUpsweepBindingCount = 3;
constexpr std::uint32_t kRadixSortSpineBindingCount = 2;
constexpr std::uint32_t kRadixSortDownsweepBindingCount = 6;
constexpr std::uint32_t kRadixSortPasses = VKSPLAT_SORTING_KEY_BITS / VKSPLAT_RADIX_BITS_PER_PASS;

static_assert(VKSPLAT_SORTING_KEY_BITS == 32, "Update radix sort tests when 64-bit keys are enabled.");
static_assert(VKSPLAT_SORTING_KEY_BITS % VKSPLAT_RADIX_BITS_PER_PASS == 0);

struct RadixSortPushConstants final {
  std::uint32_t pass{};
  std::uint32_t element_count{};
};

static_assert(sizeof(RadixSortPushConstants) == 8);

template <typename T>
[[nodiscard]] auto make_storage_buffer(const nlrc::vksplat::gpu::HeadlessContext &context, const std::vector<T> &values)
    -> nlrc::vksplat::gpu::StorageBuffer {
  const std::size_t size_bytes = values.size() * sizeof(T);
  nlrc::vksplat::gpu::StorageBuffer buffer(context, size_bytes);

  buffer.upload(values);
  return buffer;
}

[[nodiscard]] constexpr std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  return (value + divisor - 1U) / divisor;
}

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

void dispatch_radix_sort_pass(nlrc::vksplat::gpu::ComputePipeline &pipeline,
                              const std::vector<const nlrc::vksplat::gpu::StorageBuffer *> &bindings,
                              nlrc::vksplat::gpu::DispatchShape dispatch_shape,
                              RadixSortPushConstants push_constants) {
  pipeline.bind_storage_buffers(bindings);

  auto push_constants_view = nlrc::vksplat::ByteView::from_object(push_constants);
  pipeline.dispatch(dispatch_shape, push_constants_view);
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

    const auto fixture_root = nlrc::vksplat::tests::fixture_dir(stage_name);
    const auto golden_root = nlrc::vksplat::tests::golden_dir(stage_name);

    const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = nlrc::vksplat::tests::load_fixture_manifest(golden_root / "manifest.json");

    const auto sorting_keys_1 =
        nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorting_keys_1");
    const auto sorting_gauss_idx_1 =
        nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorting_gauss_idx_1");
    const auto sorting_keys_2 =
        nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorting_keys_2");
    const auto sorting_gauss_idx_2 =
        nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "sorting_gauss_idx_2");
    const auto sorting_histogram =
        nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "_sorting_histogram");
    const auto sorting_histogram_cumsum =
        nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "_sorting_histogram_cumsum");

    REQUIRE_FALSE(sorting_keys_1.empty());
    REQUIRE(sorting_keys_1.size() == sorting_gauss_idx_1.size());
    REQUIRE(sorting_keys_2.size() == sorting_keys_1.size());
    REQUIRE(sorting_gauss_idx_2.size() == sorting_keys_1.size());
    REQUIRE(sorting_histogram.size() == static_cast<std::size_t>(kRadixSortPasses * VKSPLAT_RADIX_SORT_RADIX));

    const auto element_count = sorting_keys_1.size();
    const auto num_parts = ceil_div(element_count, static_cast<std::size_t>(VKSPLAT_RADIX_PARTITION_SIZE));
    REQUIRE(sorting_histogram_cumsum.size() == num_parts * VKSPLAT_RADIX_SORT_RADIX);

    const nlrc::vksplat::gpu::HeadlessContext context;

    auto sorting_keys_1_buffer = make_storage_buffer(context, sorting_keys_1);
    auto sorting_gauss_idx_1_buffer = make_storage_buffer(context, sorting_gauss_idx_1);
    auto sorting_keys_2_buffer = make_storage_buffer(context, sorting_keys_2);
    auto sorting_gauss_idx_2_buffer = make_storage_buffer(context, sorting_gauss_idx_2);
    auto sorting_histogram_buffer = make_storage_buffer(context, sorting_histogram);
    auto sorting_histogram_cumsum_buffer = make_storage_buffer(context, sorting_histogram_cumsum);

    auto upsweep_spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kRadixSortUpsweepSpirv);
    auto spine_spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kRadixSortSpineSpirv);
    auto downsweep_spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kRadixSortDownsweepSpirv);

    nlrc::vksplat::gpu::ComputePipeline upsweep_pipeline(context, upsweep_spirv, kRadixSortUpsweepBindingCount,
                                                         sizeof(RadixSortPushConstants));
    nlrc::vksplat::gpu::ComputePipeline spine_pipeline(context, spine_spirv, kRadixSortSpineBindingCount,
                                                       sizeof(RadixSortPushConstants));
    nlrc::vksplat::gpu::ComputePipeline downsweep_pipeline(context, downsweep_spirv, kRadixSortDownsweepBindingCount,
                                                           sizeof(RadixSortPushConstants));

    const auto partition_dispatch_shape =
        nlrc::vksplat::gpu::DispatchShape{static_cast<std::uint32_t>(num_parts), 1U, 1U};
    const auto spine_dispatch_shape = nlrc::vksplat::gpu::DispatchShape{VKSPLAT_RADIX_SORT_RADIX, 1U, 1U};

    const nlrc::vksplat::gpu::StorageBuffer *keys_in = &sorting_keys_1_buffer;
    const nlrc::vksplat::gpu::StorageBuffer *indices_in = &sorting_gauss_idx_1_buffer;
    const nlrc::vksplat::gpu::StorageBuffer *keys_out = &sorting_keys_2_buffer;
    const nlrc::vksplat::gpu::StorageBuffer *indices_out = &sorting_gauss_idx_2_buffer;

    // radix sort passes
    for (std::uint32_t pass = 0; pass < kRadixSortPasses; ++pass) {
      const RadixSortPushConstants push_constants = {pass, static_cast<std::uint32_t>(element_count)};

      dispatch_radix_sort_pass(upsweep_pipeline, {keys_in, &sorting_histogram_buffer, &sorting_histogram_cumsum_buffer},
                               partition_dispatch_shape, push_constants);

      dispatch_radix_sort_pass(spine_pipeline, {&sorting_histogram_buffer, &sorting_histogram_cumsum_buffer},
                               spine_dispatch_shape, push_constants);

      dispatch_radix_sort_pass(
          downsweep_pipeline,
          {&sorting_histogram_buffer, &sorting_histogram_cumsum_buffer, keys_in, indices_in, keys_out, indices_out},
          partition_dispatch_shape, push_constants);

      std::swap(keys_in, keys_out);
      std::swap(indices_in, indices_out);
    }

    const auto expected_keys =
        nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(golden_root, golden_manifest, "sorted_keys");

    const auto expected_indices =
        nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(golden_root, golden_manifest, "sorted_gauss_idx");

    const auto actual_keys = keys_in->read_back<std::uint32_t>(expected_keys.size());
    const auto actual_indices = indices_in->read_back<std::uint32_t>(expected_indices.size());

    REQUIRE(std::is_sorted(actual_keys.begin(), actual_keys.end()));

    require_index_permutation(actual_indices);
    require_stable_duplicate_order(actual_keys, actual_indices);

    REQUIRE(actual_keys == expected_keys);
    REQUIRE(actual_indices == expected_indices);
  }
}
