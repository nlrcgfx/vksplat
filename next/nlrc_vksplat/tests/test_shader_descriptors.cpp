#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "gpu/shader_binding_resolver.hpp"
#include "gpu/shader_descriptors.hpp"
#include "gpu/shader_execution.hpp"
#include "shader_fixture_mapping.hpp"

using namespace nlrc::vksplat;

namespace {

struct ExpectedShader final {
  gpu::ShaderId id;
  const char *logical_name;
  std::uint32_t binding_count;
  std::size_t push_constant_size;
  const char *source_path;
};

template <typename T>
inline constexpr std::size_t kStorageBindingArraySize = std::tuple_size_v<std::remove_cv_t<std::remove_reference_t<T>>>;

static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::CumsumSinglePass>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::CumsumBlockScan>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::CumsumScanBlockSums>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::CumsumAddBlockOffsets>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::Sum>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::Where>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::ProjectionForward>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::GenerateKeys>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::ComputeTileRanges>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::RasterizeForward>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::RadixSortUpsweep>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::RadixSortSpine>());
static_assert(gpu::storage_binding_names_match_registry<gpu::ShaderId::RadixSortDownsweep>());

// clang-format off
static_assert(kStorageBindingArraySize<decltype(gpu::cumsum_storage_bindings(
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::CumsumSinglePass>);
static_assert(kStorageBindingArraySize<decltype(gpu::sum_storage_bindings(
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::Sum>);
static_assert(kStorageBindingArraySize<decltype(gpu::where_storage_bindings(
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::Where>);
static_assert(kStorageBindingArraySize<decltype(gpu::projection_forward_storage_bindings(
                  std::declval<const gpu::ProjectionForwardBindings &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::ProjectionForward>);
static_assert(kStorageBindingArraySize<decltype(gpu::generate_keys_storage_bindings(
                  std::declval<const gpu::GenerateKeysBindings &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::GenerateKeys>);
static_assert(kStorageBindingArraySize<decltype(gpu::compute_tile_ranges_storage_bindings(
                  std::declval<const gpu::ComputeTileRangesBindings &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::ComputeTileRanges>);
static_assert(kStorageBindingArraySize<decltype(gpu::rasterize_forward_storage_bindings(
                  std::declval<const gpu::RasterizeForwardBindings &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::RasterizeForward>);
static_assert(kStorageBindingArraySize<decltype(gpu::radix_sort_upsweep_storage_bindings(
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::RadixSortUpsweep>);
static_assert(kStorageBindingArraySize<decltype(gpu::radix_sort_spine_storage_bindings(
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::RadixSortSpine>);
static_assert(kStorageBindingArraySize<decltype(gpu::radix_sort_downsweep_storage_bindings(
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>(),
                  std::declval<const gpu::StorageBuffer &>()))> ==
              gpu::shader_binding_count_v<gpu::ShaderId::RadixSortDownsweep>);
// clang-format on

[[nodiscard]] std::vector<std::filesystem::path> manifest_paths_under(const std::filesystem::path &root) {
  std::vector<std::filesystem::path> paths;
  if (!std::filesystem::exists(root)) {
    return paths;
  }

  for (const auto &entry : std::filesystem::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().filename() != "manifest.json") {
      continue;
    }
    paths.push_back(entry.path());
  }

  std::sort(paths.begin(), paths.end());
  return paths;
}

template <gpu::ShaderId Id, std::size_t... Indices>
// NOLINTNEXTLINE(readability-named-parameter)
[[nodiscard]] std::vector<std::string> registry_binding_name_strings(std::index_sequence<Indices...>) {
  return {std::string(gpu::shader_binding_name<Id, Indices>())...};
}

template <gpu::ShaderId Id>
[[nodiscard]] std::vector<std::string> registry_binding_name_strings() {
  return registry_binding_name_strings<Id>(std::make_index_sequence<gpu::shader_binding_count_v<Id>>{});
}

} // namespace

TEST_CASE("Shader descriptor registry exposes ported logical shader metadata", "[host]") {
  // clang-format off
  const std::array expected = {
      ExpectedShader{gpu::ShaderId::CumsumSinglePass, "cumsum_single_pass", gpu::kCumsumBindingCount,
                     sizeof(gpu::ElementCountPushConstants), "slang/cumsum.slang"},
      ExpectedShader{gpu::ShaderId::CumsumBlockScan, "cumsum_block_scan", gpu::kCumsumBindingCount,
                     sizeof(gpu::ElementCountPushConstants), "slang/cumsum.slang"},
      ExpectedShader{gpu::ShaderId::CumsumScanBlockSums, "cumsum_scan_block_sums", gpu::kCumsumBindingCount,
                     sizeof(gpu::ElementCountPushConstants), "slang/cumsum.slang"},
      ExpectedShader{gpu::ShaderId::CumsumAddBlockOffsets, "cumsum_add_block_offsets", gpu::kCumsumBindingCount,
                     sizeof(gpu::ElementCountPushConstants), "slang/cumsum.slang"},
      ExpectedShader{gpu::ShaderId::Sum, "sum", gpu::kSumBindingCount, sizeof(gpu::ElementCountPushConstants),
                     "slang/sum.slang"},
      ExpectedShader{gpu::ShaderId::Where, "where", gpu::kWhereBindingCount, sizeof(gpu::ElementCountPushConstants),
                     "slang/where.slang"},
      ExpectedShader{gpu::ShaderId::ProjectionForward, "projection_forward", gpu::kProjectionForwardBindingCount,
                     sizeof(gpu::RendererUniforms), "slang/vertex_shader.slang"},
      ExpectedShader{gpu::ShaderId::GenerateKeys, "generate_keys", gpu::kGenerateKeysBindingCount,
                     sizeof(gpu::RendererUniforms), "slang/tile_shader.slang"},
      ExpectedShader{gpu::ShaderId::ComputeTileRanges, "compute_tile_ranges", gpu::kComputeTileRangesBindingCount,
                     sizeof(gpu::RendererUniforms), "slang/tile_shader.slang"},
      ExpectedShader{gpu::ShaderId::RasterizeForward, "rasterize_forward", gpu::kRasterizeForwardBindingCount,
                     sizeof(gpu::RendererUniforms), "slang/alphablend_shader.slang"},
      ExpectedShader{gpu::ShaderId::RadixSortUpsweep, "radix_sort_upsweep", gpu::kRadixSortUpsweepBindingCount,
                     sizeof(gpu::RadixSortPushConstants), "shader/radix_sort/upsweep.comp"},
      ExpectedShader{gpu::ShaderId::RadixSortSpine, "radix_sort_spine", gpu::kRadixSortSpineBindingCount,
                     sizeof(gpu::RadixSortPushConstants), "shader/radix_sort/spine.comp"},
      ExpectedShader{gpu::ShaderId::RadixSortDownsweep, "radix_sort_downsweep", gpu::kRadixSortDownsweepBindingCount,
                     sizeof(gpu::RadixSortPushConstants), "shader/radix_sort/downsweep.comp"},
  };
  // clang-format on

  REQUIRE(gpu::shader_interfaces().size() == expected.size());

  for (const auto &expected_shader : expected) {
    INFO("shader: " << expected_shader.logical_name);
    const auto &by_id = gpu::shader_interface(expected_shader.id);
    const auto &by_name = gpu::shader_interface(expected_shader.logical_name);

    REQUIRE(&by_id == &by_name);
    REQUIRE(std::string_view(by_id.logical_name) == expected_shader.logical_name);
    REQUIRE(by_id.bindings.size() == expected_shader.binding_count);
    REQUIRE(by_id.binding_count == expected_shader.binding_count);
    REQUIRE(by_id.push_constant_size == expected_shader.push_constant_size);
    REQUIRE(std::string_view(by_id.source.path) == expected_shader.source_path);
    REQUIRE_FALSE(std::string_view(by_id.dispatch.formula).empty());
  }
}

TEST_CASE("Projection descriptor matches binding count and projection fixtures", "[host]") {
  const auto &projection = gpu::shader_interface("projection_forward");
  const auto registry_bindings = gpu::binding_names(projection);

  REQUIRE(projection.binding_count == gpu::kProjectionForwardBindingCount);
  REQUIRE(registry_bindings.size() == gpu::kProjectionForwardBindingCount);

  std::size_t checked_projection_fixtures = 0;
  for (const auto &manifest_path : manifest_paths_under(tests::test_data_root() / "fixtures")) {
    const auto manifest = tests::load_fixture_manifest(manifest_path);
    if (manifest.subgraph != "projection" || manifest.stage_name.find("projection_forward_") != 0U) {
      continue;
    }

    INFO("manifest: " << manifest_path.string());
    REQUIRE(manifest.bindings == registry_bindings);
    ++checked_projection_fixtures;
  }

  REQUIRE(checked_projection_fixtures > 0);
}

TEST_CASE("Shader descriptors preserve dispatch binding order", "[host]") {
  // clang-format off
  REQUIRE(gpu::binding_names(gpu::shader_interface("cumsum_single_pass")) ==
          registry_binding_name_strings<gpu::ShaderId::CumsumSinglePass>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("cumsum_block_scan")) ==
          registry_binding_name_strings<gpu::ShaderId::CumsumBlockScan>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("cumsum_scan_block_sums")) ==
          registry_binding_name_strings<gpu::ShaderId::CumsumScanBlockSums>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("cumsum_add_block_offsets")) ==
          registry_binding_name_strings<gpu::ShaderId::CumsumAddBlockOffsets>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("sum")) ==
          registry_binding_name_strings<gpu::ShaderId::Sum>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("where")) ==
          registry_binding_name_strings<gpu::ShaderId::Where>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("projection_forward")) ==
          registry_binding_name_strings<gpu::ShaderId::ProjectionForward>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("generate_keys")) ==
          registry_binding_name_strings<gpu::ShaderId::GenerateKeys>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("compute_tile_ranges")) ==
          registry_binding_name_strings<gpu::ShaderId::ComputeTileRanges>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("rasterize_forward")) ==
          registry_binding_name_strings<gpu::ShaderId::RasterizeForward>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("radix_sort_upsweep")) ==
          registry_binding_name_strings<gpu::ShaderId::RadixSortUpsweep>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("radix_sort_spine")) ==
          registry_binding_name_strings<gpu::ShaderId::RadixSortSpine>());
  REQUIRE(gpu::binding_names(gpu::shader_interface("radix_sort_downsweep")) ==
          registry_binding_name_strings<gpu::ShaderId::RadixSortDownsweep>());
  // clang-format on
}

TEST_CASE("Fixture catalog bindings match shader descriptor registry", "[host]") {
  std::size_t checked_fixtures = 0;

  for (const auto &manifest_path : manifest_paths_under(tests::test_data_root() / "fixtures")) {
    const auto manifest = tests::load_fixture_manifest(manifest_path);
    INFO("manifest: " << manifest_path.string());
    INFO("stage: " << manifest.stage_name);

    if (const auto *shader = tests::shader_interface_for_fixture(manifest); shader != nullptr) {
      REQUIRE(manifest.bindings == gpu::binding_names(*shader));
      ++checked_fixtures;
      continue;
    }

    if (const auto *contract = tests::fixture_contract_for_manifest(manifest); contract != nullptr) {
      REQUIRE(manifest.bindings == gpu::binding_names(*contract));
      ++checked_fixtures;
      continue;
    }

    if (tests::fixture_is_intentionally_untracked(manifest)) {
      continue;
    }

    FAIL("Fixture has no shader descriptor or composite fixture binding contract");
  }

  REQUIRE(checked_fixtures > 0);
}
