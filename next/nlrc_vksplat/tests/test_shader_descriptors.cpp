#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "gpu/shader_descriptors.hpp"
#include "gpu/shader_execution.hpp"

using namespace nlrc::vksplat;

namespace {

struct ExpectedShader final {
  gpu::ShaderId id;
  const char *logical_name;
  std::uint32_t binding_count;
  std::size_t push_constant_size;
  const char *source_path;
};

[[nodiscard]] bool has_prefix(const std::string &value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

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

[[nodiscard]] const gpu::ShaderInterface *
shader_interface_for_fixture(const tests::FixtureManifest &manifest) {
  if (manifest.subgraph == "projection" && has_prefix(manifest.stage_name, "projection_forward_")) {
    return &gpu::shader_interface("projection_forward");
  }
  if (manifest.subgraph == "generate_keys" && has_prefix(manifest.stage_name, "generate_keys_")) {
    return &gpu::shader_interface("generate_keys");
  }
  if (manifest.subgraph == "compute_tile_ranges" && has_prefix(manifest.stage_name, "compute_tile_ranges")) {
    return &gpu::shader_interface("compute_tile_ranges");
  }
  if (manifest.subgraph == "rasterize_forward" && has_prefix(manifest.stage_name, "rasterize_forward_")) {
    return &gpu::shader_interface("rasterize_forward");
  }
  if (manifest.subgraph == "utility") {
    if (manifest.stage_name == "sum" || has_prefix(manifest.stage_name, "sum_")) {
      return &gpu::shader_interface("sum");
    }
    if (manifest.stage_name == "where" || has_prefix(manifest.stage_name, "where_")) {
      return &gpu::shader_interface("where");
    }
    if (manifest.stage_name == "cumsum_single_pass" || has_prefix(manifest.stage_name, "cumsum_single_pass_")) {
      return &gpu::shader_interface("cumsum_single_pass");
    }
  }
  return nullptr;
}

[[nodiscard]] const gpu::FixtureBindingContract *
fixture_contract_for_manifest(const tests::FixtureManifest &manifest) {
  if (manifest.subgraph == "radix_sort" && has_prefix(manifest.stage_name, "radix_sort_")) {
    return &gpu::fixture_binding_contract("radix_sort_pipeline");
  }
  if (manifest.subgraph == "utility" && manifest.stage_name == "cumsum_multi_block") {
    return &gpu::fixture_binding_contract("cumsum_multi_block");
  }
  if (manifest.subgraph == "utility" && manifest.stage_name == "cumsum_multi_block_two_level") {
    return &gpu::fixture_binding_contract("cumsum_multi_block_two_level");
  }
  return nullptr;
}

[[nodiscard]] bool is_intentionally_untracked_fixture(const tests::FixtureManifest &manifest) {
  return manifest.subgraph == "harness";
}

} // namespace

TEST_CASE("Shader descriptor registry exposes ported logical shader metadata", "[host]") {
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
    if (manifest.subgraph != "projection" || !has_prefix(manifest.stage_name, "projection_forward_")) {
      continue;
    }

    INFO("manifest: " << manifest_path.string());
    REQUIRE(manifest.bindings == registry_bindings);
    ++checked_projection_fixtures;
  }

  REQUIRE(checked_projection_fixtures > 0);
}

TEST_CASE("Fixture catalog bindings match shader descriptor registry", "[host]") {
  std::size_t checked_fixtures = 0;

  for (const auto &manifest_path : manifest_paths_under(tests::test_data_root() / "fixtures")) {
    const auto manifest = tests::load_fixture_manifest(manifest_path);
    INFO("manifest: " << manifest_path.string());
    INFO("stage: " << manifest.stage_name);

    if (const auto *shader = shader_interface_for_fixture(manifest); shader != nullptr) {
      REQUIRE(manifest.bindings == gpu::binding_names(*shader));
      ++checked_fixtures;
      continue;
    }

    if (const auto *contract = fixture_contract_for_manifest(manifest); contract != nullptr) {
      REQUIRE(manifest.bindings == gpu::binding_names(*contract));
      ++checked_fixtures;
      continue;
    }

    if (is_intentionally_untracked_fixture(manifest)) {
      continue;
    }

    FAIL("Fixture has no shader descriptor or composite fixture binding contract");
  }

  REQUIRE(checked_fixtures > 0);
}
