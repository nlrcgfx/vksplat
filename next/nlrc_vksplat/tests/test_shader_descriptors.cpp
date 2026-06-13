#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
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

#include <nlohmann/json.hpp>

using namespace nlrc::vksplat;

namespace {

struct ExpectedShader final {
  gpu::ShaderId id;
  const char *logical_name;
  std::uint32_t binding_count;
  const char *push_constant_type;
  std::size_t push_constant_size;
  const char *source_path;
};

[[nodiscard]] nlohmann::json load_shader_manifest() {
  const auto manifest_path = std::filesystem::path(NLRC_VKSPLAT_SHADER_GENERATED_DIR) / "shader_manifest.json";
  std::ifstream input(manifest_path);
  if (!input.is_open()) {
    throw std::runtime_error("could not open generated shader manifest: " + manifest_path.string());
  }
  return nlohmann::json::parse(input);
}

[[nodiscard]] nlohmann::json load_shader_binding_contracts() {
  const auto manifest_path = tests::test_data_root() / "shader_binding_contracts.json";
  std::ifstream input(manifest_path);
  if (!input.is_open()) {
    throw std::runtime_error("could not open shader binding contracts: " + manifest_path.string());
  }
  return nlohmann::json::parse(input);
}

[[nodiscard]] std::vector<std::string> json_string_array(const nlohmann::json &value) {
  std::vector<std::string> strings;
  strings.reserve(value.size());
  for (const auto &item : value) {
    strings.push_back(item.get<std::string>());
  }
  return strings;
}

[[nodiscard]] std::string manifest_source_path(const nlohmann::json &module) {
  const auto source = module.at("source").get<std::string>();
  const auto language = module.at("language").get<std::string>();
  if (language == "slang") {
    return "slang/" + source;
  }
  if (language == "glsl") {
    return "shader/" + source;
  }
  throw std::invalid_argument("unsupported shader manifest language: " + language);
}

[[nodiscard]] std::map<std::string, int> manifest_defines(const nlohmann::json &module) {
  std::map<std::string, int> defines;
  for (auto iter = module.at("defines").begin(); iter != module.at("defines").end(); ++iter) {
    defines.emplace(iter.key(), iter.value().get<int>());
  }
  return defines;
}

[[nodiscard]] std::map<std::string, int> shader_defines(const gpu::ShaderInterface &shader) {
  std::map<std::string, int> defines;
  for (std::size_t index = 0; index < shader.source.defines.size(); ++index) {
    defines.emplace(shader.source.defines[index].name, shader.source.defines[index].value);
  }
  return defines;
}

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
                     "push_constants::ElementCount", sizeof(gpu::push_constants::ElementCount), "slang/cumsum.slang"},
      ExpectedShader{gpu::ShaderId::CumsumBlockScan, "cumsum_block_scan", gpu::kCumsumBindingCount,
                     "push_constants::ElementCount", sizeof(gpu::push_constants::ElementCount), "slang/cumsum.slang"},
      ExpectedShader{gpu::ShaderId::CumsumScanBlockSums, "cumsum_scan_block_sums", gpu::kCumsumBindingCount,
                     "push_constants::ElementCount", sizeof(gpu::push_constants::ElementCount), "slang/cumsum.slang"},
      ExpectedShader{gpu::ShaderId::CumsumAddBlockOffsets, "cumsum_add_block_offsets", gpu::kCumsumBindingCount,
                     "push_constants::ElementCount", sizeof(gpu::push_constants::ElementCount), "slang/cumsum.slang"},
      ExpectedShader{gpu::ShaderId::Sum, "sum", gpu::kSumBindingCount, "push_constants::ElementCount",
                     sizeof(gpu::push_constants::ElementCount), "slang/sum.slang"},
      ExpectedShader{gpu::ShaderId::Where, "where", gpu::kWhereBindingCount, "push_constants::ElementCount",
                     sizeof(gpu::push_constants::ElementCount), "slang/where.slang"},
      ExpectedShader{gpu::ShaderId::ProjectionForward, "projection_forward", gpu::kProjectionForwardBindingCount,
                     "push_constants::Renderer", sizeof(gpu::push_constants::Renderer), "slang/vertex_shader.slang"},
      ExpectedShader{gpu::ShaderId::GenerateKeys, "generate_keys", gpu::kGenerateKeysBindingCount,
                     "push_constants::Renderer", sizeof(gpu::push_constants::Renderer), "slang/tile_shader.slang"},
      ExpectedShader{gpu::ShaderId::ComputeTileRanges, "compute_tile_ranges", gpu::kComputeTileRangesBindingCount,
                     "push_constants::Renderer", sizeof(gpu::push_constants::Renderer), "slang/tile_shader.slang"},
      ExpectedShader{gpu::ShaderId::RasterizeForward, "rasterize_forward", gpu::kRasterizeForwardBindingCount,
                     "push_constants::Renderer", sizeof(gpu::push_constants::Renderer),
                     "slang/alphablend_shader.slang"},
      ExpectedShader{gpu::ShaderId::RadixSortUpsweep, "radix_sort_upsweep", gpu::kRadixSortUpsweepBindingCount,
                     "push_constants::RadixSort", sizeof(gpu::push_constants::RadixSort),
                     "shader/radix_sort/upsweep.comp"},
      ExpectedShader{gpu::ShaderId::RadixSortSpine, "radix_sort_spine", gpu::kRadixSortSpineBindingCount,
                     "push_constants::RadixSort", sizeof(gpu::push_constants::RadixSort),
                     "shader/radix_sort/spine.comp"},
      ExpectedShader{gpu::ShaderId::RadixSortDownsweep, "radix_sort_downsweep", gpu::kRadixSortDownsweepBindingCount,
                     "push_constants::RadixSort", sizeof(gpu::push_constants::RadixSort),
                     "shader/radix_sort/downsweep.comp"},
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
    REQUIRE(std::string_view(by_id.push_constant_type) == expected_shader.push_constant_type);
    REQUIRE(by_id.push_constant_size == expected_shader.push_constant_size);
    REQUIRE(std::string_view(by_id.source.path) == expected_shader.source_path);
    REQUIRE_FALSE(std::string_view(by_id.dispatch.formula).empty());
  }
}

TEST_CASE("Shader descriptor registry matches compiled shader manifest", "[host]") {
  const auto manifest = load_shader_manifest();
  const auto &modules = manifest.at("modules");
  std::set<std::string> manifest_shader_names;

  std::size_t checked_modules = 0;
  for (const auto &module : modules) {
    const auto name = module.at("name").get<std::string>();
    if (name == "smoke") {
      continue;
    }

    INFO("shader manifest module: " << name);
    REQUIRE(manifest_shader_names.insert(name).second);

    const auto &shader = gpu::shader_interface(name);
    REQUIRE(std::string(shader.logical_name) == name);
    REQUIRE(std::string(shader.source.language) == module.at("language").get<std::string>());
    REQUIRE(std::string(shader.source.path) == manifest_source_path(module));
    REQUIRE(shader_defines(shader) == manifest_defines(module));
    ++checked_modules;
  }

  REQUIRE(checked_modules == gpu::shader_interfaces().size());
  const auto registry_shaders = gpu::shader_interfaces();
  for (std::size_t index = 0; index < registry_shaders.size(); ++index) {
    const auto &shader = registry_shaders[index];
    INFO("shader registry entry: " << shader.logical_name);
    REQUIRE(manifest_shader_names.count(shader.logical_name) == 1U);
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

TEST_CASE("Shader descriptor binding contract export matches descriptor registry", "[host]") {
  const auto contracts = load_shader_binding_contracts();

  REQUIRE(contracts.at("schema").get<int>() == 2);

  const auto &shaders = contracts.at("shaders");
  REQUIRE(shaders.size() == gpu::shader_interfaces().size());
  const auto registry_shaders = gpu::shader_interfaces();
  for (std::size_t index = 0; index < registry_shaders.size(); ++index) {
    const auto &shader = registry_shaders[index];
    INFO("shader: " << shader.logical_name);
    REQUIRE(shaders.contains(shader.logical_name));
    REQUIRE(json_string_array(shaders.at(shader.logical_name)) == gpu::binding_names(shader));
  }

  const auto &fixture_contracts = contracts.at("fixture_contracts");
  REQUIRE(fixture_contracts.is_object());
  REQUIRE_FALSE(fixture_contracts.empty());
  REQUIRE(contracts.at("routes").is_array());
  REQUIRE_FALSE(contracts.at("routes").empty());
  REQUIRE(contracts.at("untracked_routes").is_array());
  REQUIRE_FALSE(contracts.at("untracked_routes").empty());
}

TEST_CASE("Fixture catalog and golden bindings match JSON shader binding routes", "[host]") {
  struct Root final {
    const char *name;
    std::filesystem::path path;
    tests::FixtureManifestSide side;
  };

  const std::array roots = {
      Root{"fixtures", tests::test_data_root() / "fixtures", tests::FixtureManifestSide::Fixture},
      Root{"golden_masters", tests::test_data_root() / "golden_masters", tests::FixtureManifestSide::Golden},
  };

  std::size_t checked_manifests = 0;

  for (const auto &root : roots) {
    for (const auto &manifest_path : manifest_paths_under(root.path)) {
      const auto manifest = tests::load_fixture_manifest(manifest_path);
      INFO("root: " << root.name);
      INFO("manifest: " << manifest_path.string());
      INFO("stage: " << manifest.stage_name);

      const auto resolution = tests::fixture_binding_resolution(manifest, root.side);
      if (resolution.untracked) {
        REQUIRE(manifest.binding_contract.empty());
        continue;
      }

      REQUIRE_FALSE(resolution.contract.empty());
      REQUIRE(manifest.binding_contract == resolution.contract);
      REQUIRE(manifest.bindings == resolution.bindings);
      ++checked_manifests;
    }
  }

  REQUIRE(checked_manifests > 0);
}
