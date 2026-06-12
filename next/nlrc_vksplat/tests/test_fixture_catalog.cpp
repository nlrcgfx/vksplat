#include <algorithm>
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"

namespace {

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

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
void validate_manifest_file(const std::filesystem::path &manifest_path) {
  INFO("manifest: " << manifest_path.string());

  const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(manifest_path);
  REQUIRE_FALSE(manifest.ref_baseline_tag.empty());
  REQUIRE_FALSE(manifest.stage_name.empty());
  REQUIRE_FALSE(manifest.subgraph.empty());
  REQUIRE_FALSE(manifest.cmake_preset.empty());
  REQUIRE_FALSE(manifest.notes.empty());

  REQUIRE((manifest.emulate_int64 == 0 || manifest.emulate_int64 == 1));
  REQUIRE((manifest.emulate_f32_atomic == 0 || manifest.emulate_f32_atomic == 1));

  for (const auto &binding : manifest.bindings) {
    INFO("binding: " << binding);
    REQUIRE(manifest.buffers.find(binding) != manifest.buffers.end());
  }

  for (const auto &[buffer_name, spec] : manifest.buffers) {
    INFO("buffer: " << buffer_name);
    REQUIRE_FALSE(spec.file.empty());
    REQUIRE_FALSE(spec.shape.empty());
    for (const auto dim : spec.shape) {
      REQUIRE(dim > 0);
    }

    const auto buffer_path = manifest_path.parent_path() / spec.file;
    REQUIRE(std::filesystem::exists(buffer_path));
    REQUIRE(std::filesystem::is_regular_file(buffer_path));

    const auto expected_bytes =
        nlrc::vksplat::tests::buffer_element_count(spec) * nlrc::vksplat::tests::buffer_dtype_size(spec.dtype);
    REQUIRE(std::filesystem::file_size(buffer_path) == expected_bytes);
  }
}

} // namespace

[[nodiscard]] std::map<std::string, std::filesystem::path> stage_relative_paths_under(
    const std::filesystem::path &root) {
  std::map<std::string, std::filesystem::path> paths_by_stage;
  for (const auto &manifest_path : manifest_paths_under(root)) {
    const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(manifest_path);
    const auto relative_path = std::filesystem::relative(manifest_path.parent_path(), root);
    const auto [it, inserted] = paths_by_stage.emplace(manifest.stage_name, relative_path);
    REQUIRE(inserted);
  }
  return paths_by_stage;
}

TEST_CASE("All committed fixture and golden manifests are valid", "[host]") {
  const auto fixtures_root = nlrc::vksplat::tests::test_data_root() / "fixtures";
  const auto goldens_root = nlrc::vksplat::tests::test_data_root() / "golden_masters";
  const std::array roots = {fixtures_root, goldens_root};

  std::size_t checked_manifests = 0;
  for (const auto &root : roots) {
    INFO("root: " << root.string());
    REQUIRE(std::filesystem::exists(root));
    for (const auto &manifest_path : manifest_paths_under(root)) {
      validate_manifest_file(manifest_path);
      ++checked_manifests;
    }
  }

  REQUIRE(checked_manifests > 0);

  const auto fixture_paths = stage_relative_paths_under(fixtures_root);
  const auto golden_paths = stage_relative_paths_under(goldens_root);
  REQUIRE(fixture_paths.size() == golden_paths.size());
  for (const auto &[stage_name, fixture_path] : fixture_paths) {
    INFO("stage: " << stage_name);
    const auto golden_it = golden_paths.find(stage_name);
    REQUIRE(golden_it != golden_paths.end());
    REQUIRE(fixture_path == golden_it->second);
  }
}
