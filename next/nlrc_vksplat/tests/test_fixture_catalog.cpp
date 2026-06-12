#include <algorithm>
#include <array>
#include <filesystem>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "manifest_profile.hpp"

namespace {

[[nodiscard]] std::vector<std::filesystem::path> manifest_paths_under(const std::filesystem::path &root) {
  std::vector<std::filesystem::path> paths;
  if (!std::filesystem::exists(root)) {
    return paths;
  }

  for (const auto &entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_directory()) {
      continue;
    }
    const auto manifest_path = entry.path() / "manifest.json";
    if (std::filesystem::exists(manifest_path)) {
      paths.push_back(manifest_path);
    }
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

  REQUIRE(manifest.stage_name == manifest_path.parent_path().filename().string());
  REQUIRE(nlrc::vksplat::tests::manifest_matches_build_profile(manifest));

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

TEST_CASE("All committed fixture and golden manifests are valid", "[host]") {
  const std::array roots = {
      nlrc::vksplat::tests::test_data_root() / "fixtures",
      nlrc::vksplat::tests::test_data_root() / "golden_masters",
  };

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
}
