#include "fixture_loader.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <stdexcept>

#include "fixture_manifest.hpp"

namespace fs = std::filesystem;

namespace nlrc::vksplat::tests {

namespace {

[[nodiscard]] const std::map<std::string, fs::path> &stage_dirs_under(const fs::path &root) {
  static fs::path indexed_root;
  static std::map<std::string, fs::path> indexed_dirs;

  if (indexed_root != root) {
    indexed_root = root;
    indexed_dirs.clear();

    if (!fs::exists(root)) {
      return indexed_dirs;
    }

    for (const auto &entry : fs::recursive_directory_iterator(root)) {
      if (!entry.is_regular_file() || entry.path().filename() != "manifest.json") {
        continue;
      }

      const auto manifest = load_fixture_manifest(entry.path());
      if (indexed_dirs.find(manifest.stage_name) != indexed_dirs.end()) {
        throw std::runtime_error("Duplicate stage_name in " + root.string() + ": " + manifest.stage_name);
      }
      indexed_dirs.emplace(manifest.stage_name, entry.path().parent_path());
    }
  }

  return indexed_dirs;
}

[[nodiscard]] fs::path resolve_stage_dir(const fs::path &root, const std::string &stage_name) {
  const auto &dirs = stage_dirs_under(root);
  const auto dir_it = dirs.find(stage_name);
  if (dir_it == dirs.end()) {
    throw std::runtime_error("Stage not found under " + root.string() + ": " + stage_name);
  }
  return dir_it->second;
}

} // namespace

[[nodiscard]] fs::path test_data_root() {
  return {NLRC_VKSPLAT_TEST_DATA_DIR};
}

[[nodiscard]] fs::path fixture_dir(const std::string &stage_name) {
  return resolve_stage_dir(test_data_root() / "fixtures", stage_name);
}

[[nodiscard]] fs::path golden_dir(const std::string &stage_name) {
  return resolve_stage_dir(test_data_root() / "golden_masters", stage_name);
}

[[nodiscard]] std::vector<fs::path> manifest_paths_under(const fs::path &root) {
  std::vector<fs::path> paths;
  if (!fs::exists(root)) {
    return paths;
  }

  for (const auto &entry : fs::recursive_directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().filename() != "manifest.json") {
      continue;
    }
    paths.push_back(entry.path());
  }

  std::sort(paths.begin(), paths.end());
  return paths;
}

[[nodiscard]] std::vector<std::uint8_t> load_binary_file(const fs::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open binary file: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

[[nodiscard]] const BufferSpec &buffer_spec(const FixtureManifest &manifest, const std::string &buffer_name) {
  const auto spec_it = manifest.buffers.find(buffer_name);
  if (spec_it == manifest.buffers.end()) {
    throw std::runtime_error("Buffer not listed in manifest: " + buffer_name);
  }
  return spec_it->second;
}

[[nodiscard]] std::size_t buffer_element_count(const BufferSpec &spec) {
  if (spec.shape.empty()) {
    throw std::runtime_error("Buffer shape must not be empty");
  }
  for (const auto dim : spec.shape) {
    if (dim == 0) {
      throw std::runtime_error("Buffer shape dimensions must be positive");
    }
  }
  return std::accumulate(spec.shape.begin(), spec.shape.end(), std::size_t{1}, std::multiplies<>());
}

} // namespace nlrc::vksplat::tests
