#include "fixture_loader.hpp"

#include <fstream>
#include <functional>
#include <numeric>
#include <stdexcept>

namespace nlrc::vksplat::tests {

[[nodiscard]] std::filesystem::path test_data_root() {
  return {NLRC_VKSPLAT_TEST_DATA_DIR};
}

[[nodiscard]] std::filesystem::path fixture_dir(const std::string &stage_folder) {
  return test_data_root() / "fixtures" / stage_folder;
}

[[nodiscard]] std::filesystem::path golden_dir(const std::string &stage_folder) {
  return test_data_root() / "golden_masters" / stage_folder;
}

[[nodiscard]] std::vector<std::uint8_t> load_binary_file(const std::filesystem::path &path) {
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
  return std::accumulate(spec.shape.begin(), spec.shape.end(), std::size_t{1}, std::multiplies<>());
}

} // namespace nlrc::vksplat::tests
