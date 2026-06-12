#include "fixture_loader.hpp"

#include <cstring>
#include <fstream>
#include <numeric>
#include <stdexcept>

namespace nlrc::vksplat::tests {
namespace {

[[nodiscard]] std::size_t element_count(const std::vector<std::size_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
}

} // namespace

std::filesystem::path test_data_root() {
  return {NLRC_VKSPLAT_TEST_DATA_DIR};
}

std::filesystem::path fixture_dir(const std::string &stage_folder) {
  return test_data_root() / "fixtures" / stage_folder;
}

std::filesystem::path golden_dir(const std::string &stage_folder) {
  return test_data_root() / "golden_masters" / stage_folder;
}

std::vector<std::uint8_t> load_binary_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open binary file: " + path.string());
  }
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<float> load_fixture_float_buffer(const std::filesystem::path &fixture_root,
                                             const FixtureManifest &manifest,
                                             const std::string &buffer_name) {
  const auto file_it = manifest.buffers.find(buffer_name);
  if (file_it == manifest.buffers.end()) {
    throw std::runtime_error("Buffer not listed in manifest: " + buffer_name);
  }

  const auto dtype_it = manifest.dtypes.find(buffer_name);
  if (dtype_it == manifest.dtypes.end() || dtype_it->second != "float32") {
    throw std::runtime_error("Only float32 buffers are supported in harness loader: " + buffer_name);
  }

  const auto shape_it = manifest.shapes.find(buffer_name);
  if (shape_it == manifest.shapes.end()) {
    throw std::runtime_error("Missing shape for buffer: " + buffer_name);
  }

  const auto bytes = load_binary_file(fixture_root / file_it->second);
  const std::size_t expected_bytes = element_count(shape_it->second) * sizeof(float);
  if (bytes.size() != expected_bytes) {
    throw std::runtime_error("Binary size mismatch for buffer: " + buffer_name);
  }

  std::vector<float> values(bytes.size() / sizeof(float));
  std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}

} // namespace nlrc::vksplat::tests
