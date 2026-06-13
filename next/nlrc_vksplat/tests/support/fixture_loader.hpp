#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixture_manifest.hpp"

namespace fs = std::filesystem;

namespace nlrc::vksplat::tests {

[[nodiscard]] fs::path test_data_root();
[[nodiscard]] fs::path fixture_dir(const std::string &stage_name);
[[nodiscard]] fs::path golden_dir(const std::string &stage_name);
[[nodiscard]] std::vector<fs::path> manifest_paths_under(const fs::path &root);

[[nodiscard]] std::vector<std::uint8_t> load_binary_file(const fs::path &path);

[[nodiscard]] const BufferSpec &buffer_spec(const FixtureManifest &manifest, const std::string &buffer_name);

[[nodiscard]] std::size_t buffer_element_count(const BufferSpec &spec);

template <typename T>
[[nodiscard]] std::vector<T>
load_fixture_buffer(const fs::path &fixture_root, const FixtureManifest &manifest, const std::string &buffer_name) {
  const BufferSpec &spec = buffer_spec(manifest, buffer_name);
  if (spec.dtype != BufferDtypeTraits<T>::kValue) {
    throw std::runtime_error("Buffer dtype mismatch for " + buffer_name + ": manifest has " +
                             buffer_dtype_name(spec.dtype) + ", loader requested " + BufferDtypeTraits<T>::kName);
  }

  const auto bytes = load_binary_file(fixture_root / spec.file);
  const std::size_t expected_bytes = buffer_element_count(spec) * sizeof(T);
  if (bytes.size() != expected_bytes) {
    throw std::runtime_error("Binary size mismatch for buffer: " + buffer_name);
  }

  std::vector<T> values(bytes.size() / sizeof(T));
  if (!bytes.empty()) {
    std::memcpy(values.data(), bytes.data(), bytes.size());
  }
  return values;
}

} // namespace nlrc::vksplat::tests
