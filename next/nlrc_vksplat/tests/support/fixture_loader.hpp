#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixture_manifest.hpp"

namespace nlrc::vksplat::tests {

[[nodiscard]] std::filesystem::path test_data_root();
[[nodiscard]] std::filesystem::path fixture_dir(const std::string &stage_folder);
[[nodiscard]] std::filesystem::path golden_dir(const std::string &stage_folder);

[[nodiscard]] std::vector<std::uint8_t> load_binary_file(const std::filesystem::path &path);

[[nodiscard]] const BufferSpec &buffer_spec(const FixtureManifest &manifest, const std::string &buffer_name);

[[nodiscard]] std::size_t buffer_element_count(const BufferSpec &spec);

template <typename T>
[[nodiscard]] std::vector<T> load_fixture_buffer(const std::filesystem::path &fixture_root,
                                                 const FixtureManifest &manifest,
                                                 const std::string &buffer_name) {
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
