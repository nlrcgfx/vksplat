#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "fixture_manifest.hpp"

namespace nlrc::vksplat::tests {

[[nodiscard]] std::filesystem::path test_data_root();
[[nodiscard]] std::filesystem::path fixture_dir(const std::string &stage_folder);
[[nodiscard]] std::filesystem::path golden_dir(const std::string &stage_folder);

[[nodiscard]] std::vector<std::uint8_t> load_binary_file(const std::filesystem::path &path);
[[nodiscard]] std::vector<float> load_fixture_float_buffer(const std::filesystem::path &fixture_root,
                                                           const FixtureManifest &manifest,
                                                           const std::string &buffer_name);

} // namespace nlrc::vksplat::tests
