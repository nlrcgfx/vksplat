#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace nlrc::vksplat::tests {

struct FixtureManifest {
  std::string ref_baseline_tag;
  std::string stage_name;
  std::string subgraph;
  std::vector<std::string> bindings;
  std::map<std::string, std::string> buffers;
  std::map<std::string, std::vector<std::size_t>> shapes;
  std::map<std::string, std::string> dtypes;
  std::string cmake_preset;
  int emulate_int64{0};
  int emulate_f32_atomic{0};
  std::string notes;
  std::string vkbd_source;
  std::map<std::string, double> epsilon;
};

[[nodiscard]] FixtureManifest load_fixture_manifest(const std::filesystem::path &manifest_path);

} // namespace nlrc::vksplat::tests
