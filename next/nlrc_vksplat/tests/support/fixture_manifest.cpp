#include "fixture_manifest.hpp"

#include <fstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace nlrc::vksplat::tests {
namespace {

[[nodiscard]] std::vector<std::size_t> parse_shape(const nlohmann::json &value) {
  std::vector<std::size_t> shape;
  for (const auto &dim : value) {
    shape.push_back(dim.get<std::size_t>());
  }
  return shape;
}

} // namespace

FixtureManifest load_fixture_manifest(const std::filesystem::path &manifest_path) {
  std::ifstream input(manifest_path);
  if (!input) {
    throw std::runtime_error("Failed to open manifest: " + manifest_path.string());
  }

  const nlohmann::json json = nlohmann::json::parse(input);
  FixtureManifest manifest;
  manifest.ref_baseline_tag = json.at("ref_baseline_tag").get<std::string>();
  manifest.stage_name = json.at("stage_name").get<std::string>();
  manifest.subgraph = json.at("subgraph").get<std::string>();
  manifest.bindings = json.at("bindings").get<std::vector<std::string>>();
  manifest.buffers = json.at("buffers").get<std::map<std::string, std::string>>();
  manifest.cmake_preset = json.at("cmake_preset").get<std::string>();
  manifest.emulate_int64 = json.at("emulate_int64").get<int>();
  manifest.emulate_f32_atomic = json.at("emulate_f32_atomic").get<int>();
  manifest.notes = json.value("notes", "");

  for (const auto &[name, shape] : json.at("shapes").items()) {
    manifest.shapes[name] = parse_shape(shape);
  }
  manifest.dtypes = json.at("dtypes").get<std::map<std::string, std::string>>();

  if (json.contains("vkbd_source")) {
    manifest.vkbd_source = json.at("vkbd_source").get<std::string>();
  }
  if (json.contains("epsilon")) {
    for (const auto &[name, value] : json.at("epsilon").items()) {
      manifest.epsilon[name] = value.get<double>();
    }
  }

  return manifest;
}

} // namespace nlrc::vksplat::tests
