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

BufferDtype parse_buffer_dtype(const std::string &dtype) {
  if (dtype == "float32") {
    return BufferDtype::Float32;
  }
  if (dtype == "uint32") {
    return BufferDtype::UInt32;
  }
  if (dtype == "int32") {
    return BufferDtype::Int32;
  }
  if (dtype == "int64") {
    return BufferDtype::Int64;
  }
  throw std::invalid_argument("Unsupported buffer dtype: " + dtype);
}

const char *buffer_dtype_name(BufferDtype dtype) {
  switch (dtype) {
    case BufferDtype::Float32:
      return "float32";
    case BufferDtype::UInt32:
      return "uint32";
    case BufferDtype::Int32:
      return "int32";
    case BufferDtype::Int64:
      return "int64";
  }
  throw std::invalid_argument("Unknown buffer dtype enum value");
}

std::size_t buffer_dtype_size(BufferDtype dtype) {
  switch (dtype) {
    case BufferDtype::Float32:
      return sizeof(float);
    case BufferDtype::UInt32:
      return sizeof(std::uint32_t);
    case BufferDtype::Int32:
      return sizeof(std::int32_t);
    case BufferDtype::Int64:
      return sizeof(std::int64_t);
  }
  throw std::invalid_argument("Unknown buffer dtype enum value");
}

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
  manifest.cmake_preset = json.at("cmake_preset").get<std::string>();
  manifest.emulate_int64 = json.at("emulate_int64").get<int>();
  manifest.emulate_f32_atomic = json.at("emulate_f32_atomic").get<int>();
  manifest.notes = json.value("notes", "");

  const auto buffer_files = json.at("buffers").get<std::map<std::string, std::string>>();
  std::map<std::string, std::vector<std::size_t>> shapes;
  for (const auto &[name, shape] : json.at("shapes").items()) {
    shapes[name] = parse_shape(shape);
  }
  const auto dtypes = json.at("dtypes").get<std::map<std::string, std::string>>();

  if (json.contains("vkbd_source")) {
    manifest.vkbd_source = json.at("vkbd_source").get<std::string>();
  }
  std::map<std::string, double> epsilons;
  if (json.contains("epsilon")) {
    for (const auto &[name, value] : json.at("epsilon").items()) {
      epsilons[name] = value.get<double>();
    }
  }

  for (const auto &[name, file] : buffer_files) {
    const auto shape_it = shapes.find(name);
    if (shape_it == shapes.end()) {
      throw std::runtime_error("Missing shape for buffer: " + name);
    }

    const auto dtype_it = dtypes.find(name);
    if (dtype_it == dtypes.end()) {
      throw std::runtime_error("Missing dtype for buffer: " + name);
    }

    const auto epsilon_it = epsilons.find(name);
    manifest.buffers.emplace(name,
                             BufferSpec{file, shape_it->second, parse_buffer_dtype(dtype_it->second),
                                        epsilon_it == epsilons.end() ? kDefaultFixtureEpsilon : epsilon_it->second});
  }

  return manifest;
}

} // namespace nlrc::vksplat::tests
