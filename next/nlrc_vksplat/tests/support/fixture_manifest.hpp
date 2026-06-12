#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

namespace nlrc::vksplat::tests {

inline constexpr double kDefaultFixtureEpsilon = 1e-5;

enum class BufferDtype {
  Float32,
  UInt32,
  Int32,
  Int64,
};

struct BufferSpec final {
  std::string file;
  std::vector<std::size_t> shape;
  BufferDtype dtype{BufferDtype::Float32};
  double epsilon{kDefaultFixtureEpsilon};
};

struct FixtureManifest {
  std::string ref_baseline_tag;
  std::string stage_name;
  std::string subgraph;
  std::vector<std::string> bindings;
  std::map<std::string, BufferSpec> buffers;
  std::string cmake_preset;
  int emulate_int64{0};
  int emulate_f32_atomic{0};
  std::string notes;
  std::string vkbd_source;
};

[[nodiscard]] BufferDtype parse_buffer_dtype(const std::string &dtype);
[[nodiscard]] const char *buffer_dtype_name(BufferDtype dtype);
[[nodiscard]] std::size_t buffer_dtype_size(BufferDtype dtype);

template <typename T>
struct BufferDtypeTraits;

template <>
struct BufferDtypeTraits<float> {
  static constexpr BufferDtype value = BufferDtype::Float32;
  static constexpr const char *name = "float32";
};

template <>
struct BufferDtypeTraits<std::uint32_t> {
  static constexpr BufferDtype value = BufferDtype::UInt32;
  static constexpr const char *name = "uint32";
};

template <>
struct BufferDtypeTraits<std::int32_t> {
  static constexpr BufferDtype value = BufferDtype::Int32;
  static constexpr const char *name = "int32";
};

template <>
struct BufferDtypeTraits<std::int64_t> {
  static constexpr BufferDtype value = BufferDtype::Int64;
  static constexpr const char *name = "int64";
};

[[nodiscard]] FixtureManifest load_fixture_manifest(const std::filesystem::path &manifest_path);

} // namespace nlrc::vksplat::tests
