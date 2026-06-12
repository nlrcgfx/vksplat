#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"

using namespace nlrc::vksplat;

namespace {

struct TemporaryFile final {
  explicit TemporaryFile(std::filesystem::path file_path) : path(std::move(file_path)) {}

  TemporaryFile(const TemporaryFile &) = delete;
  TemporaryFile &operator=(const TemporaryFile &) = delete;

  TemporaryFile(TemporaryFile &&) = delete;
  TemporaryFile &operator=(TemporaryFile &&) = delete;

  ~TemporaryFile() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }

  std::filesystem::path path;
};

[[nodiscard]] TemporaryFile write_temp_manifest(const std::string &contents) {
  const auto timestamp = std::chrono::steady_clock::now().time_since_epoch().count();
  auto path = std::filesystem::temp_directory_path() / ("nlrc_vksplat_manifest_" + std::to_string(timestamp) + ".json");

  std::ofstream output(path);
  if (!output) {
    throw std::runtime_error("Failed to write temporary manifest: " + path.string());
  }
  output << contents;
  return TemporaryFile(path);
}

} // namespace

TEST_CASE("Load harness_smoke fixture manifest metadata", "[host]") {
  const auto root = tests::fixture_dir("harness_smoke");
  const auto manifest = tests::load_fixture_manifest(root / "manifest.json");
  REQUIRE(manifest.stage_name == "harness_smoke");
  REQUIRE(manifest.bindings.empty());
  REQUIRE(manifest.profile_agnostic);
}

TEST_CASE("Load harness_smoke fixture typed buffer spec", "[host]") {
  const auto root = tests::fixture_dir("harness_smoke");
  const auto manifest = tests::load_fixture_manifest(root / "manifest.json");

  const auto &spec = manifest.buffers.at("input_a");
  REQUIRE(spec.file == "input_a.bin");
  REQUIRE(spec.shape == std::vector<std::size_t>{4});
  REQUIRE(spec.dtype == tests::BufferDtype::Float32);
}

TEST_CASE("Typed fixture loader loads float buffers", "[host]") {
  const auto root = tests::fixture_dir("harness_smoke");
  const auto manifest = tests::load_fixture_manifest(root / "manifest.json");

  const auto values = tests::load_fixture_buffer<float>(root, manifest, "input_a");
  REQUIRE(values.size() == 4);
  REQUIRE(values[0] == 1.0F);
  REQUIRE(values[3] == 4.0F);
}

TEST_CASE("Fixture dtype parser maps supported manifest dtypes", "[host]") {
  using tests::BufferDtype;
  using tests::parse_buffer_dtype;

  REQUIRE(parse_buffer_dtype("float32") == BufferDtype::Float32);
  REQUIRE(parse_buffer_dtype("uint32") == BufferDtype::UInt32);
  REQUIRE(parse_buffer_dtype("int32") == BufferDtype::Int32);
  REQUIRE(parse_buffer_dtype("int64") == BufferDtype::Int64);
}

TEST_CASE("Fixture dtype helpers expose sizes and reject unknown types", "[host]") {
  using tests::buffer_dtype_size;
  using tests::BufferDtype;
  using tests::parse_buffer_dtype;

  REQUIRE(buffer_dtype_size(BufferDtype::Int64) == sizeof(std::int64_t));
  REQUIRE_THROWS_AS(parse_buffer_dtype("float16"), std::invalid_argument);
}

TEST_CASE("Typed fixture loader rejects dtype and size mismatches", "[host]") {
  const auto root = tests::fixture_dir("harness_smoke");
  auto manifest = tests::load_fixture_manifest(root / "manifest.json");

  REQUIRE_THROWS_AS(tests::load_fixture_buffer<float>(root, manifest, "missing"), std::runtime_error);

  REQUIRE_THROWS_AS(tests::load_fixture_buffer<std::uint32_t>(root, manifest, "input_a"), std::runtime_error);

  manifest.buffers.at("input_a").file = "missing.bin";
  REQUIRE_THROWS_AS(tests::load_fixture_buffer<float>(root, manifest, "input_a"), std::runtime_error);

  manifest = tests::load_fixture_manifest(root / "manifest.json");
  manifest.buffers.at("input_a").shape = {};
  REQUIRE_THROWS_AS(tests::load_fixture_buffer<float>(root, manifest, "input_a"), std::runtime_error);

  manifest = tests::load_fixture_manifest(root / "manifest.json");
  manifest.buffers.at("input_a").shape = {5};
  REQUIRE_THROWS_AS(tests::load_fixture_buffer<float>(root, manifest, "input_a"), std::runtime_error);
}

TEST_CASE("Typed fixture loader preserves little-endian int32 values", "[host]") {
  const auto root = tests::fixture_dir("sum");
  const auto manifest = tests::load_fixture_manifest(root / "manifest.json");

  const auto bytes = tests::load_binary_file(root / "input.bin");
  const std::vector<std::uint8_t> expected_bytes = {
      1, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0,
  };
  REQUIRE(bytes == expected_bytes);

  const auto values = tests::load_fixture_buffer<std::int32_t>(root, manifest, "input");
  REQUIRE(values == std::vector<std::int32_t>{1, 0, 2, 3});
}

TEST_CASE("Fixture manifest loader rejects malformed JSON", "[host]") {
  const auto manifest_file = write_temp_manifest("{not json");
  REQUIRE_THROWS(tests::load_fixture_manifest(manifest_file.path));
}

TEST_CASE("Fixture manifest loader defaults profile_agnostic to false", "[host]") {
  const auto manifest_file = write_temp_manifest(R"({
    "ref_baseline_tag": "ref-baseline-2026-06-12",
    "stage_name": "profile_default",
    "subgraph": "test",
    "bindings": [],
    "buffers": {},
    "shapes": {},
    "dtypes": {},
    "cmake_preset": "windows-debug",
    "emulate_int64": 0,
    "emulate_f32_atomic": 0
  })");

  const auto manifest = tests::load_fixture_manifest(manifest_file.path);
  REQUIRE_FALSE(manifest.profile_agnostic);
}

TEST_CASE("Fixture manifest loader rejects invalid dtype and shape metadata", "[host]") {
  const auto invalid_dtype = write_temp_manifest(R"({
    "ref_baseline_tag": "ref-baseline-2026-06-12",
    "stage_name": "bad",
    "subgraph": "test",
    "bindings": ["input"],
    "buffers": {"input": "input.bin"},
    "shapes": {"input": [1]},
    "dtypes": {"input": "float16"},
    "cmake_preset": "windows-debug",
    "emulate_int64": 0,
    "emulate_f32_atomic": 0
  })");

  REQUIRE_THROWS_AS(tests::load_fixture_manifest(invalid_dtype.path), std::invalid_argument);

  const auto empty_shape = write_temp_manifest(R"({
    "ref_baseline_tag": "ref-baseline-2026-06-12",
    "stage_name": "bad",
    "subgraph": "test",
    "bindings": ["input"],
    "buffers": {"input": "input.bin"},
    "shapes": {"input": []},
    "dtypes": {"input": "int32"},
    "cmake_preset": "windows-debug",
    "emulate_int64": 0,
    "emulate_f32_atomic": 0
  })");
  REQUIRE_THROWS_AS(tests::load_fixture_manifest(empty_shape.path), std::runtime_error);
}
