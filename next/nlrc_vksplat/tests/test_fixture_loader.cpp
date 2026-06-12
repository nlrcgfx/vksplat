#include <cstdint>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"

TEST_CASE("Load harness_smoke fixture manifest metadata", "[host]") {
  const auto root = nlrc::vksplat::tests::fixture_dir("harness_smoke");
  const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(root / "manifest.json");
  REQUIRE(manifest.stage_name == "harness_smoke");
  REQUIRE(manifest.bindings.empty());
}

TEST_CASE("Load harness_smoke fixture typed buffer spec", "[host]") {
  const auto root = nlrc::vksplat::tests::fixture_dir("harness_smoke");
  const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(root / "manifest.json");

  const auto &spec = manifest.buffers.at("input_a");
  REQUIRE(spec.file == "input_a.bin");
  REQUIRE(spec.shape == std::vector<std::size_t>{4});
  REQUIRE(spec.dtype == nlrc::vksplat::tests::BufferDtype::Float32);
}

TEST_CASE("Typed fixture loader loads float buffers", "[host]") {
  const auto root = nlrc::vksplat::tests::fixture_dir("harness_smoke");
  const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(root / "manifest.json");

  const auto values = nlrc::vksplat::tests::load_fixture_buffer<float>(root, manifest, "input_a");
  REQUIRE(values.size() == 4);
  REQUIRE(values[0] == 1.0F);
  REQUIRE(values[3] == 4.0F);
}

TEST_CASE("Fixture dtype parser maps supported manifest dtypes", "[host]") {
  using nlrc::vksplat::tests::BufferDtype;
  using nlrc::vksplat::tests::parse_buffer_dtype;

  REQUIRE(parse_buffer_dtype("float32") == BufferDtype::Float32);
  REQUIRE(parse_buffer_dtype("uint32") == BufferDtype::UInt32);
  REQUIRE(parse_buffer_dtype("int32") == BufferDtype::Int32);
  REQUIRE(parse_buffer_dtype("int64") == BufferDtype::Int64);
}

TEST_CASE("Fixture dtype helpers expose sizes and reject unknown types", "[host]") {
  using nlrc::vksplat::tests::buffer_dtype_size;
  using nlrc::vksplat::tests::BufferDtype;
  using nlrc::vksplat::tests::parse_buffer_dtype;

  REQUIRE(buffer_dtype_size(BufferDtype::Int64) == sizeof(std::int64_t));
  REQUIRE_THROWS_AS(parse_buffer_dtype("float16"), std::invalid_argument);
}

TEST_CASE("Typed fixture loader rejects dtype and size mismatches", "[host]") {
  const auto root = nlrc::vksplat::tests::fixture_dir("harness_smoke");
  auto manifest = nlrc::vksplat::tests::load_fixture_manifest(root / "manifest.json");

  REQUIRE_THROWS_AS(nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(root, manifest, "input_a"),
                    std::runtime_error);

  manifest.buffers.at("input_a").shape = {5};
  REQUIRE_THROWS_AS(nlrc::vksplat::tests::load_fixture_buffer<float>(root, manifest, "input_a"), std::runtime_error);
}
