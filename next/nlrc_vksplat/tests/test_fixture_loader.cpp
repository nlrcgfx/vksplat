#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"

TEST_CASE("Load harness_smoke fixture manifest and buffer", "[host]") {
  const auto root = nlrc::vksplat::tests::fixture_dir("harness_smoke");
  const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(root / "manifest.json");
  REQUIRE(manifest.stage_name == "harness_smoke");
  REQUIRE(manifest.bindings.empty());

  const auto values = nlrc::vksplat::tests::load_fixture_float_buffer(root, manifest, "input_a");
  REQUIRE(values.size() == 4);
  REQUIRE(values[0] == 1.0F);
  REQUIRE(values[3] == 4.0F);
}
