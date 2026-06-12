#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "manifest_profile.hpp"

TEST_CASE("harness_smoke manifest matches build profile", "[host]") {
  const auto root = nlrc::vksplat::tests::fixture_dir("harness_smoke");
  const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(root / "manifest.json");
  REQUIRE(nlrc::vksplat::tests::manifest_matches_build_profile(manifest));
}
