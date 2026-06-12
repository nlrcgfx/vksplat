#pragma once

#include "fixture_manifest.hpp"

namespace nlrc::vksplat::tests {

[[nodiscard]] bool manifest_matches_build_profile(const FixtureManifest &manifest);

} // namespace nlrc::vksplat::tests
