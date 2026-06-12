#include "manifest_profile.hpp"

namespace nlrc::vksplat::tests {

bool manifest_matches_build_profile(const FixtureManifest &manifest) {
  if (manifest.profile_agnostic) {
    return true;
  }
  return manifest.emulate_int64 == VKSPLAT_USE_EMULATED_INT64 &&
         manifest.emulate_f32_atomic == VKSPLAT_USE_EMULATED_F32_ATOMIC;
}

} // namespace nlrc::vksplat::tests
