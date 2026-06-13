#pragma once

#include "fixture_manifest.hpp"
#include "gpu/shader_descriptors.hpp"

namespace nlrc::vksplat::tests {

enum class FixtureManifestSide {
  Fixture,
  Golden,
};

[[nodiscard]] const gpu::ShaderInterface *
shader_interface_for_fixture(const FixtureManifest &manifest, FixtureManifestSide side = FixtureManifestSide::Fixture);

[[nodiscard]] const gpu::FixtureBindingContract *
fixture_contract_for_manifest(const FixtureManifest &manifest, FixtureManifestSide side = FixtureManifestSide::Fixture);

[[nodiscard]] bool fixture_is_intentionally_untracked(const FixtureManifest &manifest);

} // namespace nlrc::vksplat::tests
