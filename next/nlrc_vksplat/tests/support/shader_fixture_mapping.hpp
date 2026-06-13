#pragma once

#include <string>
#include <vector>

#include "fixture_manifest.hpp"
#include "gpu/shader_descriptors.hpp"

namespace nlrc::vksplat::tests {

enum class FixtureManifestSide : std::uint8_t {
  Fixture,
  Golden,
};

struct FixtureBindingResolution final {
  std::string contract;
  std::vector<std::string> bindings;
  bool untracked{false};
};

[[nodiscard]] FixtureBindingResolution fixture_binding_resolution(const FixtureManifest &manifest,
                                                                  FixtureManifestSide side);

[[nodiscard]] const gpu::ShaderInterface *
shader_interface_for_fixture(const FixtureManifest &manifest, FixtureManifestSide side = FixtureManifestSide::Fixture);

[[nodiscard]] bool fixture_is_intentionally_untracked(const FixtureManifest &manifest);

} // namespace nlrc::vksplat::tests
