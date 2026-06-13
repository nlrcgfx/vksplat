#include "shader_fixture_mapping.hpp"

#include <string>
#include <string_view>

namespace nlrc::vksplat::tests {
namespace {

[[nodiscard]] bool has_prefix(const std::string &value, std::string_view prefix) {
  return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

} // namespace

const gpu::ShaderInterface *shader_interface_for_fixture(const FixtureManifest &manifest, FixtureManifestSide side) {
  if (side == FixtureManifestSide::Golden) {
    return nullptr;
  }

  if (manifest.subgraph == "projection" && has_prefix(manifest.stage_name, "projection_forward_")) {
    return &gpu::shader_interface("projection_forward");
  }
  if (manifest.subgraph == "generate_keys" && has_prefix(manifest.stage_name, "generate_keys_")) {
    return &gpu::shader_interface("generate_keys");
  }
  if (manifest.subgraph == "compute_tile_ranges" && has_prefix(manifest.stage_name, "compute_tile_ranges")) {
    return &gpu::shader_interface("compute_tile_ranges");
  }
  if (manifest.subgraph == "rasterize_forward" && has_prefix(manifest.stage_name, "rasterize_forward_")) {
    return &gpu::shader_interface("rasterize_forward");
  }
  if (manifest.subgraph == "utility") {
    if (manifest.stage_name == "sum" || has_prefix(manifest.stage_name, "sum_")) {
      return &gpu::shader_interface("sum");
    }
    if (manifest.stage_name == "where" || has_prefix(manifest.stage_name, "where_")) {
      return &gpu::shader_interface("where");
    }
    if (manifest.stage_name == "cumsum_single_pass" || has_prefix(manifest.stage_name, "cumsum_single_pass_")) {
      return &gpu::shader_interface("cumsum_single_pass");
    }
  }
  return nullptr;
}

const gpu::FixtureBindingContract *fixture_contract_for_manifest(const FixtureManifest &manifest,
                                                                 FixtureManifestSide side) {
  if (side == FixtureManifestSide::Golden) {
    if (manifest.subgraph == "projection" && has_prefix(manifest.stage_name, "projection_forward_")) {
      return &gpu::fixture_binding_contract("no_exact_golden_outputs");
    }
    if (manifest.subgraph == "generate_keys" && has_prefix(manifest.stage_name, "generate_keys_")) {
      return &gpu::fixture_binding_contract("generate_keys_output");
    }
    if (manifest.subgraph == "compute_tile_ranges" && has_prefix(manifest.stage_name, "compute_tile_ranges")) {
      return &gpu::fixture_binding_contract("compute_tile_ranges_output");
    }
    if (manifest.subgraph == "rasterize_forward" && has_prefix(manifest.stage_name, "rasterize_forward_")) {
      return &gpu::fixture_binding_contract("no_exact_golden_outputs");
    }
    if (manifest.subgraph == "radix_sort" && has_prefix(manifest.stage_name, "radix_sort_")) {
      return &gpu::fixture_binding_contract("radix_sort_output");
    }
    if (manifest.subgraph == "utility") {
      if (manifest.stage_name == "sum" || has_prefix(manifest.stage_name, "sum_")) {
        return &gpu::fixture_binding_contract("sum_output");
      }
      if (manifest.stage_name == "where" || has_prefix(manifest.stage_name, "where_")) {
        return &gpu::fixture_binding_contract("where_output");
      }
      if (has_prefix(manifest.stage_name, "cumsum_")) {
        return &gpu::fixture_binding_contract("cumsum_output");
      }
    }
    return nullptr;
  }

  if (manifest.subgraph == "radix_sort" && has_prefix(manifest.stage_name, "radix_sort_")) {
    return &gpu::fixture_binding_contract("radix_sort_pipeline");
  }
  if (manifest.subgraph == "utility" && manifest.stage_name == "cumsum_multi_block") {
    return &gpu::fixture_binding_contract("cumsum_multi_block");
  }
  if (manifest.subgraph == "utility" && manifest.stage_name == "cumsum_multi_block_two_level") {
    return &gpu::fixture_binding_contract("cumsum_multi_block_two_level");
  }
  return nullptr;
}

bool fixture_is_intentionally_untracked(const FixtureManifest &manifest) {
  return manifest.subgraph == "harness";
}

} // namespace nlrc::vksplat::tests
