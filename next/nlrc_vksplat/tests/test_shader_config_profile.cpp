#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "nlrc_vksplat_config.hpp"

#include <nlohmann/json.hpp>

using namespace nlrc::vksplat;

namespace {

struct ExpectedInt final {
  const char *key;
  int actual;
  int expected;
};

struct ExpectedUint32 final {
  const char *key;
  std::uint32_t actual;
  std::uint32_t expected;
};

[[nodiscard]] nlohmann::json load_shader_config() {
  std::ifstream input(NLRC_VKSPLAT_SHADER_CONFIG_JSON);
  if (!input.is_open()) {
    throw std::runtime_error("could not open generated shader config");
  }
  return nlohmann::json::parse(input);
}

[[nodiscard]] std::vector<std::string> find_config_mismatches(const nlohmann::json &config) {
  const std::array expected_ints = {
      ExpectedInt{"use_emulated_int64", config.at("use_emulated_int64").get<int>(), VKSPLAT_USE_EMULATED_INT64},
      ExpectedInt{"use_emulated_int64 constexpr", config.at("use_emulated_int64").get<int>(),
                  static_cast<int>(kUseEmulatedInt64)},
      ExpectedInt{"use_emulated_f32_atomic", config.at("use_emulated_f32_atomic").get<int>(),
                  VKSPLAT_USE_EMULATED_F32_ATOMIC},
      ExpectedInt{"use_emulated_f32_atomic constexpr", config.at("use_emulated_f32_atomic").get<int>(),
                  static_cast<int>(kUseEmulatedF32Atomic)},
      ExpectedInt{"shader_requires_int64", config.at("shader_requires_int64").get<int>(),
                  static_cast<int>(kShaderRequiresInt64)},
  };
  const std::array expected_uint32s = {
      ExpectedUint32{"rect_tile_space_words", config.at("rect_tile_space_words").get<std::uint32_t>(),
                     kRectTileSpaceWords},
      ExpectedUint32{"tile_size", config.at("tile_size").get<std::uint32_t>(), kTileSize},
      ExpectedUint32{"subgroup_size", config.at("subgroup_size").get<std::uint32_t>(), kSubgroupSize},
      ExpectedUint32{"sorting_key_bits", config.at("sorting_key_bits").get<std::uint32_t>(), kSortingKeyBits},
  };

  std::vector<std::string> mismatches;
  for (const auto &expected : expected_ints) {
    if (expected.actual != expected.expected) {
      mismatches.emplace_back(expected.key);
    }
  }
  for (const auto &expected : expected_uint32s) {
    if (expected.actual != expected.expected) {
      mismatches.emplace_back(expected.key);
    }
  }
  return mismatches;
}

} // namespace

TEST_CASE("Generated shader config matches C++ build profile", "[host]") {
  const auto mismatches = find_config_mismatches(load_shader_config());
  REQUIRE(mismatches.empty());
}
