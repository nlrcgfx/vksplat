#pragma once

#include <cstdint>

namespace nlrc::vksplat::tests {

enum class GpuTestPolicy : std::uint8_t {
  Auto = 0U,
  Require = 1U,
  Off = 2U,
};

[[nodiscard]] GpuTestPolicy gpu_test_policy();

[[nodiscard]] bool gpu_available();

void require_gpu_for_test();

#define NLRC_REQUIRE_GPU() ::nlrc::vksplat::tests::require_gpu_for_test()

} // namespace nlrc::vksplat::tests
