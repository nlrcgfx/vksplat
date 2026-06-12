#pragma once

namespace nlrc::vksplat::tests {

enum class GpuTestPolicy : int {
  kAuto = 0,
  kRequire = 1,
  kOff = 2,
};

[[nodiscard]] GpuTestPolicy gpu_test_policy();
[[nodiscard]] bool gpu_available();
void require_gpu_for_test();

#define NLRC_REQUIRE_GPU() ::nlrc::vksplat::tests::require_gpu_for_test()

} // namespace nlrc::vksplat::tests
