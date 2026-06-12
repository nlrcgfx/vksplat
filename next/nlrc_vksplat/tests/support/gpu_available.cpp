#include "gpu_available.hpp"

#include <catch2/catch_test_macros.hpp>

#include "gpu/headless_context.hpp"

namespace nlrc::vksplat::tests {

GpuTestPolicy gpu_test_policy() {
#if NLRC_VKSPLAT_GPU_TEST_POLICY == 1
  return GpuTestPolicy::Require;
#elif NLRC_VKSPLAT_GPU_TEST_POLICY == 2
  return GpuTestPolicy::Off;
#else
  return GpuTestPolicy::Auto;
#endif
}

bool gpu_available() {
  static const bool available = nlrc::vksplat::gpu::probe_compute_device();
  return available;
}

void require_gpu_for_test() {
  const auto policy = gpu_test_policy();
  if (policy == GpuTestPolicy::Off) {
    SKIP("GPU tests disabled (NLRC_VKSPLAT_GPU_TESTS=OFF)");
  }
  if (!gpu_available()) {
    if (policy == GpuTestPolicy::Require) {
      FAIL("Vulkan compute device required but not available");
    }
    SKIP("No Vulkan compute device available");
  }
}

} // namespace nlrc::vksplat::tests
