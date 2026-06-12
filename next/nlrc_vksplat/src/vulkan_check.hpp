#pragma once

#include <stdexcept>
#include <string>

#include <vulkan/vulkan.h>

namespace nlrc::vksplat {

[[nodiscard]] constexpr const char *vk_result_string(VkResult result) {
  switch (result) {
    case VK_SUCCESS:
      return "VK_SUCCESS";
    case VK_NOT_READY:
      return "VK_NOT_READY";
    case VK_TIMEOUT:
      return "VK_TIMEOUT";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
      return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
      return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
      return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
      return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
      return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_LAYER_NOT_PRESENT:
      return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
      return "VK_ERROR_INCOMPATIBLE_DRIVER";
    default:
      return "VK_ERROR_UNKNOWN";
  }
}

inline void check_vk(VkResult result, const char *context) {
  if (result == VK_SUCCESS) {
    return;
  }

  // clang-format off
  throw std::runtime_error(
    std::string(context) + " failed: " +
    vk_result_string(result) + " (" + std::to_string(static_cast<int>(result)) + ")"
  );
  // clang-format on
}

} // namespace nlrc::vksplat
