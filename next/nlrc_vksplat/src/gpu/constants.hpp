#pragma once

#include <cstdint>
#include <limits>

#include <vulkan/vulkan.h>

namespace nlrc::vksplat::gpu {

inline constexpr const char *kApplicationName = "nlrc_vksplat";
inline constexpr const char *kProbeApplicationName = "nlrc_vksplat_probe";
inline constexpr std::uint32_t kVulkanApiVersion = VK_API_VERSION_1_2;

inline constexpr float kQueuePriority = 1.0F;
inline constexpr std::uint32_t kQueueIndex = 0;
inline constexpr std::uint32_t kQueueCount = 1;
inline constexpr std::uint64_t kFenceWaitTimeout = std::numeric_limits<std::uint64_t>::max();

inline constexpr int kDiscreteGpuScore = 3;
inline constexpr int kIntegratedGpuScore = 2;
inline constexpr int kCpuDeviceScore = 1;
inline constexpr int kUnsupportedDeviceScore = 0;
inline constexpr int kNoDeviceScore = -1;

inline constexpr std::uint32_t kDescriptorSetIndex = 0;
inline constexpr std::uint32_t kDescriptorSetCount = 1;
inline constexpr std::uint32_t kDescriptorCountPerBinding = 1;
inline constexpr std::uint32_t kDescriptorArrayElement = 0;
inline constexpr std::uint32_t kDescriptorPoolMaxSets = 1;
inline constexpr std::uint32_t kDescriptorPoolSizeCount = 1;
inline constexpr std::uint32_t kNoDynamicOffsetCount = 0;
inline constexpr std::uint32_t kNoCommandBufferResetFlags = 0;
inline constexpr std::uint32_t kPushConstantOffset = 0;
inline constexpr std::uint32_t kPushConstantRangeCount = 1;
inline constexpr std::uint32_t kSingleCommandBufferCount = 1;
inline constexpr std::uint32_t kSinglePipelineCreateCount = 1;
inline constexpr std::uint32_t kSingleSubmitCount = 1;

} // namespace nlrc::vksplat::gpu
