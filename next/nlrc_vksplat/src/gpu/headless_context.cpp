#include "gpu/headless_context.hpp"

#include <stdexcept>
#include <vector>

#include "gpu/constants.hpp"
#include "vulkan_check.hpp"

namespace nlrc::vksplat::gpu {
namespace {

[[nodiscard]] bool has_compute_queue(VkPhysicalDevice physical_device, uint32_t &out_family) {
  uint32_t family_count = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, nullptr);

  std::vector<VkQueueFamilyProperties> families(family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &family_count, families.data());

  for (uint32_t i = 0; i < family_count; ++i) {
    if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U) {
      out_family = i;
      return true;
    }
  }

  return false;
}

[[nodiscard]] int device_score(VkPhysicalDevice physical_device) {
  VkPhysicalDeviceProperties props{};
  vkGetPhysicalDeviceProperties(physical_device, &props);

  switch (props.deviceType) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
      return kDiscreteGpuScore;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
      return kIntegratedGpuScore;
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
      return kCpuDeviceScore;
    default:
      return kUnsupportedDeviceScore;
  }
}

[[nodiscard]] VkPhysicalDevice pick_physical_device(VkInstance instance) {
  uint32_t count = 0;
  check_vk(vkEnumeratePhysicalDevices(instance, &count, nullptr), "vkEnumeratePhysicalDevices(count)");

  if (count == 0) {
    throw std::runtime_error("No Vulkan physical devices found");
  }

  std::vector<VkPhysicalDevice> devices(count);
  check_vk(vkEnumeratePhysicalDevices(instance, &count, devices.data()), "vkEnumeratePhysicalDevices(list)");

  VkPhysicalDevice best = {};
  int best_score = kNoDeviceScore;
  uint32_t family = 0;

  for (VkPhysicalDevice device : devices) {
    if (!has_compute_queue(device, family)) {
      continue;
    }

    const int score = device_score(device);
    if (score > best_score) {
      best_score = score;
      best = device;
    }
  }

  if (best == VK_NULL_HANDLE) {
    throw std::runtime_error("No Vulkan device with a compute queue found");
  }

  return best;
}

} // namespace

bool probe_compute_device() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = kProbeApplicationName;
  app_info.apiVersion = kVulkanApiVersion;

  VkInstanceCreateInfo create_info{};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;

  VkInstance instance = VK_NULL_HANDLE;
  if (vkCreateInstance(&create_info, nullptr, &instance) != VK_SUCCESS) {
    return false;
  }

  bool ok = false;
  try {
    uint32_t family = 0;
    const VkPhysicalDevice device = pick_physical_device(instance);
    ok = has_compute_queue(device, family);
  } catch (...) {
    ok = false;
  }

  vkDestroyInstance(instance, nullptr);
  return ok;
}

HeadlessContext::HeadlessContext() {
  VkApplicationInfo app_info{};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.pApplicationName = kApplicationName;
  app_info.apiVersion = kVulkanApiVersion;

  VkInstanceCreateInfo instance_info{};
  instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.pApplicationInfo = &app_info;
  check_vk(vkCreateInstance(&instance_info, nullptr, &instance_), "vkCreateInstance");

  physical_device_ = pick_physical_device(instance_);
  if (!has_compute_queue(physical_device_, compute_queue_family_)) {
    throw std::runtime_error("Selected device has no compute queue");
  }

  const float queue_priority = kQueuePriority;
  VkDeviceQueueCreateInfo queue_info{};
  queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_info.queueFamilyIndex = compute_queue_family_;
  queue_info.queueCount = kQueueCount;
  queue_info.pQueuePriorities = &queue_priority;

  VkDeviceCreateInfo device_info{};
  device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_info.queueCreateInfoCount = kQueueCount;
  device_info.pQueueCreateInfos = &queue_info;

  check_vk(vkCreateDevice(physical_device_, &device_info, nullptr, &device_), "vkCreateDevice");
  vkGetDeviceQueue(device_, compute_queue_family_, kQueueIndex, &compute_queue_);

  VkCommandPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
  pool_info.queueFamilyIndex = compute_queue_family_;
  check_vk(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_), "vkCreateCommandPool");

  VkFenceCreateInfo fence_info{};
  fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  check_vk(vkCreateFence(device_, &fence_info, nullptr, &fence_), "vkCreateFence");
}

HeadlessContext::~HeadlessContext() {
  if (device_ != VK_NULL_HANDLE) {
    vkDeviceWaitIdle(device_);
  }
  if (fence_ != VK_NULL_HANDLE) {
    vkDestroyFence(device_, fence_, nullptr);
  }
  if (command_pool_ != VK_NULL_HANDLE) {
    vkDestroyCommandPool(device_, command_pool_, nullptr);
  }
  if (device_ != VK_NULL_HANDLE) {
    vkDestroyDevice(device_, nullptr);
  }
  if (instance_ != VK_NULL_HANDLE) {
    vkDestroyInstance(instance_, nullptr);
  }
}

void HeadlessContext::wait_idle() const {
  check_vk(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle");
}

void HeadlessContext::submit_and_wait(VkCommandBuffer command_buffer) const {
  check_vk(vkResetFences(device_, kSingleSubmitCount, &fence_), "vkResetFences");

  VkSubmitInfo submit_info{};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = kSingleCommandBufferCount;
  submit_info.pCommandBuffers = &command_buffer;
  check_vk(vkQueueSubmit(compute_queue_, kSingleSubmitCount, &submit_info, fence_), "vkQueueSubmit");
  check_vk(vkWaitForFences(device_, kSingleSubmitCount, &fence_, VK_TRUE, kFenceWaitTimeout), "vkWaitForFences");
}

} // namespace nlrc::vksplat::gpu
