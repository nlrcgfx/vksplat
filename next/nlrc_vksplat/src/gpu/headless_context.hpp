#pragma once

#include <cstdint>

#include <vulkan/vulkan.h>

namespace nlrc::vksplat::gpu {

class HeadlessContext final {
public:
  HeadlessContext();
  ~HeadlessContext();

  HeadlessContext(const HeadlessContext &) = delete;
  HeadlessContext &operator=(const HeadlessContext &) = delete;

  HeadlessContext(HeadlessContext &&) = delete;
  HeadlessContext &operator=(HeadlessContext &&) = delete;

  [[nodiscard]] VkInstance instance() const {
    return instance_;
  }

  [[nodiscard]] VkPhysicalDevice physical_device() const {
    return physical_device_;
  }

  [[nodiscard]] VkDevice device() const {
    return device_;
  }

  [[nodiscard]] VkQueue compute_queue() const {
    return compute_queue_;
  }

  [[nodiscard]] uint32_t compute_queue_family() const {
    return compute_queue_family_;
  }

  [[nodiscard]] VkCommandPool command_pool() const {
    return command_pool_;
  }

  void wait_idle() const;
  void submit_and_wait(VkCommandBuffer command_buffer) const;

private:
  VkInstance instance_{};

  VkPhysicalDevice physical_device_{};
  VkDevice device_{};

  VkQueue compute_queue_{};
  uint32_t compute_queue_family_{};

  VkCommandPool command_pool_{};
  VkFence fence_{};
};

[[nodiscard]] bool probe_compute_device();

} // namespace nlrc::vksplat::gpu
