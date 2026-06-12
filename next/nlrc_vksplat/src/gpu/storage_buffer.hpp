#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "gpu/headless_context.hpp"

namespace nlrc::vksplat::gpu {

class StorageBuffer final {
public:
  StorageBuffer(const HeadlessContext &context, std::size_t size_bytes);
  ~StorageBuffer();

  StorageBuffer(const StorageBuffer &) = delete;
  StorageBuffer &operator=(const StorageBuffer &) = delete;

  StorageBuffer(StorageBuffer &&other) noexcept;
  StorageBuffer &operator=(StorageBuffer &&other) noexcept;

  [[nodiscard]] VkBuffer buffer() const {
    return buffer_;
  }

  [[nodiscard]] std::size_t size_bytes() const {
    return size_bytes_;
  }

  void upload(const void *data, std::size_t size_bytes);

  void read_back(void *data, std::size_t size_bytes) const;

  template <typename T>
  void upload(const std::vector<T> &data) {
    upload(data.data(), data.size() * sizeof(T));
  }

  template <typename T>
  [[nodiscard]] std::vector<T> read_back(std::size_t count) const {
    std::vector<T> out(count);
    read_back(out.data(), count * sizeof(T));
    return out;
  }

private:
  void destroy() noexcept;

  VkDevice device_{};
  VkBuffer buffer_{};
  VkDeviceMemory memory_{};
  std::size_t size_bytes_{};
};

} // namespace nlrc::vksplat::gpu
