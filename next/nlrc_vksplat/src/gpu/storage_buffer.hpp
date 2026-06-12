#pragma once

#include <cstddef>
#include <vector>

#include <vulkan/vulkan.h>

#include "gpu/headless_context.hpp"
#include "span.hpp"

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

  void upload(ByteView data);

  void read_back(MutableByteView data) const;

  template <typename T>
  void upload(Span<const T> data) {
    upload(ByteView(data));
  }

  template <typename T>
  void read_back(Span<T> data) const {
    read_back(MutableByteView(data));
  }

  template <typename T>
  void upload(const std::vector<T> &data) {
    upload(make_span(data));
  }

  template <typename T>
  [[nodiscard]] std::vector<T> read_back(std::size_t count) const {
    std::vector<T> out(count);
    read_back(make_span(out));
    return out;
  }

private:
  void destroy() noexcept;

  VkDevice device_{};
  VkBuffer buffer_{};
  VkDeviceMemory memory_{};
  std::size_t size_bytes_{};
};

template <typename T>
[[nodiscard]] StorageBuffer make_storage_buffer(const HeadlessContext &context, Span<const T> values) {
  StorageBuffer buffer(context, values.size_bytes());
  buffer.upload(values);
  return buffer;
}

template <typename T>
[[nodiscard]] StorageBuffer make_storage_buffer(const HeadlessContext &context, const std::vector<T> &values) {
  return make_storage_buffer(context, make_span(values));
}

} // namespace nlrc::vksplat::gpu
