#include "gpu/storage_buffer.hpp"

#include <cstring>
#include <stdexcept>

#include "vulkan_check.hpp"

namespace nlrc::vksplat::gpu {
namespace {

[[nodiscard]] uint32_t
find_memory_type(VkPhysicalDevice physical_device, uint32_t type_bits, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memory_props{};
  vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_props);

  // NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
  for (uint32_t index = 0; index < memory_props.memoryTypeCount; ++index) {
    if ((type_bits & (1U << index)) != 0U &&
        (memory_props.memoryTypes[index].propertyFlags & properties) == properties) {
      return index;
    }
  }
  // NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
  throw std::runtime_error("No suitable Vulkan memory type found");
}

} // namespace

StorageBuffer::StorageBuffer(const HeadlessContext &context, std::size_t size_bytes)
  : device_(context.device()), size_bytes_(size_bytes) {
  if (size_bytes == 0) {
    throw std::invalid_argument("StorageBuffer size must be > 0");
  }

  VkBufferCreateInfo buffer_info{};
  buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_info.size = size_bytes;
  buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  check_vk(vkCreateBuffer(device_, &buffer_info, nullptr, &buffer_), "vkCreateBuffer");

  VkMemoryRequirements requirements{};
  vkGetBufferMemoryRequirements(device_, buffer_, &requirements);

  const VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
  const uint32_t memory_type = find_memory_type(context.physical_device(), requirements.memoryTypeBits, properties);

  VkMemoryAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = requirements.size;
  alloc_info.memoryTypeIndex = memory_type;
  check_vk(vkAllocateMemory(device_, &alloc_info, nullptr, &memory_), "vkAllocateMemory");
  check_vk(vkBindBufferMemory(device_, buffer_, memory_, 0), "vkBindBufferMemory");
}

StorageBuffer::~StorageBuffer() {
  destroy();
}

StorageBuffer::StorageBuffer(StorageBuffer &&other) noexcept
  : device_(other.device_), buffer_(other.buffer_), memory_(other.memory_), size_bytes_(other.size_bytes_) {
  other.device_ = VK_NULL_HANDLE;
  other.buffer_ = VK_NULL_HANDLE;
  other.memory_ = VK_NULL_HANDLE;
  other.size_bytes_ = 0;
}

StorageBuffer &StorageBuffer::operator=(StorageBuffer &&other) noexcept {
  if (this != &other) {
    destroy();
    device_ = other.device_;
    buffer_ = other.buffer_;
    memory_ = other.memory_;
    size_bytes_ = other.size_bytes_;
    other.device_ = VK_NULL_HANDLE;
    other.buffer_ = VK_NULL_HANDLE;
    other.memory_ = VK_NULL_HANDLE;
    other.size_bytes_ = 0;
  }
  return *this;
}

void StorageBuffer::destroy() noexcept {
  if (device_ != VK_NULL_HANDLE && buffer_ != VK_NULL_HANDLE) {
    vkDestroyBuffer(device_, buffer_, nullptr);
  }

  if (device_ != VK_NULL_HANDLE && memory_ != VK_NULL_HANDLE) {
    vkFreeMemory(device_, memory_, nullptr);
  }

  device_ = VK_NULL_HANDLE;
  buffer_ = VK_NULL_HANDLE;
  memory_ = VK_NULL_HANDLE;
  size_bytes_ = 0;
}

void StorageBuffer::upload(ByteView data) {
  if (data.size_bytes() > size_bytes_) {
    throw std::invalid_argument("StorageBuffer upload exceeds allocation");
  }
  if (data.data() == nullptr && !data.empty()) {
    throw std::invalid_argument("StorageBuffer upload data is null");
  }

  void *mapped = nullptr;
  check_vk(vkMapMemory(device_, memory_, 0, size_bytes_, 0, &mapped), "vkMapMemory");
  if (!data.empty()) {
    std::memcpy(mapped, data.data(), data.size_bytes());
  }
  vkUnmapMemory(device_, memory_);
}

void StorageBuffer::read_back(MutableByteView data) const {
  if (data.size_bytes() > size_bytes_) {
    throw std::invalid_argument("StorageBuffer read_back exceeds allocation");
  }
  if (data.data() == nullptr && !data.empty()) {
    throw std::invalid_argument("StorageBuffer read_back data is null");
  }

  void *mapped = nullptr;
  check_vk(vkMapMemory(device_, memory_, 0, size_bytes_, 0, &mapped), "vkMapMemory");
  if (!data.empty()) {
    std::memcpy(data.data(), mapped, data.size_bytes());
  }
  vkUnmapMemory(device_, memory_);
}

} // namespace nlrc::vksplat::gpu
