#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "gpu/headless_context.hpp"
#include "gpu/storage_buffer.hpp"

namespace nlrc::vksplat::gpu {

class ComputePipeline final {
public:
  ComputePipeline(const HeadlessContext &context,
                  const uint32_t *spirv_code,
                  std::size_t spirv_word_count,
                  std::uint32_t storage_buffer_count = 0,
                  std::size_t push_constant_size = 0);

  ~ComputePipeline();

  ComputePipeline(const ComputePipeline &) = delete;
  ComputePipeline &operator=(const ComputePipeline &) = delete;

  ComputePipeline(ComputePipeline &&) = delete;
  ComputePipeline &operator=(ComputePipeline &&) = delete;

  void bind_storage_buffers(const std::vector<const StorageBuffer *> &buffers);

  void dispatch(std::uint32_t groups_x,
                std::uint32_t groups_y = 1,
                std::uint32_t groups_z = 1,
                const void *push_constants = nullptr,
                std::size_t push_constant_size = 0);

private:
  void destroy() noexcept;

  const HeadlessContext *context_{};
  VkShaderModule shader_module_{};
  VkDescriptorSetLayout descriptor_set_layout_{};
  VkPipelineLayout pipeline_layout_{};
  VkPipeline pipeline_{};
  VkDescriptorPool descriptor_pool_{};
  VkDescriptorSet descriptor_set_{};
  VkCommandBuffer command_buffer_{};
  std::uint32_t storage_buffer_count_{};
  std::size_t push_constant_size_{};
};

} // namespace nlrc::vksplat::gpu
