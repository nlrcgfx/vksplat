#include "gpu/compute_pipeline.hpp"

#include <cstring>
#include <stdexcept>

#include "vulkan_check.hpp"

namespace nlrc::vksplat::gpu {

ComputePipeline::ComputePipeline(const HeadlessContext &context,
                                 const uint32_t *spirv_code,
                                 std::size_t spirv_word_count,
                                 std::uint32_t storage_buffer_count,
                                 std::size_t push_constant_size)
  : context_(&context), storage_buffer_count_(storage_buffer_count), push_constant_size_(push_constant_size) {
  if (push_constant_size > 192) {
    throw std::invalid_argument("Push constant size exceeds 192 bytes");
  }

  VkShaderModuleCreateInfo module_info{};
  module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  module_info.codeSize = spirv_word_count * sizeof(uint32_t);
  module_info.pCode = spirv_code;
  check_vk(vkCreateShaderModule(context.device(), &module_info, nullptr, &shader_module_), "vkCreateShaderModule");

  std::vector<VkDescriptorSetLayoutBinding> bindings(storage_buffer_count);
  for (std::uint32_t i = 0; i < storage_buffer_count; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = 1;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
  layout_info.pBindings = bindings.empty() ? nullptr : bindings.data();

  // clang-format off
  check_vk(
    vkCreateDescriptorSetLayout(context.device(), &layout_info, nullptr, &descriptor_set_layout_),
    "vkCreateDescriptorSetLayout"
  );
  // clang-format on

  const VkPushConstantRange push_range{VK_SHADER_STAGE_COMPUTE_BIT, 0, static_cast<uint32_t>(push_constant_size)};

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = storage_buffer_count > 0 ? 1U : 0U;
  pipeline_layout_info.pSetLayouts = storage_buffer_count > 0 ? &descriptor_set_layout_ : nullptr;

  if (push_constant_size > 0) {
    pipeline_layout_info.pushConstantRangeCount = 1;
    pipeline_layout_info.pPushConstantRanges = &push_range;
  }

  // clang-format off
  check_vk(
    vkCreatePipelineLayout(context.device(), &pipeline_layout_info, nullptr, &pipeline_layout_),
    "vkCreatePipelineLayout"
  );
  // clang-format on

  const VkPipelineShaderStageCreateInfo stage_info{
      VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
      nullptr,
      0,
      VK_SHADER_STAGE_COMPUTE_BIT,
      shader_module_,
      "main",
      nullptr,
  };

  const VkComputePipelineCreateInfo pipeline_info{
      VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, nullptr, 0, stage_info, pipeline_layout_, VK_NULL_HANDLE, 0,
  };

  // clang-format off
  check_vk(
    vkCreateComputePipelines(context.device(), VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline_),
    "vkCreateComputePipelines"
  );
  // clang-format on

  if (storage_buffer_count > 0) {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = storage_buffer_count;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    // clang-format off
    check_vk(
      vkCreateDescriptorPool(context.device(), &pool_info, nullptr, &descriptor_pool_),
    "vkCreateDescriptorPool"
    );
    // clang-format on

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = 1;
    alloc_info.pSetLayouts = &descriptor_set_layout_;
    check_vk(vkAllocateDescriptorSets(context.device(), &alloc_info, &descriptor_set_), "vkAllocateDescriptorSets");
  }

  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = context.command_pool();
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = 1;
  check_vk(vkAllocateCommandBuffers(context.device(), &cmd_alloc, &command_buffer_), "vkAllocateCommandBuffers");
}

ComputePipeline::~ComputePipeline() {
  destroy();
}

void ComputePipeline::destroy() noexcept {
  if (context_ != nullptr && command_buffer_ != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(context_->device(), context_->command_pool(), 1, &command_buffer_);
  }
  if (context_ != nullptr && pipeline_ != VK_NULL_HANDLE) {
    vkDestroyPipeline(context_->device(), pipeline_, nullptr);
  }
  if (context_ != nullptr && pipeline_layout_ != VK_NULL_HANDLE) {
    vkDestroyPipelineLayout(context_->device(), pipeline_layout_, nullptr);
  }
  if (context_ != nullptr && descriptor_pool_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorPool(context_->device(), descriptor_pool_, nullptr);
  }
  if (context_ != nullptr && descriptor_set_layout_ != VK_NULL_HANDLE) {
    vkDestroyDescriptorSetLayout(context_->device(), descriptor_set_layout_, nullptr);
  }
  if (context_ != nullptr && shader_module_ != VK_NULL_HANDLE) {
    vkDestroyShaderModule(context_->device(), shader_module_, nullptr);
  }
  context_ = nullptr;
  command_buffer_ = VK_NULL_HANDLE;
  pipeline_ = VK_NULL_HANDLE;
  pipeline_layout_ = VK_NULL_HANDLE;
  descriptor_pool_ = VK_NULL_HANDLE;
  descriptor_set_ = VK_NULL_HANDLE;
  descriptor_set_layout_ = VK_NULL_HANDLE;
  shader_module_ = VK_NULL_HANDLE;
}

void ComputePipeline::bind_storage_buffers(const std::vector<const StorageBuffer *> &buffers) {
  if (buffers.size() != storage_buffer_count_) {
    throw std::invalid_argument("Storage buffer count mismatch");
  }

  std::vector<VkDescriptorBufferInfo> buffer_infos(buffers.size());
  std::vector<VkWriteDescriptorSet> writes(buffers.size());
  for (std::size_t i = 0; i < buffers.size(); ++i) {
    buffer_infos[i].buffer = buffers[i]->buffer();
    buffer_infos[i].offset = 0;
    buffer_infos[i].range = buffers[i]->size_bytes();

    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptor_set_;
    writes[i].dstBinding = static_cast<uint32_t>(i);
    writes[i].dstArrayElement = 0;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].descriptorCount = 1;
    writes[i].pBufferInfo = &buffer_infos[i];
  }

  vkUpdateDescriptorSets(context_->device(), static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void ComputePipeline::dispatch(std::uint32_t groups_x,
                               std::uint32_t groups_y,
                               std::uint32_t groups_z,
                               const void *push_constants,
                               std::size_t push_constant_size) {
  if (push_constant_size != push_constant_size_) {
    throw std::invalid_argument("Push constant size mismatch");
  }

  check_vk(vkResetCommandBuffer(command_buffer_, 0), "vkResetCommandBuffer");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check_vk(vkBeginCommandBuffer(command_buffer_, &begin_info), "vkBeginCommandBuffer");

  vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  if (storage_buffer_count_ > 0) {
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0, 1, &descriptor_set_,
                            0, nullptr);
  }
  if (push_constants != nullptr && push_constant_size > 0) {
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       static_cast<uint32_t>(push_constant_size), push_constants);
  }
  vkCmdDispatch(command_buffer_, groups_x, groups_y, groups_z);
  check_vk(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

  context_->submit_and_wait(command_buffer_);
}

} // namespace nlrc::vksplat::gpu
