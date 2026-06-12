#include "gpu/compute_pipeline.hpp"

#include <stdexcept>

#include "gpu/constants.hpp"
#include "nlrc_vksplat_config.hpp"
#include "vulkan_check.hpp"

namespace nlrc::vksplat::gpu {

ComputePipeline::ComputePipeline(const HeadlessContext &context,
                                 Span<const std::uint32_t> spirv_code,
                                 std::uint32_t storage_buffer_count,
                                 std::size_t push_constant_size)
  : context_(&context), storage_buffer_count_(storage_buffer_count), push_constant_size_(push_constant_size),
    storage_buffers_bound_(storage_buffer_count == 0) {
  validate_constructor_args(spirv_code);
  create_shader_module(spirv_code);
  create_descriptor_set_layout();
  create_pipeline_layout();
  create_compute_pipeline();
  create_descriptor_pool_and_set();
  allocate_command_buffer();
}

ComputePipeline::~ComputePipeline() {
  destroy();
}

void ComputePipeline::validate_constructor_args(Span<const std::uint32_t> spirv_code) const {
  if (spirv_code.empty()) {
    throw std::invalid_argument("SPIR-V bytecode is empty");
  }
  if (spirv_code.data() == nullptr) {
    throw std::invalid_argument("SPIR-V bytecode data is null");
  }
  if (push_constant_size_ > kMaxPushConstantBytes) {
    throw std::invalid_argument("Push constant size exceeds kMaxPushConstantBytes");
  }
}

void ComputePipeline::create_shader_module(Span<const std::uint32_t> spirv_code) {
  VkShaderModuleCreateInfo module_info{};
  module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
  module_info.codeSize = spirv_code.size_bytes();
  module_info.pCode = spirv_code.data();
  check_vk(vkCreateShaderModule(context_->device(), &module_info, nullptr, &shader_module_), "vkCreateShaderModule");
}

void ComputePipeline::create_descriptor_set_layout() {
  std::vector<VkDescriptorSetLayoutBinding> bindings(storage_buffer_count_);
  for (std::uint32_t i = 0; i < storage_buffer_count_; ++i) {
    bindings[i].binding = i;
    bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[i].descriptorCount = kDescriptorCountPerBinding;
    bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  }

  VkDescriptorSetLayoutCreateInfo layout_info{};
  layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  layout_info.bindingCount = static_cast<std::uint32_t>(bindings.size());
  layout_info.pBindings = bindings.empty() ? nullptr : bindings.data();

  check_vk(vkCreateDescriptorSetLayout(context_->device(), &layout_info, nullptr, &descriptor_set_layout_),
           "vkCreateDescriptorSetLayout");
}

void ComputePipeline::create_pipeline_layout() {
  const VkPushConstantRange push_range{
      VK_SHADER_STAGE_COMPUTE_BIT,
      kPushConstantOffset,
      static_cast<std::uint32_t>(push_constant_size_),
  };

  VkPipelineLayoutCreateInfo pipeline_layout_info{};
  pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_info.setLayoutCount = storage_buffer_count_ > 0 ? kDescriptorSetCount : 0U;
  pipeline_layout_info.pSetLayouts = storage_buffer_count_ > 0 ? &descriptor_set_layout_ : nullptr;

  if (push_constant_size_ > 0) {
    pipeline_layout_info.pushConstantRangeCount = kPushConstantRangeCount;
    pipeline_layout_info.pPushConstantRanges = &push_range;
  }

  check_vk(vkCreatePipelineLayout(context_->device(), &pipeline_layout_info, nullptr, &pipeline_layout_),
           "vkCreatePipelineLayout");
}

void ComputePipeline::create_compute_pipeline() {
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

  check_vk(vkCreateComputePipelines(context_->device(), VK_NULL_HANDLE, kSinglePipelineCreateCount, &pipeline_info,
                                    nullptr, &pipeline_),
           "vkCreateComputePipelines");
}

void ComputePipeline::create_descriptor_pool_and_set() {
  if (storage_buffer_count_ == 0) {
    return;
  }

  VkDescriptorPoolSize pool_size{};
  pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  pool_size.descriptorCount = storage_buffer_count_;

  VkDescriptorPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info.maxSets = kDescriptorPoolMaxSets;
  pool_info.poolSizeCount = kDescriptorPoolSizeCount;
  pool_info.pPoolSizes = &pool_size;

  check_vk(vkCreateDescriptorPool(context_->device(), &pool_info, nullptr, &descriptor_pool_),
           "vkCreateDescriptorPool");

  VkDescriptorSetAllocateInfo alloc_info{};
  alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  alloc_info.descriptorPool = descriptor_pool_;
  alloc_info.descriptorSetCount = kDescriptorSetCount;
  alloc_info.pSetLayouts = &descriptor_set_layout_;
  check_vk(vkAllocateDescriptorSets(context_->device(), &alloc_info, &descriptor_set_), "vkAllocateDescriptorSets");
}

void ComputePipeline::allocate_command_buffer() {
  VkCommandBufferAllocateInfo cmd_alloc{};
  cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  cmd_alloc.commandPool = context_->command_pool();
  cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  cmd_alloc.commandBufferCount = kSingleCommandBufferCount;
  check_vk(vkAllocateCommandBuffers(context_->device(), &cmd_alloc, &command_buffer_), "vkAllocateCommandBuffers");
}

void ComputePipeline::destroy() noexcept {
  if (context_ != nullptr && command_buffer_ != VK_NULL_HANDLE) {
    vkFreeCommandBuffers(context_->device(), context_->command_pool(), kSingleCommandBufferCount, &command_buffer_);
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
  storage_buffers_bound_ = storage_buffer_count_ == 0;
}

void ComputePipeline::bind_storage_buffers(StorageBindingList bindings) {
  if (bindings.size() != storage_buffer_count_) {
    throw std::invalid_argument("Storage buffer count mismatch");
  }
  if (bindings.empty()) {
    storage_buffers_bound_ = true;
    return;
  }

  std::vector<VkDescriptorBufferInfo> buffer_infos(bindings.size());
  std::vector<VkWriteDescriptorSet> writes(bindings.size());
  for (std::size_t i = 0; i < bindings.size(); ++i) {
    const StorageBuffer *buffer = bindings[i];
    if (buffer == nullptr) {
      throw std::invalid_argument("Storage buffer binding is null");
    }

    buffer_infos[i].buffer = buffer->buffer();
    buffer_infos[i].offset = 0;
    buffer_infos[i].range = buffer->size_bytes();

    writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[i].dstSet = descriptor_set_;
    writes[i].dstBinding = static_cast<std::uint32_t>(i);
    writes[i].dstArrayElement = kDescriptorArrayElement;
    writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    writes[i].descriptorCount = kDescriptorCountPerBinding;
    writes[i].pBufferInfo = &buffer_infos[i];
  }

  vkUpdateDescriptorSets(context_->device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0, nullptr);
  storage_buffers_bound_ = true;
}

void ComputePipeline::dispatch(DispatchShape shape, PushConstantsView push_constants) {
  if (push_constants.size_bytes() != push_constant_size_) {
    throw std::invalid_argument("Push constant size mismatch");
  }
  if (push_constants.data() == nullptr && !push_constants.empty()) {
    throw std::invalid_argument("Push constant data is null");
  }
  if (storage_buffer_count_ > 0 && !storage_buffers_bound_) {
    throw std::logic_error("Storage buffers must be bound before dispatch");
  }

  check_vk(vkResetCommandBuffer(command_buffer_, kNoCommandBufferResetFlags), "vkResetCommandBuffer");

  VkCommandBufferBeginInfo begin_info{};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  check_vk(vkBeginCommandBuffer(command_buffer_, &begin_info), "vkBeginCommandBuffer");

  vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_);
  if (storage_buffer_count_ > 0) {
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, kDescriptorSetIndex,
                            kDescriptorSetCount, &descriptor_set_, kNoDynamicOffsetCount, nullptr);
  }
  if (!push_constants.empty()) {
    vkCmdPushConstants(command_buffer_, pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, kPushConstantOffset,
                       static_cast<std::uint32_t>(push_constants.size_bytes()), push_constants.data());
  }
  vkCmdDispatch(command_buffer_, shape.groups_x, shape.groups_y, shape.groups_z);
  check_vk(vkEndCommandBuffer(command_buffer_), "vkEndCommandBuffer");

  context_->submit_and_wait(command_buffer_);
}

} // namespace nlrc::vksplat::gpu
