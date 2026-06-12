#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

#include "gpu/headless_context.hpp"
#include "gpu/storage_buffer.hpp"
#include "span.hpp"

namespace nlrc::vksplat::gpu {

struct DispatchShape final {
  std::uint32_t groups_x{1};
  std::uint32_t groups_y{1};
  std::uint32_t groups_z{1};
};

using PushConstantsView = ByteView;

class StorageBindingList final {
public:
  explicit constexpr StorageBindingList(Span<const StorageBuffer *const> buffers) noexcept : buffers_(buffers) {}

  [[nodiscard]] constexpr std::size_t size() const noexcept {
    return buffers_.size();
  }

  [[nodiscard]] constexpr bool empty() const noexcept {
    return buffers_.empty();
  }

  [[nodiscard]] constexpr const StorageBuffer *operator[](std::size_t index) const noexcept {
    return buffers_[index];
  }

private:
  Span<const StorageBuffer *const> buffers_;
};

class ComputePipeline final {
public:
  ComputePipeline(const HeadlessContext &context,
                  Span<const std::uint32_t> spirv_code,
                  std::uint32_t storage_buffer_count = 0,
                  std::size_t push_constant_size = 0);

  ~ComputePipeline();

  ComputePipeline(const ComputePipeline &) = delete;
  ComputePipeline &operator=(const ComputePipeline &) = delete;

  ComputePipeline(ComputePipeline &&) = delete;
  ComputePipeline &operator=(ComputePipeline &&) = delete;

  void bind_storage_buffers(StorageBindingList bindings);

  void bind_storage_buffers(const std::vector<const StorageBuffer *> &buffers) {
    bind_storage_buffers(StorageBindingList(make_span(buffers)));
  }

  void dispatch(DispatchShape shape, PushConstantsView push_constants = {});

private:
  void validate_constructor_args(Span<const std::uint32_t> spirv_code) const;
  void create_shader_module(Span<const std::uint32_t> spirv_code);
  void create_descriptor_set_layout();
  void create_pipeline_layout();
  void create_compute_pipeline();
  void create_descriptor_pool_and_set();
  void allocate_command_buffer();
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
  bool storage_buffers_bound_{true};
};

} // namespace nlrc::vksplat::gpu
