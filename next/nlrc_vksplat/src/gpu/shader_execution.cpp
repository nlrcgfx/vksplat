#include "gpu/shader_execution.hpp"

#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "compute_tile_ranges_spirv.hpp"
#include "cumsum_add_block_offsets_spirv.hpp"
#include "cumsum_block_scan_spirv.hpp"
#include "cumsum_scan_block_sums_spirv.hpp"
#include "cumsum_single_pass_spirv.hpp"
#include "generate_keys_spirv.hpp"
#include "gpu/shader_binding_resolver.hpp"
#include "projection_forward_spirv.hpp"
#include "radix_sort_downsweep_spirv.hpp"
#include "radix_sort_spine_spirv.hpp"
#include "radix_sort_upsweep_spirv.hpp"
#include "rasterize_forward_spirv.hpp"
#include "span.hpp"
#include "sum_spirv.hpp"
#include "where_spirv.hpp"

namespace nlrc::vksplat::gpu {
namespace {

static_assert(VKSPLAT_SORTING_KEY_BITS == 32, "Update execute_sort when 64-bit keys are enabled.");

template <typename T>
[[nodiscard]] std::size_t required_byte_count(std::size_t count, const char *name) {
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
    throw std::invalid_argument(std::string(name) + " byte count overflows size_t");
  }
  return count * sizeof(T);
}

template <typename T>
void require_buffer_elements(const StorageBuffer &buffer, std::size_t count, const char *name) {
  const auto required_bytes = required_byte_count<T>(count, name);
  if (buffer.size_bytes() < required_bytes) {
    throw std::invalid_argument(std::string(name) + " buffer is too small");
  }
}

void require_binding(const StorageBuffer *buffer, const char *name) {
  if (buffer == nullptr) {
    throw std::invalid_argument(std::string(name) + " binding is null");
  }
}

void require_storage_bindings(const ShaderInterface &shader, Span<const NamedStorageBinding> bindings) {
  for (std::size_t index = 0; index < shader.bindings.size(); ++index) {
    const auto *buffer = storage_buffer_by_name(bindings, shader.bindings[index].name);
    require_binding(buffer, shader.bindings[index].name);
  }
}

template <ShaderId Id, std::size_t Index>
[[nodiscard]] const StorageBuffer &storage_binding_at(Span<const NamedStorageBinding> bindings) {
  constexpr const char *kName = shader_binding_name<Id, Index>();
  const auto *buffer = storage_buffer_by_name(bindings, kName);
  require_binding(buffer, kName);
  return *buffer;
}

std::uint32_t require_uint32_count(std::size_t count, const char *name) {
  if (count == 0) {
    throw std::invalid_argument(std::string(name) + " must be > 0");
  }
  if (count > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string(name) + " exceeds uint32_t");
  }
  return static_cast<std::uint32_t>(count);
}

template <typename T>
[[nodiscard]] StorageBuffer make_zero_buffer(const HeadlessContext &context, std::size_t count) {
  std::vector<T> values(count, T{});
  return make_storage_buffer(context, values);
}

void dispatch_with_element_count(ComputePipeline &pipeline,
                                 const ShaderInterface &shader,
                                 Span<const NamedStorageBinding> bindings,
                                 std::size_t element_count,
                                 std::uint32_t block_size) {
  require_storage_bindings(shader, bindings);
  pipeline.bind_storage_buffers(resolve_storage_bindings(shader, bindings));

  const ElementCountPushConstants push_constants{require_uint32_count(element_count, "element_count")};
  const auto push_constants_view = ByteView::from_object(push_constants);
  pipeline.dispatch(dispatch_groups_for(element_count, block_size), push_constants_view);
}

void dispatch_radix_sort_pass(ComputePipeline &pipeline,
                              const ShaderInterface &shader,
                              Span<const NamedStorageBinding> bindings,
                              DispatchShape dispatch_shape,
                              RadixSortPushConstants push_constants) {
  require_storage_bindings(shader, bindings);
  pipeline.bind_storage_buffers(resolve_storage_bindings(shader, bindings));

  const auto push_constants_view = ByteView::from_object(push_constants);
  pipeline.dispatch(dispatch_shape, push_constants_view);
}

[[nodiscard]] std::size_t projection_sh_coeff_float_count(std::size_t num_splats) {
  const auto reorder_groups = ceil_div(num_splats, static_cast<std::size_t>(VKSPLAT_SH_REORDER_SIZE));
  return reorder_groups * 12U * VKSPLAT_SH_REORDER_SIZE * 4U;
}

void validate_projection_forward_bindings(const ProjectionForwardBindings &bindings, std::size_t num_splats) {
  constexpr auto kShader = ShaderId::ProjectionForward;
  const auto named_bindings = projection_forward_storage_bindings(bindings);
  const auto binding_span = make_span(named_bindings);
  require_storage_bindings(shader_interface(kShader), binding_span);

  // clang-format off
  require_buffer_elements<float>(storage_binding_at<kShader, 0>(binding_span), num_splats * 3U,
                                 shader_binding_name<kShader, 0>());
  require_buffer_elements<float>(storage_binding_at<kShader, 1>(binding_span),
                                 projection_sh_coeff_float_count(num_splats), shader_binding_name<kShader, 1>());
  require_buffer_elements<float>(storage_binding_at<kShader, 2>(binding_span), num_splats * 4U,
                                 shader_binding_name<kShader, 2>());
  require_buffer_elements<float>(storage_binding_at<kShader, 3>(binding_span), num_splats * 4U,
                                 shader_binding_name<kShader, 3>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 4>(binding_span), num_splats,
                                        shader_binding_name<kShader, 4>());
  require_buffer_elements<RectTileSpace>(storage_binding_at<kShader, 5>(binding_span),
                                         num_splats * kRectTileSpaceWords,
                                         shader_binding_name<kShader, 5>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 6>(binding_span), num_splats,
                                        shader_binding_name<kShader, 6>());
  require_buffer_elements<float>(storage_binding_at<kShader, 7>(binding_span), num_splats * 2U,
                                 shader_binding_name<kShader, 7>());
  require_buffer_elements<float>(storage_binding_at<kShader, 8>(binding_span), num_splats,
                                 shader_binding_name<kShader, 8>());
  require_buffer_elements<float>(storage_binding_at<kShader, 9>(binding_span), num_splats * 4U,
                                 shader_binding_name<kShader, 9>());
  require_buffer_elements<float>(storage_binding_at<kShader, 10>(binding_span), num_splats * 3U,
                                 shader_binding_name<kShader, 10>());
  // clang-format on
}

void validate_generate_keys_bindings(const GenerateKeysBindings &bindings,
                                     std::size_t num_splats,
                                     std::size_t output_count) {
  constexpr auto kShader = ShaderId::GenerateKeys;
  const auto named_bindings = generate_keys_storage_bindings(bindings);
  const auto binding_span = make_span(named_bindings);
  require_storage_bindings(shader_interface(kShader), binding_span);

  // clang-format off
  require_buffer_elements<float>(storage_binding_at<kShader, 0>(binding_span), num_splats * 2U,
                                 shader_binding_name<kShader, 0>());
  require_buffer_elements<float>(storage_binding_at<kShader, 1>(binding_span), num_splats * 4U,
                                 shader_binding_name<kShader, 1>());
  require_buffer_elements<float>(storage_binding_at<kShader, 2>(binding_span), num_splats,
                                 shader_binding_name<kShader, 2>());
  require_buffer_elements<RectTileSpace>(storage_binding_at<kShader, 3>(binding_span),
                                         num_splats * kRectTileSpaceWords,
                                         shader_binding_name<kShader, 3>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 4>(binding_span), num_splats,
                                        shader_binding_name<kShader, 4>());
  require_buffer_elements<SortingKey>(storage_binding_at<kShader, 5>(binding_span), output_count,
                                      shader_binding_name<kShader, 5>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 6>(binding_span), output_count,
                                        shader_binding_name<kShader, 6>());
  // clang-format on
}

[[nodiscard]] std::size_t require_tile_count(const RendererUniforms &uniforms) {
  if (uniforms.grid_width == 0U) {
    throw std::invalid_argument("grid_width must be > 0");
  }
  if (uniforms.grid_height == 0U) {
    throw std::invalid_argument("grid_height must be > 0");
  }
  const auto grid_width = static_cast<std::size_t>(uniforms.grid_width);
  const auto grid_height = static_cast<std::size_t>(uniforms.grid_height);
  if (grid_height > std::numeric_limits<std::size_t>::max() / grid_width) {
    throw std::invalid_argument("tile count overflows size_t");
  }
  return grid_height * grid_width;
}

[[nodiscard]] std::size_t require_pixel_count(const RendererUniforms &uniforms) {
  if (uniforms.image_width == 0U) {
    throw std::invalid_argument("image_width must be > 0");
  }
  if (uniforms.image_height == 0U) {
    throw std::invalid_argument("image_height must be > 0");
  }
  const auto image_width = static_cast<std::size_t>(uniforms.image_width);
  const auto image_height = static_cast<std::size_t>(uniforms.image_height);
  if (image_height > std::numeric_limits<std::size_t>::max() / image_width) {
    throw std::invalid_argument("pixel count overflows size_t");
  }
  return image_height * image_width;
}

void validate_compute_tile_ranges_bindings(const ComputeTileRangesBindings &bindings,
                                           std::size_t num_indices,
                                           std::size_t num_tiles) {
  constexpr auto kShader = ShaderId::ComputeTileRanges;
  const auto named_bindings = compute_tile_ranges_storage_bindings(bindings);
  const auto binding_span = make_span(named_bindings);
  require_storage_bindings(shader_interface(kShader), binding_span);

  // clang-format off
  require_buffer_elements<SortingKey>(storage_binding_at<kShader, 0>(binding_span), num_indices,
                                      shader_binding_name<kShader, 0>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 1>(binding_span), num_tiles + 1U,
                                        shader_binding_name<kShader, 1>());
  // clang-format on
}

void validate_rasterize_forward_bindings(const RasterizeForwardBindings &bindings,
                                         std::size_t num_splats,
                                         std::size_t num_indices,
                                         std::size_t num_tiles,
                                         std::size_t num_pixels) {
  constexpr auto kShader = ShaderId::RasterizeForward;
  const auto named_bindings = rasterize_forward_storage_bindings(bindings);
  const auto binding_span = make_span(named_bindings);
  require_storage_bindings(shader_interface(kShader), binding_span);

  // clang-format off
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 0>(binding_span), num_indices,
                                        shader_binding_name<kShader, 0>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 1>(binding_span), num_tiles + 1U,
                                        shader_binding_name<kShader, 1>());
  require_buffer_elements<float>(storage_binding_at<kShader, 2>(binding_span), num_splats * 2U,
                                 shader_binding_name<kShader, 2>());
  require_buffer_elements<float>(storage_binding_at<kShader, 3>(binding_span), num_splats * 4U,
                                 shader_binding_name<kShader, 3>());
  require_buffer_elements<float>(storage_binding_at<kShader, 4>(binding_span), num_splats * 3U,
                                 shader_binding_name<kShader, 4>());
  require_buffer_elements<float>(storage_binding_at<kShader, 5>(binding_span), num_pixels * 4U,
                                 shader_binding_name<kShader, 5>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 6>(binding_span), num_pixels,
                                        shader_binding_name<kShader, 6>());
  // clang-format on
}

void validate_forward_bindings(const ForwardBindings &bindings, const RendererUniforms &uniforms) {
  require_binding(bindings.index_buffer_offset, "index_buffer_offset");
  require_binding(bindings.sorting_keys_1, "sorting_keys_1");
  require_binding(bindings.sorting_gauss_idx_1, "sorting_gauss_idx_1");
  require_binding(bindings.sorting_keys_2, "sorting_keys_2");
  require_binding(bindings.sorting_gauss_idx_2, "sorting_gauss_idx_2");
  require_binding(bindings.tile_ranges, "tile_ranges");
  require_binding(bindings.pixel_state, "pixel_state");
  require_binding(bindings.n_contributors, "n_contributors");

  const auto num_splats = static_cast<std::size_t>(require_uint32_count(uniforms.num_splats, "num_splats"));
  const auto num_tiles = require_tile_count(uniforms);
  const auto num_pixels = require_pixel_count(uniforms);
  if (num_splats > std::numeric_limits<std::size_t>::max() / num_tiles) {
    throw std::invalid_argument("forward scratch capacity overflows size_t");
  }
  const auto max_indices = num_splats * num_tiles;

  // clang-format off
  require_buffer_elements<std::int32_t>(*bindings.index_buffer_offset, num_splats, "index_buffer_offset");
  require_buffer_elements<SortingKey>(*bindings.sorting_keys_1, max_indices, "sorting_keys_1");
  require_buffer_elements<std::int32_t>(*bindings.sorting_gauss_idx_1, max_indices, "sorting_gauss_idx_1");
  require_buffer_elements<SortingKey>(*bindings.sorting_keys_2, max_indices, "sorting_keys_2");
  require_buffer_elements<std::int32_t>(*bindings.sorting_gauss_idx_2, max_indices, "sorting_gauss_idx_2");
  require_buffer_elements<std::int32_t>(*bindings.tile_ranges, num_tiles + 1U, "tile_ranges");
  require_buffer_elements<float>(*bindings.pixel_state, num_pixels * 4U, "pixel_state");
  require_buffer_elements<std::int32_t>(*bindings.n_contributors, num_pixels, "n_contributors");
  // clang-format on
}

void validate_radix_sort_bindings(const RadixSortBindings &bindings, std::size_t element_count) {
  require_binding(bindings.keys_1, "keys_1");
  require_binding(bindings.indices_1, "indices_1");
  require_binding(bindings.keys_2, "keys_2");
  require_binding(bindings.indices_2, "indices_2");

  // clang-format off
  require_buffer_elements<SortingKey>(*bindings.keys_1, element_count, "keys_1");
  require_buffer_elements<std::uint32_t>(*bindings.indices_1, element_count, "indices_1");
  require_buffer_elements<SortingKey>(*bindings.keys_2, element_count, "keys_2");
  require_buffer_elements<std::uint32_t>(*bindings.indices_2, element_count, "indices_2");
  // clang-format on
}

} // namespace

std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  if (divisor == 0) {
    throw std::invalid_argument("ceil_div divisor must be > 0");
  }
  return (value + divisor - 1U) / divisor;
}

DispatchShape dispatch_groups_for(std::size_t element_count, std::uint32_t block_size) {
  if (block_size == 0) {
    throw std::invalid_argument("dispatch block_size must be > 0");
  }
  const auto group_count = ceil_div(element_count, block_size);
  return {require_uint32_count(group_count, "dispatch group count"), 1U, 1U};
}

void execute_cumsum(const HeadlessContext &context,
                    const StorageBuffer &input,
                    StorageBuffer &output,
                    std::size_t element_count) {
  constexpr auto kShader = ShaderId::CumsumSinglePass;
  require_uint32_count(element_count, "element_count");
  // clang-format off
  require_buffer_elements<std::int32_t>(input, element_count, shader_binding_name<kShader, 0>());
  require_buffer_elements<std::int32_t>(output, element_count, shader_binding_name<kShader, 1>());
  // clang-format on

  const auto num_blocks = ceil_div(element_count, static_cast<std::size_t>(VKSPLAT_CUMSUM_BLOCK_SIZE));
  auto block_sums = make_zero_buffer<std::int32_t>(context, num_blocks);

  if (element_count <= VKSPLAT_CUMSUM_BLOCK_SIZE) {
    const auto &shader = shader_interface("cumsum_single_pass");
    auto spirv = make_span(shaders::kCumsumSinglePassSpirv);
    ComputePipeline pipeline(context, spirv, shader.binding_count, shader.push_constant_size);

    const auto bindings = cumsum_storage_bindings(input, output, block_sums);
    dispatch_with_element_count(pipeline, shader, make_span(bindings), element_count, VKSPLAT_CUMSUM_BLOCK_SIZE);
    return;
  }

  const auto &block_scan_shader = shader_interface("cumsum_block_scan");
  const auto &scan_block_sums_shader = shader_interface("cumsum_scan_block_sums");
  const auto &add_offsets_shader = shader_interface("cumsum_add_block_offsets");

  auto block_scan_spirv = make_span(shaders::kCumsumBlockScanSpirv);
  auto scan_block_sums_spirv = make_span(shaders::kCumsumScanBlockSumsSpirv);
  auto add_offsets_spirv = make_span(shaders::kCumsumAddBlockOffsetsSpirv);

  ComputePipeline block_scan_pipeline(context, block_scan_spirv, block_scan_shader.binding_count,
                                      block_scan_shader.push_constant_size);
  ComputePipeline scan_block_sums_pipeline(context, scan_block_sums_spirv, scan_block_sums_shader.binding_count,
                                           scan_block_sums_shader.push_constant_size);
  ComputePipeline add_offsets_pipeline(context, add_offsets_spirv, add_offsets_shader.binding_count,
                                       add_offsets_shader.push_constant_size);

  const auto primary_bindings = cumsum_storage_bindings(input, output, block_sums);

  // clang-format off
  dispatch_with_element_count(block_scan_pipeline, block_scan_shader, make_span(primary_bindings), element_count,
                              VKSPLAT_CUMSUM_BLOCK_SIZE);

  if (num_blocks > VKSPLAT_CUMSUM_BLOCK_SIZE) {
    const auto num_blocks2 = ceil_div(num_blocks, static_cast<std::size_t>(VKSPLAT_CUMSUM_BLOCK_SIZE));
    auto block_sums2 = make_zero_buffer<std::int32_t>(context, num_blocks2);

    // NOLINTNEXTLINE(readability-suspicious-call-argument)
    const auto block_sums_bindings = cumsum_storage_bindings(block_sums, block_sums, block_sums2);

    dispatch_with_element_count(block_scan_pipeline, block_scan_shader, make_span(block_sums_bindings), num_blocks,
                                VKSPLAT_CUMSUM_BLOCK_SIZE);
    dispatch_with_element_count(scan_block_sums_pipeline, scan_block_sums_shader, make_span(block_sums_bindings),
                                num_blocks2, VKSPLAT_CUMSUM_BLOCK_SIZE);
    dispatch_with_element_count(add_offsets_pipeline, add_offsets_shader, make_span(block_sums_bindings), num_blocks,
                                VKSPLAT_CUMSUM_BLOCK_SIZE);
  } else {
    dispatch_with_element_count(scan_block_sums_pipeline, scan_block_sums_shader, make_span(primary_bindings),
                                num_blocks, VKSPLAT_CUMSUM_BLOCK_SIZE);
  }

  dispatch_with_element_count(add_offsets_pipeline, add_offsets_shader, make_span(primary_bindings), element_count,
                              VKSPLAT_CUMSUM_BLOCK_SIZE);
  // clang-format on
}

void execute_sum(const HeadlessContext &context,
                 const StorageBuffer &input,
                 StorageBuffer &output,
                 std::size_t element_count) {
  constexpr auto kShader = ShaderId::Sum;
  const auto &shader = shader_interface(kShader);
  const auto bindings = sum_storage_bindings(input, output);
  const auto binding_span = make_span(bindings);

  require_uint32_count(element_count, "element_count");
  require_storage_bindings(shader, binding_span);
  // clang-format off
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 0>(binding_span), element_count,
                                        shader_binding_name<kShader, 0>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 1>(binding_span), 1U,
                                        shader_binding_name<kShader, 1>());
  // clang-format on

  const std::int32_t zero = 0;
  output.upload(ByteView::from_object(zero));

  auto spirv = make_span(shaders::kSumSpirv);
  ComputePipeline pipeline(context, spirv, shader.binding_count, shader.push_constant_size);

  dispatch_with_element_count(pipeline, shader, binding_span, element_count, VKSPLAT_SUM_BLOCK_SIZE);
}

void execute_where(const HeadlessContext &context,
                   const StorageBuffer &mask,
                   const StorageBuffer &mask_cumsum,
                   StorageBuffer &out_indices,
                   std::size_t element_count) {
  constexpr auto kShader = ShaderId::Where;
  const auto &shader = shader_interface(kShader);
  const auto bindings = where_storage_bindings(mask, mask_cumsum, out_indices);
  const auto binding_span = make_span(bindings);

  require_uint32_count(element_count, "element_count");
  require_storage_bindings(shader, binding_span);
  // clang-format off
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 0>(binding_span), element_count,
                                        shader_binding_name<kShader, 0>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 1>(binding_span), element_count,
                                        shader_binding_name<kShader, 1>());
  require_buffer_elements<std::int32_t>(storage_binding_at<kShader, 2>(binding_span), 1U,
                                        shader_binding_name<kShader, 2>());
  // clang-format on

  auto spirv = make_span(shaders::kWhereSpirv);
  ComputePipeline pipeline(context, spirv, shader.binding_count, shader.push_constant_size);

  dispatch_with_element_count(pipeline, shader, binding_span, element_count, VKSPLAT_WHERE_BLOCK_SIZE);
}

void execute_projection_forward(const HeadlessContext &context,
                                const ProjectionForwardBindings &bindings,
                                const RendererUniforms &uniforms) {
  const auto num_splats = static_cast<std::size_t>(require_uint32_count(uniforms.num_splats, "num_splats"));
  validate_projection_forward_bindings(bindings, num_splats);

  const auto &shader = shader_interface("projection_forward");
  auto spirv = make_span(shaders::kProjectionForwardSpirv);
  ComputePipeline pipeline(context, spirv, shader.binding_count, shader.push_constant_size);

  const auto named_bindings = projection_forward_storage_bindings(bindings);
  pipeline.bind_storage_buffers(resolve_storage_bindings(shader, make_span(named_bindings)));

  const auto push_constants = ByteView::from_object(uniforms);
  pipeline.dispatch(dispatch_groups_for(num_splats, VKSPLAT_SUBGROUP_SIZE), push_constants);
}

void execute_generate_keys(const HeadlessContext &context,
                           const GenerateKeysBindings &bindings,
                           const RendererUniforms &uniforms,
                           std::size_t output_count) {
  const auto num_splats = static_cast<std::size_t>(require_uint32_count(uniforms.num_splats, "num_splats"));
  require_uint32_count(output_count, "output_count");
  validate_generate_keys_bindings(bindings, num_splats, output_count);

  const auto &shader = shader_interface("generate_keys");
  auto spirv = make_span(shaders::kGenerateKeysSpirv);
  ComputePipeline pipeline(context, spirv, shader.binding_count, shader.push_constant_size);

  const auto named_bindings = generate_keys_storage_bindings(bindings);
  pipeline.bind_storage_buffers(resolve_storage_bindings(shader, make_span(named_bindings)));

  const auto push_constants = ByteView::from_object(uniforms);
  pipeline.dispatch(dispatch_groups_for(num_splats, VKSPLAT_TILE_SHADER_GENERATE_KEYS_BLOCK_SIZE), push_constants);
}

void execute_compute_tile_ranges(const HeadlessContext &context,
                                 const ComputeTileRangesBindings &bindings,
                                 const RendererUniforms &uniforms,
                                 std::size_t num_indices) {
  const auto num_indices_u32 = require_uint32_count(num_indices, "num_indices");
  const auto num_tiles = require_tile_count(uniforms);
  validate_compute_tile_ranges_bindings(bindings, num_indices, num_tiles);

  auto tile_range_uniforms = uniforms;
  tile_range_uniforms.active_sh = num_indices_u32;

  const auto &shader = shader_interface("compute_tile_ranges");
  auto spirv = make_span(shaders::kComputeTileRangesSpirv);
  ComputePipeline pipeline(context, spirv, shader.binding_count, shader.push_constant_size);

  const auto named_bindings = compute_tile_ranges_storage_bindings(bindings);
  pipeline.bind_storage_buffers(resolve_storage_bindings(shader, make_span(named_bindings)));

  const auto push_constants = ByteView::from_object(tile_range_uniforms);
  pipeline.dispatch(dispatch_groups_for(num_indices + 1U, VKSPLAT_TILE_SHADER_TILE_RANGES_THREADS), push_constants);
}

void execute_rasterize_forward(const HeadlessContext &context,
                               const RasterizeForwardBindings &bindings,
                               const RendererUniforms &uniforms,
                               std::size_t num_indices) {
  require_uint32_count(num_indices, "num_indices");
  const auto num_splats = static_cast<std::size_t>(require_uint32_count(uniforms.num_splats, "num_splats"));
  const auto num_tiles = require_tile_count(uniforms);
  const auto num_pixels = require_pixel_count(uniforms);
  validate_rasterize_forward_bindings(bindings, num_splats, num_indices, num_tiles, num_pixels);

  const auto &shader = shader_interface("rasterize_forward");
  auto spirv = make_span(shaders::kRasterizeForwardSpirv);
  ComputePipeline pipeline(context, spirv, shader.binding_count, shader.push_constant_size);

  const auto named_bindings = rasterize_forward_storage_bindings(bindings);
  pipeline.bind_storage_buffers(resolve_storage_bindings(shader, make_span(named_bindings)));

  const auto groups_x =
      require_uint32_count(ceil_div(uniforms.image_width, VKSPLAT_TILE_WIDTH), "rasterize_forward groups_x");
  const auto groups_y =
      require_uint32_count(ceil_div(uniforms.image_height, VKSPLAT_TILE_HEIGHT), "rasterize_forward groups_y");
  const DispatchShape dispatch_shape{groups_x, groups_y, 1U};
  const auto push_constants = ByteView::from_object(uniforms);
  pipeline.dispatch(dispatch_shape, push_constants);
}

auto execute_sort(const HeadlessContext &context, const RadixSortBindings &bindings, std::size_t element_count)
    -> RadixSortResult {
  const auto element_count_u32 = require_uint32_count(element_count, "element_count");
  validate_radix_sort_bindings(bindings, element_count);

  const auto num_parts = ceil_div(element_count, static_cast<std::size_t>(VKSPLAT_RADIX_PARTITION_SIZE));
  const auto num_parts_u32 = require_uint32_count(num_parts, "radix sort partition count");

  auto global_histogram =
      make_zero_buffer<std::uint32_t>(context, static_cast<std::size_t>(kRadixSortPasses) * VKSPLAT_RADIX_SORT_RADIX);

  auto partition_histogram = make_zero_buffer<std::uint32_t>(context, num_parts * VKSPLAT_RADIX_SORT_RADIX);

  auto upsweep_spirv = make_span(shaders::kRadixSortUpsweepSpirv);
  auto spine_spirv = make_span(shaders::kRadixSortSpineSpirv);
  auto downsweep_spirv = make_span(shaders::kRadixSortDownsweepSpirv);

  const auto &upsweep_shader = shader_interface("radix_sort_upsweep");
  const auto &spine_shader = shader_interface("radix_sort_spine");
  const auto &downsweep_shader = shader_interface("radix_sort_downsweep");

  ComputePipeline upsweep_pipeline(context, upsweep_spirv, upsweep_shader.binding_count,
                                   upsweep_shader.push_constant_size);
  ComputePipeline spine_pipeline(context, spine_spirv, spine_shader.binding_count, spine_shader.push_constant_size);
  ComputePipeline downsweep_pipeline(context, downsweep_spirv, downsweep_shader.binding_count,
                                     downsweep_shader.push_constant_size);

  const DispatchShape partition_dispatch_shape{num_parts_u32, 1U, 1U};
  const DispatchShape spine_dispatch_shape{VKSPLAT_RADIX_SORT_RADIX, 1U, 1U};

  const StorageBuffer *keys_in = bindings.keys_1;
  const StorageBuffer *indices_in = bindings.indices_1;
  const StorageBuffer *keys_out = bindings.keys_2;
  const StorageBuffer *indices_out = bindings.indices_2;

  for (std::uint32_t pass = 0; pass < kRadixSortPasses; ++pass) {
    const RadixSortPushConstants push_constants{pass, element_count_u32};

    const auto upsweep_bindings = radix_sort_upsweep_storage_bindings(*keys_in, global_histogram, partition_histogram);
    dispatch_radix_sort_pass(upsweep_pipeline, upsweep_shader, make_span(upsweep_bindings), partition_dispatch_shape,
                             push_constants);

    const auto spine_bindings = radix_sort_spine_storage_bindings(global_histogram, partition_histogram);
    dispatch_radix_sort_pass(spine_pipeline, spine_shader, make_span(spine_bindings), spine_dispatch_shape,
                             push_constants);

    // clang-format off
    const auto downsweep_bindings = radix_sort_downsweep_storage_bindings(
        global_histogram, partition_histogram, *keys_in, *indices_in, *keys_out, *indices_out);
    dispatch_radix_sort_pass(downsweep_pipeline, downsweep_shader, make_span(downsweep_bindings),
                             partition_dispatch_shape, push_constants);
    // clang-format on

    std::swap(keys_in, keys_out);
    std::swap(indices_in, indices_out);
  }

  return {keys_in, indices_in};
}

auto execute_forward(const HeadlessContext &context, const ForwardBindings &bindings, const RendererUniforms &uniforms)
    -> ForwardResult {
  validate_forward_bindings(bindings, uniforms);

  execute_projection_forward(context, bindings.projection, uniforms);
  execute_cumsum(context, *bindings.projection.tiles_touched, *bindings.index_buffer_offset, uniforms.num_splats);

  const auto index_buffer_offset = bindings.index_buffer_offset->read_back<std::int32_t>(uniforms.num_splats);
  const auto num_indices_i32 = index_buffer_offset.back();
  if (num_indices_i32 < 0) {
    throw std::runtime_error("forward num_indices is negative");
  }

  const auto num_indices = static_cast<std::size_t>(num_indices_i32);
  if (num_indices == 0U) {
    return {0U, false, nullptr, nullptr};
  }

  GenerateKeysBindings generate_keys_bindings{};
  generate_keys_bindings.xy_vs = bindings.projection.xy_vs;
  generate_keys_bindings.inv_cov_vs_opacity = bindings.projection.inv_cov_vs_opacity;
  generate_keys_bindings.depths = bindings.projection.depths;
  generate_keys_bindings.rect_tile_space = bindings.projection.rect_tile_space;
  generate_keys_bindings.index_buffer_offset = bindings.index_buffer_offset;
  generate_keys_bindings.unsorted_keys = bindings.sorting_keys_1;
  generate_keys_bindings.unsorted_gauss_idx = bindings.sorting_gauss_idx_1;
  execute_generate_keys(context, generate_keys_bindings, uniforms, num_indices);

  RadixSortBindings sort_bindings{};
  sort_bindings.keys_1 = bindings.sorting_keys_1;
  sort_bindings.indices_1 = bindings.sorting_gauss_idx_1;
  sort_bindings.keys_2 = bindings.sorting_keys_2;
  sort_bindings.indices_2 = bindings.sorting_gauss_idx_2;
  const auto sort_result = execute_sort(context, sort_bindings, num_indices);

  ComputeTileRangesBindings tile_range_bindings{};
  tile_range_bindings.sorted_keys = sort_result.keys;
  tile_range_bindings.tile_ranges = bindings.tile_ranges;
  execute_compute_tile_ranges(context, tile_range_bindings, uniforms, num_indices);

  RasterizeForwardBindings raster_bindings{};
  raster_bindings.sorted_gauss_idx = sort_result.indices;
  raster_bindings.tile_ranges = bindings.tile_ranges;
  raster_bindings.xy_vs = bindings.projection.xy_vs;
  raster_bindings.inv_cov_vs_opacity = bindings.projection.inv_cov_vs_opacity;
  raster_bindings.rgb = bindings.projection.rgb;
  raster_bindings.pixel_state = bindings.pixel_state;
  raster_bindings.n_contributors = bindings.n_contributors;
  execute_rasterize_forward(context, raster_bindings, uniforms, num_indices);

  return {num_indices, true, sort_result.keys, sort_result.indices};
}

} // namespace nlrc::vksplat::gpu
