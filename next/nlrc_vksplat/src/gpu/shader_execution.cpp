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
                                 const std::vector<const StorageBuffer *> &bindings,
                                 std::size_t element_count,
                                 std::uint32_t block_size) {
  pipeline.bind_storage_buffers(bindings);

  const ElementCountPushConstants push_constants{require_uint32_count(element_count, "element_count")};
  const auto push_constants_view = ByteView::from_object(push_constants);
  pipeline.dispatch(dispatch_groups_for(element_count, block_size), push_constants_view);
}

void dispatch_radix_sort_pass(ComputePipeline &pipeline,
                              const std::vector<const StorageBuffer *> &bindings,
                              DispatchShape dispatch_shape,
                              RadixSortPushConstants push_constants) {
  pipeline.bind_storage_buffers(bindings);

  const auto push_constants_view = ByteView::from_object(push_constants);
  pipeline.dispatch(dispatch_shape, push_constants_view);
}

[[nodiscard]] std::size_t projection_sh_coeff_float_count(std::size_t num_splats) {
  const auto reorder_groups = ceil_div(num_splats, static_cast<std::size_t>(VKSPLAT_SH_REORDER_SIZE));
  return reorder_groups * 12U * VKSPLAT_SH_REORDER_SIZE * 4U;
}

void validate_projection_forward_bindings(const ProjectionForwardBindings &bindings, std::size_t num_splats) {
  require_binding(bindings.xyz_ws, "xyz_ws");
  require_binding(bindings.sh_coeffs, "sh_coeffs");
  require_binding(bindings.rotations, "rotations");
  require_binding(bindings.scales_opacs, "scales_opacs");
  require_binding(bindings.tiles_touched, "tiles_touched");
  require_binding(bindings.rect_tile_space, "rect_tile_space");
  require_binding(bindings.radii, "radii");
  require_binding(bindings.xy_vs, "xy_vs");
  require_binding(bindings.depths, "depths");
  require_binding(bindings.inv_cov_vs_opacity, "inv_cov_vs_opacity");
  require_binding(bindings.rgb, "rgb");

  require_buffer_elements<float>(*bindings.xyz_ws, num_splats * 3U, "xyz_ws");
  require_buffer_elements<float>(*bindings.sh_coeffs, projection_sh_coeff_float_count(num_splats), "sh_coeffs");
  require_buffer_elements<float>(*bindings.rotations, num_splats * 4U, "rotations");
  require_buffer_elements<float>(*bindings.scales_opacs, num_splats * 4U, "scales_opacs");
  require_buffer_elements<std::int32_t>(*bindings.tiles_touched, num_splats, "tiles_touched");
  require_buffer_elements<RectTileSpace>(*bindings.rect_tile_space, num_splats * kRectTileSpaceWords,
                                         "rect_tile_space");
  require_buffer_elements<std::int32_t>(*bindings.radii, num_splats, "radii");
  require_buffer_elements<float>(*bindings.xy_vs, num_splats * 2U, "xy_vs");
  require_buffer_elements<float>(*bindings.depths, num_splats, "depths");
  require_buffer_elements<float>(*bindings.inv_cov_vs_opacity, num_splats * 4U, "inv_cov_vs_opacity");
  require_buffer_elements<float>(*bindings.rgb, num_splats * 3U, "rgb");
}

void validate_generate_keys_bindings(const GenerateKeysBindings &bindings,
                                     std::size_t num_splats,
                                     std::size_t output_count) {
  require_binding(bindings.xy_vs, "xy_vs");
  require_binding(bindings.inv_cov_vs_opacity, "inv_cov_vs_opacity");
  require_binding(bindings.depths, "depths");
  require_binding(bindings.rect_tile_space, "rect_tile_space");
  require_binding(bindings.index_buffer_offset, "index_buffer_offset");
  require_binding(bindings.unsorted_keys, "unsorted_keys");
  require_binding(bindings.unsorted_gauss_idx, "unsorted_gauss_idx");

  require_buffer_elements<float>(*bindings.xy_vs, num_splats * 2U, "xy_vs");
  require_buffer_elements<float>(*bindings.inv_cov_vs_opacity, num_splats * 4U, "inv_cov_vs_opacity");
  require_buffer_elements<float>(*bindings.depths, num_splats, "depths");
  require_buffer_elements<RectTileSpace>(*bindings.rect_tile_space, num_splats * kRectTileSpaceWords,
                                         "rect_tile_space");
  require_buffer_elements<std::int32_t>(*bindings.index_buffer_offset, num_splats, "index_buffer_offset");
  require_buffer_elements<SortingKey>(*bindings.unsorted_keys, output_count, "unsorted_keys");
  require_buffer_elements<std::int32_t>(*bindings.unsorted_gauss_idx, output_count, "unsorted_gauss_idx");
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
  require_binding(bindings.sorted_keys, "sorted_keys");
  require_binding(bindings.tile_ranges, "tile_ranges");

  require_buffer_elements<SortingKey>(*bindings.sorted_keys, num_indices, "sorted_keys");
  require_buffer_elements<std::int32_t>(*bindings.tile_ranges, num_tiles + 1U, "tile_ranges");
}

void validate_rasterize_forward_bindings(const RasterizeForwardBindings &bindings,
                                         std::size_t num_splats,
                                         std::size_t num_indices,
                                         std::size_t num_tiles,
                                         std::size_t num_pixels) {
  require_binding(bindings.sorted_gauss_idx, "sorted_gauss_idx");
  require_binding(bindings.tile_ranges, "tile_ranges");
  require_binding(bindings.xy_vs, "xy_vs");
  require_binding(bindings.inv_cov_vs_opacity, "inv_cov_vs_opacity");
  require_binding(bindings.rgb, "rgb");
  require_binding(bindings.pixel_state, "pixel_state");
  require_binding(bindings.n_contributors, "n_contributors");

  require_buffer_elements<std::int32_t>(*bindings.sorted_gauss_idx, num_indices, "sorted_gauss_idx");
  require_buffer_elements<std::int32_t>(*bindings.tile_ranges, num_tiles + 1U, "tile_ranges");
  require_buffer_elements<float>(*bindings.xy_vs, num_splats * 2U, "xy_vs");
  require_buffer_elements<float>(*bindings.inv_cov_vs_opacity, num_splats * 4U, "inv_cov_vs_opacity");
  require_buffer_elements<float>(*bindings.rgb, num_splats * 3U, "rgb");
  require_buffer_elements<float>(*bindings.pixel_state, num_pixels * 4U, "pixel_state");
  require_buffer_elements<std::int32_t>(*bindings.n_contributors, num_pixels, "n_contributors");
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

  require_buffer_elements<std::int32_t>(*bindings.index_buffer_offset, num_splats, "index_buffer_offset");
  require_buffer_elements<SortingKey>(*bindings.sorting_keys_1, max_indices, "sorting_keys_1");
  require_buffer_elements<std::int32_t>(*bindings.sorting_gauss_idx_1, max_indices, "sorting_gauss_idx_1");
  require_buffer_elements<SortingKey>(*bindings.sorting_keys_2, max_indices, "sorting_keys_2");
  require_buffer_elements<std::int32_t>(*bindings.sorting_gauss_idx_2, max_indices, "sorting_gauss_idx_2");
  require_buffer_elements<std::int32_t>(*bindings.tile_ranges, num_tiles + 1U, "tile_ranges");
  require_buffer_elements<float>(*bindings.pixel_state, num_pixels * 4U, "pixel_state");
  require_buffer_elements<std::int32_t>(*bindings.n_contributors, num_pixels, "n_contributors");
}

void validate_radix_sort_bindings(const RadixSortBindings &bindings, std::size_t element_count) {
  require_binding(bindings.keys_1, "keys_1");
  require_binding(bindings.indices_1, "indices_1");
  require_binding(bindings.keys_2, "keys_2");
  require_binding(bindings.indices_2, "indices_2");

  require_buffer_elements<SortingKey>(*bindings.keys_1, element_count, "keys_1");
  require_buffer_elements<std::uint32_t>(*bindings.indices_1, element_count, "indices_1");
  require_buffer_elements<SortingKey>(*bindings.keys_2, element_count, "keys_2");
  require_buffer_elements<std::uint32_t>(*bindings.indices_2, element_count, "indices_2");
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
  require_uint32_count(element_count, "element_count");
  require_buffer_elements<std::int32_t>(input, element_count, "input");
  require_buffer_elements<std::int32_t>(output, element_count, "output");

  const auto num_blocks = ceil_div(element_count, static_cast<std::size_t>(VKSPLAT_CUMSUM_BLOCK_SIZE));
  auto block_sums = make_zero_buffer<std::int32_t>(context, num_blocks);

  if (element_count <= VKSPLAT_CUMSUM_BLOCK_SIZE) {
    auto spirv = make_span(shaders::kCumsumSinglePassSpirv);
    ComputePipeline pipeline(context, spirv, kCumsumBindingCount, sizeof(ElementCountPushConstants));
    dispatch_with_element_count(pipeline, {&input, &output, &block_sums}, element_count, VKSPLAT_CUMSUM_BLOCK_SIZE);
    return;
  }

  auto block_scan_spirv = make_span(shaders::kCumsumBlockScanSpirv);
  auto scan_block_sums_spirv = make_span(shaders::kCumsumScanBlockSumsSpirv);
  auto add_offsets_spirv = make_span(shaders::kCumsumAddBlockOffsetsSpirv);

  ComputePipeline block_scan_pipeline(context, block_scan_spirv, kCumsumBindingCount,
                                      sizeof(ElementCountPushConstants));
  ComputePipeline scan_block_sums_pipeline(context, scan_block_sums_spirv, kCumsumBindingCount,
                                           sizeof(ElementCountPushConstants));
  ComputePipeline add_offsets_pipeline(context, add_offsets_spirv, kCumsumBindingCount,
                                       sizeof(ElementCountPushConstants));

  const std::vector<const StorageBuffer *> primary_bindings = {
      &input,
      &output,
      &block_sums,
  };

  dispatch_with_element_count(block_scan_pipeline, primary_bindings, element_count, VKSPLAT_CUMSUM_BLOCK_SIZE);

  if (num_blocks > VKSPLAT_CUMSUM_BLOCK_SIZE) {
    const auto num_blocks2 = ceil_div(num_blocks, static_cast<std::size_t>(VKSPLAT_CUMSUM_BLOCK_SIZE));
    auto block_sums2 = make_zero_buffer<std::int32_t>(context, num_blocks2);

    const std::vector<const StorageBuffer *> block_sums_bindings = {
        &block_sums,
        &block_sums,
        &block_sums2,
    };

    dispatch_with_element_count(block_scan_pipeline, block_sums_bindings, num_blocks, VKSPLAT_CUMSUM_BLOCK_SIZE);
    dispatch_with_element_count(scan_block_sums_pipeline, block_sums_bindings, num_blocks2, VKSPLAT_CUMSUM_BLOCK_SIZE);
    dispatch_with_element_count(add_offsets_pipeline, block_sums_bindings, num_blocks, VKSPLAT_CUMSUM_BLOCK_SIZE);
  } else {
    dispatch_with_element_count(scan_block_sums_pipeline, primary_bindings, num_blocks, VKSPLAT_CUMSUM_BLOCK_SIZE);
  }

  dispatch_with_element_count(add_offsets_pipeline, primary_bindings, element_count, VKSPLAT_CUMSUM_BLOCK_SIZE);
}

void execute_sum(const HeadlessContext &context,
                 const StorageBuffer &input,
                 StorageBuffer &output,
                 std::size_t element_count) {
  require_uint32_count(element_count, "element_count");
  require_buffer_elements<std::int32_t>(input, element_count, "input");
  require_buffer_elements<std::int32_t>(output, 1U, "output");

  const std::int32_t zero = 0;
  output.upload(ByteView::from_object(zero));

  auto spirv = make_span(shaders::kSumSpirv);
  ComputePipeline pipeline(context, spirv, kSumBindingCount, sizeof(ElementCountPushConstants));
  dispatch_with_element_count(pipeline, {&input, &output}, element_count, VKSPLAT_SUM_BLOCK_SIZE);
}

void execute_where(const HeadlessContext &context,
                   const StorageBuffer &mask,
                   const StorageBuffer &mask_cumsum,
                   StorageBuffer &out_indices,
                   std::size_t element_count) {
  require_uint32_count(element_count, "element_count");
  require_buffer_elements<std::int32_t>(mask, element_count, "mask");
  require_buffer_elements<std::int32_t>(mask_cumsum, element_count, "mask_cumsum");
  require_buffer_elements<std::int32_t>(out_indices, 1U, "out_indices");

  auto spirv = make_span(shaders::kWhereSpirv);
  ComputePipeline pipeline(context, spirv, kWhereBindingCount, sizeof(ElementCountPushConstants));
  dispatch_with_element_count(pipeline, {&mask, &mask_cumsum, &out_indices}, element_count, VKSPLAT_WHERE_BLOCK_SIZE);
}

void execute_projection_forward(const HeadlessContext &context,
                                const ProjectionForwardBindings &bindings,
                                const RendererUniforms &uniforms) {
  const auto num_splats = static_cast<std::size_t>(require_uint32_count(uniforms.num_splats, "num_splats"));
  validate_projection_forward_bindings(bindings, num_splats);

  auto spirv = make_span(shaders::kProjectionForwardSpirv);
  ComputePipeline pipeline(context, spirv, kProjectionForwardBindingCount, sizeof(RendererUniforms));

  pipeline.bind_storage_buffers({
      bindings.xyz_ws,
      bindings.sh_coeffs,
      bindings.rotations,
      bindings.scales_opacs,
      bindings.tiles_touched,
      bindings.rect_tile_space,
      bindings.radii,
      bindings.xy_vs,
      bindings.depths,
      bindings.inv_cov_vs_opacity,
      bindings.rgb,
  });

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

  auto spirv = make_span(shaders::kGenerateKeysSpirv);
  ComputePipeline pipeline(context, spirv, kGenerateKeysBindingCount, sizeof(RendererUniforms));

  pipeline.bind_storage_buffers({
      bindings.xy_vs,
      bindings.inv_cov_vs_opacity,
      bindings.depths,
      bindings.rect_tile_space,
      bindings.index_buffer_offset,
      bindings.unsorted_keys,
      bindings.unsorted_gauss_idx,
  });

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

  auto spirv = make_span(shaders::kComputeTileRangesSpirv);
  ComputePipeline pipeline(context, spirv, kComputeTileRangesBindingCount, sizeof(RendererUniforms));

  pipeline.bind_storage_buffers({
      bindings.sorted_keys,
      bindings.tile_ranges,
  });

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

  auto spirv = make_span(shaders::kRasterizeForwardSpirv);
  ComputePipeline pipeline(context, spirv, kRasterizeForwardBindingCount, sizeof(RendererUniforms));

  pipeline.bind_storage_buffers({
      bindings.sorted_gauss_idx,
      bindings.tile_ranges,
      bindings.xy_vs,
      bindings.inv_cov_vs_opacity,
      bindings.rgb,
      bindings.pixel_state,
      bindings.n_contributors,
  });

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

  ComputePipeline upsweep_pipeline(context, upsweep_spirv, kRadixSortUpsweepBindingCount,
                                   sizeof(RadixSortPushConstants));
  ComputePipeline spine_pipeline(context, spine_spirv, kRadixSortSpineBindingCount, sizeof(RadixSortPushConstants));
  ComputePipeline downsweep_pipeline(context, downsweep_spirv, kRadixSortDownsweepBindingCount,
                                     sizeof(RadixSortPushConstants));

  const DispatchShape partition_dispatch_shape{num_parts_u32, 1U, 1U};
  const DispatchShape spine_dispatch_shape{VKSPLAT_RADIX_SORT_RADIX, 1U, 1U};

  const StorageBuffer *keys_in = bindings.keys_1;
  const StorageBuffer *indices_in = bindings.indices_1;
  const StorageBuffer *keys_out = bindings.keys_2;
  const StorageBuffer *indices_out = bindings.indices_2;

  for (std::uint32_t pass = 0; pass < kRadixSortPasses; ++pass) {
    const RadixSortPushConstants push_constants{pass, element_count_u32};

    dispatch_radix_sort_pass(upsweep_pipeline, {keys_in, &global_histogram, &partition_histogram},
                             partition_dispatch_shape, push_constants);

    dispatch_radix_sort_pass(spine_pipeline, {&global_histogram, &partition_histogram}, spine_dispatch_shape,
                             push_constants);

    dispatch_radix_sort_pass(downsweep_pipeline,
                             {&global_histogram, &partition_histogram, keys_in, indices_in, keys_out, indices_out},
                             partition_dispatch_shape, push_constants);

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
