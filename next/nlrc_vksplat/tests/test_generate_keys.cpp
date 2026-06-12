#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "cumsum_single_pass_spirv.hpp"
#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "generate_keys_spirv.hpp"
#include "gpu/compute_pipeline.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "nlrc_vksplat_config.hpp"
#include "projection_forward_spirv.hpp"
#include "span.hpp"

namespace {

inline constexpr std::uint32_t kGenerateKeysBindingCount = 7;
inline constexpr std::uint32_t kProjectionForwardBindingCount = 11;
inline constexpr std::uint32_t kCumsumStorageBufferCount = 3;
inline constexpr std::uint32_t kImageHeight = 32;
inline constexpr std::uint32_t kImageWidth = 32;
inline constexpr std::uint32_t kGridHeight = 2;
inline constexpr std::uint32_t kGridWidth = 2;
inline constexpr std::uint32_t kDepthBits = 23;

static_assert(VKSPLAT_SORTING_KEY_BITS == 32, "Update generate_keys tests when 64-bit keys are enabled.");

struct VulkanGSRendererUniforms final {
  std::uint32_t image_height;
  std::uint32_t image_width;
  std::uint32_t grid_height;
  std::uint32_t grid_width;
  std::uint32_t num_splats;
  std::uint32_t active_sh;
  std::uint32_t step;
  std::uint32_t camera_model;
  float fx;
  float fy;
  float cx;
  float cy;
  std::array<float, 4> dist_coeffs;
  std::array<float, 16> world_view_transform;
};

static_assert(sizeof(VulkanGSRendererUniforms) == 128);
static_assert(sizeof(VulkanGSRendererUniforms) <= nlrc::vksplat::kMaxPushConstantBytes);
static_assert(offsetof(VulkanGSRendererUniforms, fx) == 32);
static_assert(offsetof(VulkanGSRendererUniforms, dist_coeffs) == 48);
static_assert(offsetof(VulkanGSRendererUniforms, world_view_transform) == 64);

struct ElementCountPushConstants final {
  std::uint32_t num_elements;
};

static_assert(sizeof(ElementCountPushConstants) == 4);

using StorageBufferMap = std::map<std::string, const nlrc::vksplat::gpu::StorageBuffer *>;

template <typename T>
[[nodiscard]] auto make_storage_buffer(const nlrc::vksplat::gpu::HeadlessContext &context, const std::vector<T> &values)
    -> nlrc::vksplat::gpu::StorageBuffer {
  const std::size_t size_bytes = values.size() * sizeof(T);
  nlrc::vksplat::gpu::StorageBuffer buffer(context, size_bytes);

  buffer.upload(values);
  return buffer;
}

[[nodiscard]] std::vector<const nlrc::vksplat::gpu::StorageBuffer *>
bindings_from_manifest(const nlrc::vksplat::tests::FixtureManifest &manifest, const StorageBufferMap &buffers) {
  std::vector<const nlrc::vksplat::gpu::StorageBuffer *> bindings;
  bindings.reserve(manifest.bindings.size());

  for (const auto &name : manifest.bindings) {
    bindings.push_back(buffers.at(name));
  }

  return bindings;
}

[[nodiscard]] constexpr std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  return (value + divisor - 1U) / divisor;
}

[[nodiscard]] VulkanGSRendererUniforms make_uniforms(std::uint32_t num_splats) {
  return {
      kImageHeight,
      kImageWidth,
      kGridHeight,
      kGridWidth,
      num_splats,
      0U,
      0U,
      0U,
      16.0F,
      16.0F,
      16.0F,
      16.0F,
      {0.0F, 0.0F, 0.0F, 0.0F},
      {
          1.0F,
          0.0F,
          0.0F,
          0.0F,
          0.0F,
          1.0F,
          0.0F,
          0.0F,
          0.0F,
          0.0F,
          1.0F,
          0.0F,
          0.0F,
          0.0F,
          0.0F,
          1.0F,
      },
  };
}

void dispatch_cumsum_single_pass(nlrc::vksplat::gpu::ComputePipeline &pipeline,
                                 const std::vector<const nlrc::vksplat::gpu::StorageBuffer *> &storage_buffers,
                                 std::size_t element_count) {
  pipeline.bind_storage_buffers(storage_buffers);

  const ElementCountPushConstants push_constants{static_cast<std::uint32_t>(element_count)};
  const auto push_constants_view = nlrc::vksplat::ByteView::from_object(push_constants);
  pipeline.dispatch({1U, 1U, 1U}, push_constants_view);
}

void dispatch_generate_keys(nlrc::vksplat::gpu::ComputePipeline &pipeline,
                            const std::vector<const nlrc::vksplat::gpu::StorageBuffer *> &storage_buffers,
                            const VulkanGSRendererUniforms &uniforms) {
  pipeline.bind_storage_buffers(storage_buffers);

  const auto push_constants = nlrc::vksplat::ByteView::from_object(uniforms);

  const auto groups = static_cast<std::uint32_t>(
      ceil_div(static_cast<std::size_t>(uniforms.num_splats), VKSPLAT_TILE_SHADER_GENERATE_KEYS_BLOCK_SIZE));

  const nlrc::vksplat::gpu::DispatchShape dispatch_shape = {groups, 1U, 1U};
  pipeline.dispatch(dispatch_shape, push_constants);
}

void require_valid_generate_keys_output(const std::vector<std::uint32_t> &keys,
                                        const std::vector<std::int32_t> &indices,
                                        std::uint32_t num_splats) {
  REQUIRE(keys.size() == indices.size());
  for (std::size_t index = 0; index < keys.size(); ++index) {
    INFO("output index: " << index);
    const auto tile_id = keys[index] >> kDepthBits;
    REQUIRE(tile_id < kGridWidth * kGridHeight);
    REQUIRE(indices[index] >= 0);
    REQUIRE(indices[index] < static_cast<std::int32_t>(num_splats));
  }
}

} // namespace

TEST_CASE("Dispatch generate_keys shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

#if VKSPLAT_USE_EMULATED_INT64
  constexpr const char *kStageName = "generate_keys_emulated_int64";
#else
  constexpr const char *kStageName = "generate_keys_native_int64";
#endif

  const auto fixture_root = nlrc::vksplat::tests::fixture_dir(kStageName);
  const auto golden_root = nlrc::vksplat::tests::golden_dir(kStageName);

  const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(fixture_root / "manifest.json");
  const auto golden_manifest = nlrc::vksplat::tests::load_fixture_manifest(golden_root / "manifest.json");

  const std::vector<std::string> expected_bindings = {
      "xy_vs",         "inv_cov_vs_opacity", "depths", "rect_tile_space", "index_buffer_offset",
      "unsorted_keys", "unsorted_gauss_idx",
  };
  REQUIRE(manifest.bindings == expected_bindings);
  REQUIRE(manifest.bindings.size() == kGenerateKeysBindingCount);
  REQUIRE_FALSE(manifest.profile_agnostic);

  const auto xy_vs = nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "xy_vs");
  const auto inv_cov_vs_opacity =
      nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "inv_cov_vs_opacity");
  const auto depths = nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "depths");
  const auto rect_tile_space = nlrc::vksplat::tests::load_fixture_buffer<nlrc::vksplat::RectTileSpace>(
      fixture_root, manifest, "rect_tile_space");
  const auto tiles_touched =
      nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "tiles_touched");
  const auto index_buffer_offset =
      nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "index_buffer_offset");
  const auto initial_keys =
      nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(fixture_root, manifest, "unsorted_keys");
  const auto initial_indices =
      nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "unsorted_gauss_idx");

  REQUIRE(index_buffer_offset.size() == tiles_touched.size());
  REQUIRE_FALSE(index_buffer_offset.empty());
  REQUIRE(index_buffer_offset.back() == static_cast<std::int32_t>(initial_keys.size()));
  REQUIRE(initial_keys.size() == initial_indices.size());
  REQUIRE(xy_vs.size() == tiles_touched.size() * 2U);
  REQUIRE(inv_cov_vs_opacity.size() == tiles_touched.size() * 4U);
  REQUIRE(depths.size() == tiles_touched.size());
  REQUIRE(rect_tile_space.size() == tiles_touched.size() * nlrc::vksplat::kRectTileSpaceWords);

  std::int32_t running_total = 0;
  for (std::size_t index = 0; index < tiles_touched.size(); ++index) {
    running_total += tiles_touched[index];
    REQUIRE(index_buffer_offset[index] == running_total);
  }

  const nlrc::vksplat::gpu::HeadlessContext context;

  auto xy_vs_buffer = make_storage_buffer(context, xy_vs);
  auto inv_cov_vs_opacity_buffer = make_storage_buffer(context, inv_cov_vs_opacity);
  auto depths_buffer = make_storage_buffer(context, depths);
  auto rect_tile_space_buffer = make_storage_buffer(context, rect_tile_space);
  auto index_buffer_offset_buffer = make_storage_buffer(context, index_buffer_offset);
  auto keys_buffer = make_storage_buffer(context, initial_keys);
  auto indices_buffer = make_storage_buffer(context, initial_indices);

  auto spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kGenerateKeysSpirv);
  nlrc::vksplat::gpu::ComputePipeline pipeline(context, spirv, kGenerateKeysBindingCount,
                                               sizeof(VulkanGSRendererUniforms));

  const StorageBufferMap storage_buffers = {
      {"xy_vs", &xy_vs_buffer},
      {"inv_cov_vs_opacity", &inv_cov_vs_opacity_buffer},
      {"depths", &depths_buffer},
      {"rect_tile_space", &rect_tile_space_buffer},
      {"index_buffer_offset", &index_buffer_offset_buffer},
      {"unsorted_keys", &keys_buffer},
      {"unsorted_gauss_idx", &indices_buffer},
  };

  const auto uniforms = make_uniforms(static_cast<std::uint32_t>(tiles_touched.size()));
  dispatch_generate_keys(pipeline, bindings_from_manifest(manifest, storage_buffers), uniforms);

  const auto expected_keys =
      nlrc::vksplat::tests::load_fixture_buffer<std::uint32_t>(golden_root, golden_manifest, "unsorted_keys");
  const auto expected_indices =
      nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "unsorted_gauss_idx");

  const auto actual_keys = keys_buffer.read_back<std::uint32_t>(expected_keys.size());
  const auto actual_indices = indices_buffer.read_back<std::int32_t>(expected_indices.size());

  REQUIRE(actual_keys.size() == static_cast<std::size_t>(index_buffer_offset.back()));
  require_valid_generate_keys_output(actual_keys, actual_indices, static_cast<std::uint32_t>(tiles_touched.size()));

  for (std::size_t splat_index = 0; splat_index < tiles_touched.size(); ++splat_index) {
    if (tiles_touched[splat_index] == 0) {
      const auto zero_touch_output = static_cast<std::int32_t>(splat_index);
      REQUIRE(std::none_of(actual_indices.begin(), actual_indices.end(), [zero_touch_output](std::int32_t value) {
        return value == zero_touch_output;
      }));
    }
  }

  REQUIRE(actual_keys == expected_keys);
  REQUIRE(actual_indices == expected_indices);
}

TEST_CASE("Chain projection_forward cumsum generate_keys", "[gpu]") {
  NLRC_REQUIRE_GPU();

#if VKSPLAT_USE_EMULATED_INT64
  constexpr const char *kProjectionStageName = "projection_forward_emulated_int64";
#else
  constexpr const char *kProjectionStageName = "projection_forward_native_int64";
#endif

  const auto fixture_root = nlrc::vksplat::tests::fixture_dir(kProjectionStageName);
  const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(fixture_root / "manifest.json");

  const auto xyz_ws = nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "xyz_ws");
  const auto sh_coeffs = nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "sh_coeffs");
  const auto rotations = nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "rotations");
  const auto scales_opacs = nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "scales_opacs");
  const auto tiles_touched =
      nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "tiles_touched");
  const auto rect_tile_space = nlrc::vksplat::tests::load_fixture_buffer<nlrc::vksplat::RectTileSpace>(
      fixture_root, manifest, "rect_tile_space");
  const auto radii = nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "radii");
  const auto xy_vs = nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "xy_vs");
  const auto depths = nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "depths");
  const auto inv_cov_vs_opacity =
      nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "inv_cov_vs_opacity");
  const auto rgb = nlrc::vksplat::tests::load_fixture_buffer<float>(fixture_root, manifest, "rgb");

  REQUIRE(tiles_touched.size() == 1);

  const nlrc::vksplat::gpu::HeadlessContext context;

  auto xyz_ws_buffer = make_storage_buffer(context, xyz_ws);
  auto sh_coeffs_buffer = make_storage_buffer(context, sh_coeffs);
  auto rotations_buffer = make_storage_buffer(context, rotations);
  auto scales_opacs_buffer = make_storage_buffer(context, scales_opacs);
  auto tiles_touched_buffer = make_storage_buffer(context, tiles_touched);
  auto rect_tile_space_buffer = make_storage_buffer(context, rect_tile_space);
  auto radii_buffer = make_storage_buffer(context, radii);
  auto xy_vs_buffer = make_storage_buffer(context, xy_vs);
  auto depths_buffer = make_storage_buffer(context, depths);
  auto inv_cov_vs_opacity_buffer = make_storage_buffer(context, inv_cov_vs_opacity);
  auto rgb_buffer = make_storage_buffer(context, rgb);

  auto projection_spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kProjectionForwardSpirv);
  nlrc::vksplat::gpu::ComputePipeline projection_pipeline(context, projection_spirv, kProjectionForwardBindingCount,
                                                          sizeof(VulkanGSRendererUniforms));

  projection_pipeline.bind_storage_buffers({
      &xyz_ws_buffer,
      &sh_coeffs_buffer,
      &rotations_buffer,
      &scales_opacs_buffer,
      &tiles_touched_buffer,
      &rect_tile_space_buffer,
      &radii_buffer,
      &xy_vs_buffer,
      &depths_buffer,
      &inv_cov_vs_opacity_buffer,
      &rgb_buffer,
  });

  const auto uniforms = make_uniforms(1U);
  projection_pipeline.dispatch({1U, 1U, 1U}, nlrc::vksplat::ByteView::from_object(uniforms));

  std::vector<std::int32_t> index_buffer_offset(tiles_touched.size(), 0);
  std::vector<std::int32_t> block_sums(1, 0);
  auto index_buffer_offset_buffer = make_storage_buffer(context, index_buffer_offset);
  auto block_sums_buffer = make_storage_buffer(context, block_sums);

  auto cumsum_spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kCumsumSinglePassSpirv);
  nlrc::vksplat::gpu::ComputePipeline cumsum_pipeline(context, cumsum_spirv, kCumsumStorageBufferCount,
                                                      sizeof(ElementCountPushConstants));
  dispatch_cumsum_single_pass(cumsum_pipeline, {&tiles_touched_buffer, &index_buffer_offset_buffer, &block_sums_buffer},
                              tiles_touched.size());

  index_buffer_offset = index_buffer_offset_buffer.read_back<std::int32_t>(tiles_touched.size());
  REQUIRE_FALSE(index_buffer_offset.empty());
  const auto num_indices = index_buffer_offset.back();
  REQUIRE(num_indices > 0);

  std::vector<std::uint32_t> initial_keys(static_cast<std::size_t>(num_indices), 0U);
  std::vector<std::int32_t> initial_indices(static_cast<std::size_t>(num_indices), 0);
  auto keys_buffer = make_storage_buffer(context, initial_keys);
  auto indices_buffer = make_storage_buffer(context, initial_indices);

  auto generate_keys_spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kGenerateKeysSpirv);
  nlrc::vksplat::gpu::ComputePipeline generate_keys_pipeline(context, generate_keys_spirv, kGenerateKeysBindingCount,
                                                             sizeof(VulkanGSRendererUniforms));

  dispatch_generate_keys(generate_keys_pipeline,
                         {
                             &xy_vs_buffer,
                             &inv_cov_vs_opacity_buffer,
                             &depths_buffer,
                             &rect_tile_space_buffer,
                             &index_buffer_offset_buffer,
                             &keys_buffer,
                             &indices_buffer,
                         },
                         uniforms);

  const auto actual_keys = keys_buffer.read_back<std::uint32_t>(static_cast<std::size_t>(num_indices));
  const auto actual_indices = indices_buffer.read_back<std::int32_t>(static_cast<std::size_t>(num_indices));

  REQUIRE(actual_keys.size() == static_cast<std::size_t>(num_indices));
  require_valid_generate_keys_output(actual_keys, actual_indices, uniforms.num_splats);
}
