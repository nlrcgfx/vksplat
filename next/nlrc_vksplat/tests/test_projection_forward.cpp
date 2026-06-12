#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "golden_compare.hpp"
#include "gpu/compute_pipeline.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "nlrc_vksplat_config.hpp"
#include "projection_forward_spirv.hpp"
#include "span.hpp"

namespace {

inline constexpr std::uint32_t kProjectionForwardBindingCount = 11;
inline constexpr std::uint32_t kImageHeight = 32;
inline constexpr std::uint32_t kImageWidth = 32;
inline constexpr std::uint32_t kGridHeight = 2;
inline constexpr std::uint32_t kGridWidth = 2;

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

struct RectBounds final {
  std::uint32_t min_x;
  std::uint32_t min_y;
  std::uint32_t max_x;
  std::uint32_t max_y;
};

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

[[nodiscard]] VulkanGSRendererUniforms projection_uniforms() {
  return {
      kImageHeight,
      kImageWidth,
      kGridHeight,
      kGridWidth,
      1U,
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

[[nodiscard]] RectBounds decode_rect_tile_space(const std::vector<nlrc::vksplat::RectTileSpace> &rect_words) {
  if constexpr (nlrc::vksplat::kUseEmulatedInt64) {
    REQUIRE(rect_words.size() == 2);
    const auto min_word = static_cast<std::uint32_t>(rect_words[0]);
    const auto max_word = static_cast<std::uint32_t>(rect_words[1]);
    return {
        min_word & 0xFFFFU,
        (min_word >> 16U) & 0xFFFFU,
        max_word & 0xFFFFU,
        (max_word >> 16U) & 0xFFFFU,
    };
  } else {
    REQUIRE(rect_words.size() == 1);
    const auto packed = static_cast<std::uint64_t>(rect_words[0]);
    return {
        static_cast<std::uint32_t>(packed & 0xFFFFULL),
        static_cast<std::uint32_t>((packed >> 16U) & 0xFFFFULL),
        static_cast<std::uint32_t>((packed >> 32U) & 0xFFFFULL),
        static_cast<std::uint32_t>((packed >> 48U) & 0xFFFFULL),
    };
  }
}

} // namespace

TEST_CASE("Dispatch projection_forward Subgraph A shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

#if VKSPLAT_USE_EMULATED_INT64
  constexpr const char *kStageName = "projection_forward_emulated_int64";
#else
  constexpr const char *kStageName = "projection_forward_native_int64";
#endif

  const auto fixture_root = nlrc::vksplat::tests::fixture_dir(kStageName);
  const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(fixture_root / "manifest.json");

  const std::vector<std::string> expected_bindings = {
      "xyz_ws", "sh_coeffs", "rotations", "scales_opacs",       "tiles_touched", "rect_tile_space",
      "radii",  "xy_vs",     "depths",    "inv_cov_vs_opacity", "rgb",
  };
  REQUIRE(manifest.bindings == expected_bindings);
  REQUIRE(manifest.bindings.size() == kProjectionForwardBindingCount);
  REQUIRE_FALSE(manifest.profile_agnostic);

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

  REQUIRE(sh_coeffs.size() == static_cast<std::size_t>(12U * VKSPLAT_SH_REORDER_SIZE * 4U));
  REQUIRE(tiles_touched.size() == 1);
  REQUIRE(radii.size() == 1);
  REQUIRE(xy_vs.size() == 2);
  REQUIRE(depths.size() == 1);
  REQUIRE(inv_cov_vs_opacity.size() == 4);
  REQUIRE(rgb.size() == 3);

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

  auto spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kProjectionForwardSpirv);
  nlrc::vksplat::gpu::ComputePipeline pipeline(context, spirv, kProjectionForwardBindingCount,
                                               sizeof(VulkanGSRendererUniforms));

  const StorageBufferMap storage_buffers = {
      {"xyz_ws", &xyz_ws_buffer},
      {"sh_coeffs", &sh_coeffs_buffer},
      {"rotations", &rotations_buffer},
      {"scales_opacs", &scales_opacs_buffer},
      {"tiles_touched", &tiles_touched_buffer},
      {"rect_tile_space", &rect_tile_space_buffer},
      {"radii", &radii_buffer},
      {"xy_vs", &xy_vs_buffer},
      {"depths", &depths_buffer},
      {"inv_cov_vs_opacity", &inv_cov_vs_opacity_buffer},
      {"rgb", &rgb_buffer},
  };
  pipeline.bind_storage_buffers(bindings_from_manifest(manifest, storage_buffers));

  const auto uniforms = projection_uniforms();
  const auto push_constants = nlrc::vksplat::ByteView::from_object(uniforms);
  const auto dispatch_shape = nlrc::vksplat::gpu::DispatchShape{1U, 1U, 1U};
  pipeline.dispatch(dispatch_shape, push_constants);

  const auto actual_tiles_touched = tiles_touched_buffer.read_back<std::int32_t>(1);

  const auto actual_rect_tile_space =
      rect_tile_space_buffer.read_back<nlrc::vksplat::RectTileSpace>(rect_tile_space.size());

  const auto actual_radii = radii_buffer.read_back<std::int32_t>(1);
  const auto actual_xy_vs = xy_vs_buffer.read_back<float>(2);
  const auto actual_depths = depths_buffer.read_back<float>(1);
  const auto actual_inv_cov_vs_opacity = inv_cov_vs_opacity_buffer.read_back<float>(4);
  const auto actual_rgb = rgb_buffer.read_back<float>(3);

  REQUIRE_NOTHROW(nlrc::vksplat::tests::assert_no_nan_inf(nlrc::vksplat::make_span(actual_xy_vs)));
  REQUIRE_NOTHROW(nlrc::vksplat::tests::assert_no_nan_inf(nlrc::vksplat::make_span(actual_depths)));
  REQUIRE_NOTHROW(nlrc::vksplat::tests::assert_no_nan_inf(nlrc::vksplat::make_span(actual_inv_cov_vs_opacity)));
  REQUIRE_NOTHROW(nlrc::vksplat::tests::assert_no_nan_inf(nlrc::vksplat::make_span(actual_rgb)));

  REQUIRE(actual_tiles_touched[0] > 0);
  REQUIRE(actual_tiles_touched[0] <= static_cast<std::int32_t>(kGridWidth * kGridHeight));
  REQUIRE(actual_radii[0] > 0);

  REQUIRE(actual_xy_vs[0] >= 0.0F);
  REQUIRE(actual_xy_vs[0] < static_cast<float>(kImageWidth));

  REQUIRE(actual_xy_vs[1] >= 0.0F);
  REQUIRE(actual_xy_vs[1] < static_cast<float>(kImageHeight));

  REQUIRE(actual_depths[0] > 0.0F);
  REQUIRE(actual_inv_cov_vs_opacity[3] >= 0.0F);
  REQUIRE(actual_inv_cov_vs_opacity[3] <= 1.0F);

  for (const float channel : actual_rgb) {
    REQUIRE(channel >= 0.0F);
    REQUIRE(channel <= 1.0F);
  }

  const RectBounds rect = decode_rect_tile_space(actual_rect_tile_space);
  REQUIRE(rect.min_x <= rect.max_x);
  REQUIRE(rect.min_y <= rect.max_y);
  REQUIRE(rect.max_x <= kGridWidth);
  REQUIRE(rect.max_y <= kGridHeight);
  REQUIRE(rect.min_x < rect.max_x);
  REQUIRE(rect.min_y < rect.max_y);

  const auto rect_area = static_cast<std::int32_t>((rect.max_x - rect.min_x) * (rect.max_y - rect.min_y));
  REQUIRE(actual_tiles_touched[0] <= rect_area);
}
