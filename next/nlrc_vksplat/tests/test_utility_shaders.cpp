#include <array>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "cumsum_single_pass_spirv.hpp"
#include "fixture_loader.hpp"
#include "fixture_manifest.hpp"
#include "gpu/compute_pipeline.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "nlrc_vksplat_config.hpp"
#include "span.hpp"
#include "sum_spirv.hpp"
#include "where_spirv.hpp"

namespace {

using StorageBufferMap = std::map<std::string, const nlrc::vksplat::gpu::StorageBuffer *>;

struct ElementCountPushConstants final {
  std::uint32_t num_elements;
};

[[nodiscard]] std::vector<const nlrc::vksplat::gpu::StorageBuffer *>
bindings_from_manifest(const nlrc::vksplat::tests::FixtureManifest &manifest, const StorageBufferMap &buffers) {
  std::vector<const nlrc::vksplat::gpu::StorageBuffer *> bindings;
  bindings.reserve(manifest.bindings.size());

  for (const auto &name : manifest.bindings) {
    bindings.push_back(buffers.at(name));
  }

  return bindings;
}

[[nodiscard]] std::uint32_t binding_count(const nlrc::vksplat::tests::FixtureManifest &manifest) {
  return static_cast<std::uint32_t>(manifest.bindings.size());
}

[[nodiscard]] auto make_storage_buffer(const nlrc::vksplat::gpu::HeadlessContext &context,
                                       const std::vector<std::int32_t> &values) -> nlrc::vksplat::gpu::StorageBuffer {
  const std::size_t size_bytes = values.size() * sizeof(std::int32_t);
  nlrc::vksplat::gpu::StorageBuffer buffer(context, size_bytes);

  buffer.upload(values);
  return buffer;
}

[[nodiscard]] auto dispatch_groups_for(std::size_t element_count, std::uint32_t block_size)
    -> nlrc::vksplat::gpu::DispatchShape {
  const auto groups = static_cast<std::uint32_t>((element_count + block_size - 1U) / block_size);
  return {groups, 1U, 1U};
}

} // namespace

TEST_CASE("Dispatch cumsum single-pass utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 3> cases = {
      "D_cumsum_single_pass",
      "D_cumsum_single_pass_near_block",
      "D_cumsum_single_pass_exact_block",
  };

  for (const auto *stage_name : cases) {
    INFO("stage: " << stage_name);

    const auto fixture_root = nlrc::vksplat::tests::fixture_dir(stage_name);
    const auto golden_root = nlrc::vksplat::tests::golden_dir(stage_name);

    const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = nlrc::vksplat::tests::load_fixture_manifest(golden_root / "manifest.json");

    const auto input = nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "input");

    const auto initial_output =
        nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "output");

    const auto block_sums =
        nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "block_sums");

    REQUIRE(input.size() <= VKSPLAT_CUMSUM_BLOCK_SIZE);

    const nlrc::vksplat::gpu::HeadlessContext context;

    auto input_buffer = make_storage_buffer(context, input);
    auto output_buffer = make_storage_buffer(context, initial_output);
    auto block_sums_buffer = make_storage_buffer(context, block_sums);

    auto spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kCumsumSinglePassSpirv);
    nlrc::vksplat::gpu::ComputePipeline pipeline(context, spirv, binding_count(manifest),
                                                 sizeof(ElementCountPushConstants));

    const StorageBufferMap storage_buffers = {
        {"input", &input_buffer},
        {"output", &output_buffer},
        {"block_sums", &block_sums_buffer},
    };

    pipeline.bind_storage_buffers(bindings_from_manifest(manifest, storage_buffers));

    const ElementCountPushConstants push_constants = {static_cast<std::uint32_t>(input.size())};
    auto push_constants_view = nlrc::vksplat::ByteView::from_object(push_constants);
    auto dispatch_shape = nlrc::vksplat::gpu::DispatchShape{1U, 1U, 1U};

    pipeline.dispatch(dispatch_shape, push_constants_view);

    const auto expected =
        nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "output");

    const auto actual = output_buffer.read_back<std::int32_t>(expected.size());

    REQUIRE(actual == expected);
  }
}

TEST_CASE("Dispatch sum utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 2> cases = {
      "D_sum",
      "D_sum_multi_block",
  };

  for (const auto *stage_name : cases) {
    INFO("stage: " << stage_name);

    const auto fixture_root = nlrc::vksplat::tests::fixture_dir(stage_name);
    const auto golden_root = nlrc::vksplat::tests::golden_dir(stage_name);

    const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = nlrc::vksplat::tests::load_fixture_manifest(golden_root / "manifest.json");

    const auto input = nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "input");

    const auto initial_output =
        nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "output");

    const nlrc::vksplat::gpu::HeadlessContext context;

    auto input_buffer = make_storage_buffer(context, input);
    auto output_buffer = make_storage_buffer(context, initial_output);

    auto spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kSumSpirv);
    nlrc::vksplat::gpu::ComputePipeline pipeline(context, spirv, binding_count(manifest),
                                                 sizeof(ElementCountPushConstants));

    const StorageBufferMap storage_buffers = {
        {"input", &input_buffer},
        {"output", &output_buffer},
    };

    pipeline.bind_storage_buffers(bindings_from_manifest(manifest, storage_buffers));

    const ElementCountPushConstants push_constants{static_cast<std::uint32_t>(input.size())};
    auto push_constants_view = nlrc::vksplat::ByteView::from_object(push_constants);
    auto dispatch_shape = dispatch_groups_for(input.size(), VKSPLAT_SUM_BLOCK_SIZE);
    pipeline.dispatch(dispatch_shape, push_constants_view);

    const auto expected =
        nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "output");

    const auto actual = output_buffer.read_back<std::int32_t>(expected.size());

    REQUIRE(actual == expected);
  }
}

TEST_CASE("Dispatch where utility shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const std::array<const char *, 4> cases = {
      "D_where",
      "D_where_no_true",
      "D_where_first_last",
      "D_where_block_boundary",
  };

  for (const auto *stage_name : cases) {
    INFO("stage: " << stage_name);

    const auto fixture_root = nlrc::vksplat::tests::fixture_dir(stage_name);
    const auto golden_root = nlrc::vksplat::tests::golden_dir(stage_name);

    const auto manifest = nlrc::vksplat::tests::load_fixture_manifest(fixture_root / "manifest.json");
    const auto golden_manifest = nlrc::vksplat::tests::load_fixture_manifest(golden_root / "manifest.json");

    const auto mask = nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "mask");

    const auto mask_cumsum =
        nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "mask_cumsum");

    const auto initial_out_indices =
        nlrc::vksplat::tests::load_fixture_buffer<std::int32_t>(fixture_root, manifest, "out_indices");

    const nlrc::vksplat::gpu::HeadlessContext context;

    auto mask_buffer = make_storage_buffer(context, mask);
    auto mask_cumsum_buffer = make_storage_buffer(context, mask_cumsum);
    auto out_indices_buffer = make_storage_buffer(context, initial_out_indices);

    auto spirv = nlrc::vksplat::make_span(nlrc::vksplat::shaders::kWhereSpirv);
    nlrc::vksplat::gpu::ComputePipeline pipeline(context, spirv, binding_count(manifest),
                                                 sizeof(ElementCountPushConstants));

    const StorageBufferMap storage_buffers = {
        {"mask", &mask_buffer},
        {"mask_cumsum", &mask_cumsum_buffer},
        {"out_indices", &out_indices_buffer},
    };

    pipeline.bind_storage_buffers(bindings_from_manifest(manifest, storage_buffers));

    const ElementCountPushConstants push_constants{static_cast<std::uint32_t>(mask.size())};
    auto push_constants_view = nlrc::vksplat::ByteView::from_object(push_constants);
    auto dispatch_shape = dispatch_groups_for(mask.size(), VKSPLAT_WHERE_BLOCK_SIZE);

    pipeline.dispatch(dispatch_shape, push_constants_view);

    using nlrc::vksplat::tests::load_fixture_buffer;
    const auto expected = load_fixture_buffer<std::int32_t>(golden_root, golden_manifest, "out_indices");
    const auto actual = out_indices_buffer.read_back<std::int32_t>(expected.size());
    REQUIRE(actual == expected);
  }
}
