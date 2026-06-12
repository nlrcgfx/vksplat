#include <cstdint>
#include <stdexcept>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include "gpu/compute_pipeline.hpp"
#include "gpu/headless_context.hpp"
#include "gpu/storage_buffer.hpp"
#include "gpu_available.hpp"
#include "nlrc_vksplat_config.hpp"
#include "smoke_spirv.hpp"
#include "span.hpp"

TEST_CASE("Dispatch embedded smoke compute shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const nlrc::vksplat::gpu::HeadlessContext context;
  nlrc::vksplat::gpu::ComputePipeline pipeline(context, nlrc::vksplat::make_span(nlrc::vksplat::shaders::kSmokeSpirv));

  REQUIRE_NOTHROW(pipeline.dispatch({1, 1, 1}));
}

TEST_CASE("ComputePipeline rejects invalid constructor inputs", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const nlrc::vksplat::gpu::HeadlessContext context;
  const std::vector<std::uint32_t> empty_spirv;

  REQUIRE_THROWS_AS(nlrc::vksplat::gpu::ComputePipeline(context, nlrc::vksplat::make_span(empty_spirv)),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(nlrc::vksplat::gpu::ComputePipeline(context,
                                                        nlrc::vksplat::make_span(nlrc::vksplat::shaders::kSmokeSpirv),
                                                        0, nlrc::vksplat::kMaxPushConstantBytes + 1),
                    std::invalid_argument);
}

TEST_CASE("ComputePipeline rejects descriptor count mismatch", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const nlrc::vksplat::gpu::HeadlessContext context;
  nlrc::vksplat::gpu::ComputePipeline pipeline(context, nlrc::vksplat::make_span(nlrc::vksplat::shaders::kSmokeSpirv),
                                               1);
  const std::vector<const nlrc::vksplat::gpu::StorageBuffer *> no_buffers;
  REQUIRE_THROWS_AS(pipeline.bind_storage_buffers(no_buffers), std::invalid_argument);
}

TEST_CASE("ComputePipeline rejects null descriptor bindings", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const nlrc::vksplat::gpu::HeadlessContext context;
  nlrc::vksplat::gpu::ComputePipeline pipeline(context, nlrc::vksplat::make_span(nlrc::vksplat::shaders::kSmokeSpirv),
                                               1);

  const std::vector<const nlrc::vksplat::gpu::StorageBuffer *> null_buffer{nullptr};
  REQUIRE_THROWS_AS(pipeline.bind_storage_buffers(null_buffer), std::invalid_argument);
}

TEST_CASE("ComputePipeline rejects dispatch with unbound descriptors", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const nlrc::vksplat::gpu::HeadlessContext context;
  nlrc::vksplat::gpu::ComputePipeline pipeline(context, nlrc::vksplat::make_span(nlrc::vksplat::shaders::kSmokeSpirv),
                                               1);

  REQUIRE_THROWS_AS(pipeline.dispatch({1, 1, 1}), std::logic_error);
}

TEST_CASE("ComputePipeline validates push constant view size", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const nlrc::vksplat::gpu::HeadlessContext context;
  nlrc::vksplat::gpu::ComputePipeline pipeline(context, nlrc::vksplat::make_span(nlrc::vksplat::shaders::kSmokeSpirv),
                                               0, sizeof(std::uint32_t));
  const std::uint64_t wrong_size = 0;

  REQUIRE_THROWS_AS(pipeline.dispatch({1, 1, 1}), std::invalid_argument);
  REQUIRE_THROWS_AS(pipeline.dispatch({1, 1, 1}, nlrc::vksplat::ByteView::from_object(wrong_size)),
                    std::invalid_argument);
}
