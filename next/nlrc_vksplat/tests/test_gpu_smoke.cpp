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

using namespace nlrc::vksplat;

TEST_CASE("Dispatch embedded smoke compute shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const gpu::HeadlessContext context;

  auto spirv = make_span(shaders::kSmokeSpirv);
  gpu::ComputePipeline pipeline(context, spirv);

  auto shape = gpu::DispatchShape{1, 1, 1};
  REQUIRE_NOTHROW(pipeline.dispatch(shape));
}

TEST_CASE("ComputePipeline rejects invalid constructor inputs", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const gpu::HeadlessContext context;
  const std::vector<std::uint32_t> empty_spirv;

  auto spirv = make_span(empty_spirv);
  REQUIRE_THROWS_AS(gpu::ComputePipeline(context, spirv), std::invalid_argument);

  const std::size_t push_constants_size = kMaxPushConstantBytes + 1;
  REQUIRE_THROWS_AS(gpu::ComputePipeline(context, spirv, 0, push_constants_size), std::invalid_argument);
}

TEST_CASE("ComputePipeline rejects descriptor count mismatch", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const gpu::HeadlessContext context;

  auto spirv = make_span(shaders::kSmokeSpirv);
  gpu::ComputePipeline pipeline(context, spirv, 1);

  const std::vector<const gpu::StorageBuffer *> no_buffers;
  REQUIRE_THROWS_AS(pipeline.bind_storage_buffers(no_buffers), std::invalid_argument);
}

TEST_CASE("ComputePipeline rejects null descriptor bindings", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const gpu::HeadlessContext context;

  auto spirv = make_span(shaders::kSmokeSpirv);
  gpu::ComputePipeline pipeline(context, spirv, 1);

  const std::vector<const gpu::StorageBuffer *> null_buffer{nullptr};
  REQUIRE_THROWS_AS(pipeline.bind_storage_buffers(null_buffer), std::invalid_argument);
}

TEST_CASE("ComputePipeline rejects dispatch with unbound descriptors", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const gpu::HeadlessContext context;

  auto spirv = make_span(shaders::kSmokeSpirv);
  gpu::ComputePipeline pipeline(context, spirv, 1);

  auto shape = gpu::DispatchShape{1, 1, 1};
  REQUIRE_THROWS_AS(pipeline.dispatch(shape), std::logic_error);
}

TEST_CASE("ComputePipeline validates push constant view size", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const gpu::HeadlessContext context;

  auto spirv = make_span(shaders::kSmokeSpirv);
  gpu::ComputePipeline pipeline(context, spirv, 0, sizeof(std::uint32_t));

  const std::uint64_t wrong_size = 0ULL;

  auto shape = gpu::DispatchShape{1, 1, 1};
  auto push_constants = ByteView::from_object(wrong_size);

  REQUIRE_THROWS_AS(pipeline.dispatch(shape), std::invalid_argument);
  REQUIRE_THROWS_AS(pipeline.dispatch(shape, push_constants), std::invalid_argument);
}
