#include <catch2/catch_test_macros.hpp>

#include "gpu/compute_pipeline.hpp"
#include "gpu/headless_context.hpp"
#include "gpu_available.hpp"
#include "smoke_spirv.hpp"

TEST_CASE("Dispatch embedded smoke compute shader", "[gpu]") {
  NLRC_REQUIRE_GPU();

  const nlrc::vksplat::gpu::HeadlessContext context;
  nlrc::vksplat::gpu::ComputePipeline pipeline(context, &nlrc::vksplat::shaders::kSmokeSpirv[0],
                                               nlrc::vksplat::shaders::kSmokeSpirvWordCount);

  REQUIRE_NOTHROW(pipeline.dispatch(1, 1, 1));
}
