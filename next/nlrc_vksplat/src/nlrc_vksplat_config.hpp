#pragma once

#include <array>
#include <cstdint>

// Canonical VkSplat configuration for the rewrite.
//
// Keep VKSPLAT_* values in the simple #define form below. The shader config
// generator parses these definitions and emits Slang/GLSL fragments from them.

#ifndef VKSPLAT_SUBGROUP_SIZE
#define VKSPLAT_SUBGROUP_SIZE 32
#endif

#define VKSPLAT_TILE_HEIGHT 16
#define VKSPLAT_TILE_WIDTH 16
#define VKSPLAT_TILE_SIZE (VKSPLAT_TILE_HEIGHT * VKSPLAT_TILE_WIDTH)
#define VKSPLAT_SH_REORDER_SIZE VKSPLAT_SUBGROUP_SIZE

#ifndef VKSPLAT_USE_EMULATED_INT64
#define VKSPLAT_USE_EMULATED_INT64 0
#endif

#ifndef VKSPLAT_USE_EMULATED_F32_ATOMIC
#define VKSPLAT_USE_EMULATED_F32_ATOMIC 0
#endif

#define VKSPLAT_SORTING_KEY_BITS 32

#if VKSPLAT_USE_EMULATED_INT64
#define VKSPLAT_RECT_TILE_SPACE_WORDS 2
#else
#define VKSPLAT_RECT_TILE_SPACE_WORDS 1
#endif

#define VKSPLAT_SHADER_REQUIRES_INT64 ((VKSPLAT_USE_EMULATED_INT64 == 0) || (VKSPLAT_SORTING_KEY_BITS == 64))

#define VKSPLAT_RADIX_SORT_RADIX 256
#define VKSPLAT_RADIX_WORKGROUP_SIZE 512
#define VKSPLAT_RADIX_PARTITION_DIVISION 8
#define VKSPLAT_RADIX_PARTITION_SIZE (VKSPLAT_RADIX_PARTITION_DIVISION * VKSPLAT_RADIX_WORKGROUP_SIZE)
#define VKSPLAT_RADIX_BITS_PER_PASS 8

#define VKSPLAT_TILE_SHADER_GENERATE_KEYS_BLOCK_SIZE 64
#define VKSPLAT_TILE_SHADER_TILE_RANGES_THREADS 256
#define VKSPLAT_CUMSUM_BLOCK_SIZE 512
#define VKSPLAT_SUM_BLOCK_SIZE 512
#define VKSPLAT_WHERE_BLOCK_SIZE 256
#define VKSPLAT_DEFAULT_GROUP_SIZE 256
#define VKSPLAT_MCMC_GROUP_SIZE 256
#define VKSPLAT_MCMC_GROUP_SIZE_SPARSE 64
#define VKSPLAT_MORTON_STATS_THREADS (VKSPLAT_SUBGROUP_SIZE * VKSPLAT_SUBGROUP_SIZE)
#define VKSPLAT_MORTON_GENERATE_KEYS_THREADS 256
#define VKSPLAT_MORTON_APPLY_THREADS 512
#define VKSPLAT_MORTON_SORT_KEY_BITS 32
#define VKSPLAT_RASTERIZE_BWD_PER_SPLAT_THREADS 128
#define VKSPLAT_SSIM_BLOCK_X 16
#define VKSPLAT_SSIM_BLOCK_Y 16
#define VKSPLAT_SSIM_HALO 5

#define VKSPLAT_TENSOR_BWD_CONFIG_COUNT 3
#define VKSPLAT_TENSOR_BWD_CONFIG_0_USE_SUBGROUP_OPERATIONS 0
#define VKSPLAT_TENSOR_BWD_CONFIG_0_SPLAT_BATCH_SIZE 8
#define VKSPLAT_TENSOR_BWD_CONFIG_0_GROUP_REDUCE_BEFORE_ATOMIC 0
#define VKSPLAT_TENSOR_BWD_CONFIG_1_USE_SUBGROUP_OPERATIONS 0
#define VKSPLAT_TENSOR_BWD_CONFIG_1_SPLAT_BATCH_SIZE 8
#define VKSPLAT_TENSOR_BWD_CONFIG_1_GROUP_REDUCE_BEFORE_ATOMIC 8
#define VKSPLAT_TENSOR_BWD_CONFIG_2_USE_SUBGROUP_OPERATIONS 1
#define VKSPLAT_TENSOR_BWD_CONFIG_2_SPLAT_BATCH_SIZE 16
#define VKSPLAT_TENSOR_BWD_CONFIG_2_GROUP_REDUCE_BEFORE_ATOMIC 0

namespace nlrc::vksplat {

#if VKSPLAT_SORTING_KEY_BITS == 32
using SortingKey = uint32_t;
#elif VKSPLAT_SORTING_KEY_BITS == 64
using SortingKey = uint64_t;
#else
#error "VKSPLAT_SORTING_KEY_BITS must be 32 or 64"
#endif

#if VKSPLAT_USE_EMULATED_INT64
using RectTileSpace = int32_t;
#else
using RectTileSpace = int64_t;
#endif

struct TensorBwdConfig final {
  bool use_subgroup_operations;
  int splat_batch_size;
  int group_reduce_before_atomic;
  uint32_t min_shared_mem;
};

[[nodiscard]] constexpr uint32_t
tensor_bwd_shared_memory(bool use_subgroup_operations, int splat_batch_size, int group_reduce_before_atomic) {
  constexpr uint32_t kTileSize = VKSPLAT_TILE_SIZE;

  uint32_t words = 0;
  words += kTileSize * 1;
  words += kTileSize * 2;
  words += kTileSize * 4;
  words += kTileSize * 3;
  words += kTileSize * splat_batch_size * 2;

  if (!use_subgroup_operations) {
    words += kTileSize * 6;
    words += kTileSize * 3;
  }

  if (group_reduce_before_atomic > 0) {
    words += kTileSize * 6;
    if (use_subgroup_operations) {
      words += kTileSize * 3;
    }
  }

  return words * sizeof(float);
}

inline constexpr std::array<TensorBwdConfig, VKSPLAT_TENSOR_BWD_CONFIG_COUNT> kTensorBwdConfigs = {{
    {VKSPLAT_TENSOR_BWD_CONFIG_0_USE_SUBGROUP_OPERATIONS != 0, VKSPLAT_TENSOR_BWD_CONFIG_0_SPLAT_BATCH_SIZE,
     VKSPLAT_TENSOR_BWD_CONFIG_0_GROUP_REDUCE_BEFORE_ATOMIC,
     tensor_bwd_shared_memory(VKSPLAT_TENSOR_BWD_CONFIG_0_USE_SUBGROUP_OPERATIONS != 0,
                              VKSPLAT_TENSOR_BWD_CONFIG_0_SPLAT_BATCH_SIZE,
                              VKSPLAT_TENSOR_BWD_CONFIG_0_GROUP_REDUCE_BEFORE_ATOMIC)},
    {VKSPLAT_TENSOR_BWD_CONFIG_1_USE_SUBGROUP_OPERATIONS != 0, VKSPLAT_TENSOR_BWD_CONFIG_1_SPLAT_BATCH_SIZE,
     VKSPLAT_TENSOR_BWD_CONFIG_1_GROUP_REDUCE_BEFORE_ATOMIC,
     tensor_bwd_shared_memory(VKSPLAT_TENSOR_BWD_CONFIG_1_USE_SUBGROUP_OPERATIONS != 0,
                              VKSPLAT_TENSOR_BWD_CONFIG_1_SPLAT_BATCH_SIZE,
                              VKSPLAT_TENSOR_BWD_CONFIG_1_GROUP_REDUCE_BEFORE_ATOMIC)},
    {VKSPLAT_TENSOR_BWD_CONFIG_2_USE_SUBGROUP_OPERATIONS != 0, VKSPLAT_TENSOR_BWD_CONFIG_2_SPLAT_BATCH_SIZE,
     VKSPLAT_TENSOR_BWD_CONFIG_2_GROUP_REDUCE_BEFORE_ATOMIC,
     tensor_bwd_shared_memory(VKSPLAT_TENSOR_BWD_CONFIG_2_USE_SUBGROUP_OPERATIONS != 0,
                              VKSPLAT_TENSOR_BWD_CONFIG_2_SPLAT_BATCH_SIZE,
                              VKSPLAT_TENSOR_BWD_CONFIG_2_GROUP_REDUCE_BEFORE_ATOMIC)},
}};

inline constexpr bool kShaderRequiresInt64 = VKSPLAT_SHADER_REQUIRES_INT64;

} // namespace nlrc::vksplat
