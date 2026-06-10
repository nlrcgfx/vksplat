#pragma once

#include <cstdint>
#include "vksplat_config.h"

#define ENABLE_VULKAN_VALIDATION_LAYER 0
#define ENABLE_ASSERTION 0

#define SUBGROUP_SIZE VKSPLAT_SUBGROUP_SIZE
#define TILE_HEIGHT VKSPLAT_TILE_HEIGHT
#define TILE_WIDTH VKSPLAT_TILE_WIDTH
#define TILE_SIZE VKSPLAT_TILE_SIZE

// reordering for better memory colaescing
// see config.slang for details
#define SH_REORDER_SIZE VKSPLAT_SH_REORDER_SIZE
// #define SH_REORDER_SIZE 1

#define RASTERIZE_BACKWARD_USE_SCHEDULING 1


#include <cstdio>
#define _DEBUG_PRINT do { printf("%s %d\n", __FILE__, __LINE__); fflush(stdout); } while(0)
// #define _DEBUG_PRINT ;
#include <cassert>

#include <stdexcept>

#define _THROW_ERROR_ALWAYS(message) do { \
    std::string msg = std::string(message) + \
        ". From file `" + __FILE__ + "`, line " + std::to_string(__LINE__); \
    printf("\033[91m%s\033[m\n", msg.c_str()); fflush(stdout); \
    throw std::runtime_error(msg); \
  } while(0)

#if ENABLE_ASSERTION
#define _THROW_ERROR(...) _THROW_ERROR_ALWAYS(__VA_ARGS__)
#else
#define _THROW_ERROR(...) do { } while(0)
#endif

#define _CEIL_DIV(x,m) (((x)+(m)-1)/(m))
#define _CEIL_ROUND(x,m) (_CEIL_DIV(x,m)*(m))
