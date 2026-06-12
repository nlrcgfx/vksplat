#ifndef NLRC_VKSPLAT_CONFIG_SLANG
#define NLRC_VKSPLAT_CONFIG_SLANG

#include "config_generated.slang"

#define ALPHA_THRESHOLD (1.0f / 255.0f)

#ifndef ANTIALIASE_MODE
#define ANTIALIASE_MODE 0
#endif

#define AA_EPS2D 0.3
#define TILE_CULLING_MODE 2
#define NEAR_CLIP 0.01f

#define EXPORT_MODE_VULKAN_FORWARD 0
#define EXPORT_MODE_VULKAN_BACKWARD 1

#define ALPHA_BLEND_MAX_OPACITY 0.999f

#if TILE_HEIGHT >= 16 && TILE_WIDTH >= 16
#define SUBTILE_HEIGHT 4
#define SUBTILE_WIDTH 8
#else
#define SUBTILE_HEIGHT 0
#define SUBTILE_WIDTH 0
#endif

#define SUBTILE_SIZE (SUBTILE_HEIGHT * SUBTILE_WIDTH)
#define NORMALIZE_QUATS_IN_OPTIMIZER 0

static const float M_PI = 3.14159265358979;

#if USE_EMULATED_F32_ATOMIC
void AtomicAddFloat(RWByteAddressBuffer buffer, uint byteAddress, float valueToAdd) {
  uint oldInt = buffer.Load(byteAddress);
  uint newInt;
  uint assumed;

  do {
    assumed = oldInt;
    float fAssumed = asfloat(assumed);
    float fNew = fAssumed + valueToAdd;
    newInt = asuint(fNew);

    uint resultInt;
    buffer.InterlockedCompareExchange(byteAddress, assumed, newInt, resultInt);
    oldInt = resultInt;
  } while (assumed != oldInt);
}
#endif

#endif
