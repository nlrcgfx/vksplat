#include "config.slang"

static const uint BLOCK_SIZE = VKSPLAT_SUM_BLOCK_SIZE;

layout(binding = 0) StructuredBuffer<int32_t> g_input;
layout(binding = 1) RWStructuredBuffer<int32_t> g_output;

groupshared int32_t s_data[BLOCK_SIZE];

[numthreads(BLOCK_SIZE, 1, 1)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 dispatchThreadID: SV_DispatchThreadID, uniform uint32_t arraySize) {
  uint tid = groupThreadID.x;
  uint gid = dispatchThreadID.x;

  s_data[tid] = (gid < arraySize) ? g_input[gid] : 0;
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      s_data[tid] += s_data[tid + stride];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (tid == 0) {
    if (arraySize <= BLOCK_SIZE) {
      g_output[0] = s_data[0];
    } else {
      InterlockedAdd(g_output[0], s_data[0]);
    }
  }
}
