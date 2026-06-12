#define CUMSUM_PHASE_singlePassPrefixSum 0
#define CUMSUM_PHASE_blockScan 1
#define CUMSUM_PHASE_scanBlockSums 2
#define CUMSUM_PHASE_addBlockOffsets 3

#ifndef CUMSUM_PHASE
#define CUMSUM_PHASE -1
#endif

#if CUMSUM_PHASE == CUMSUM_PHASE_blockScan
#define blockScan main
#elif CUMSUM_PHASE == CUMSUM_PHASE_scanBlockSums
#define scanBlockSums main
#elif CUMSUM_PHASE == CUMSUM_PHASE_addBlockOffsets
#define addBlockOffsets main
#elif CUMSUM_PHASE == CUMSUM_PHASE_singlePassPrefixSum
#define singlePassPrefixSum main
#endif

#include "config.slang"

static const uint BLOCK_SIZE_0 = VKSPLAT_CUMSUM_BLOCK_SIZE;
#define BLOCK_SIZE min(BLOCK_SIZE_0, SUBGROUP_SIZE * SUBGROUP_SIZE * SUBGROUP_SIZE)

layout(binding = 0) StructuredBuffer<int32_t> g_input;
layout(binding = 1) RWStructuredBuffer<int32_t> g_output;
layout(binding = 2) RWStructuredBuffer<int32_t> g_blockSums;

groupshared int32_t s_data[BLOCK_SIZE_0];

struct Uniforms {
  uint32_t numElements;
};

int32_t subgroupScan(int32_t val, uint laneId) {
  [ForceUnroll]
  for (uint offset = 1; offset < SUBGROUP_SIZE; offset <<= 1) {
    int32_t temp = WaveReadLaneAt(val, laneId - offset);
    if (laneId >= offset) {
      val += temp;
    }
  }
  return val;
}

[numthreads(BLOCK_SIZE_0, 1, 1)]
void blockScan(uint3 groupId: SV_GroupID, uint3 localId: SV_GroupThreadID, uniform Uniforms uniforms) {
  uint tid = localId.x;
  if (tid >= BLOCK_SIZE) {
    return;
  }

  uint blockId = groupId.x;
  uint gid = blockId * BLOCK_SIZE + tid;

  s_data[tid] = (gid < uniforms.numElements) ? g_input[gid] : 0;
  GroupMemoryBarrierWithGroupSync();

  uint subgroupId = tid / SUBGROUP_SIZE;
  uint laneId = tid % SUBGROUP_SIZE;

  int32_t subgroupSum = subgroupScan(s_data[tid], laneId);
  s_data[tid] = subgroupSum;
  GroupMemoryBarrierWithGroupSync();

  if (tid < (BLOCK_SIZE / SUBGROUP_SIZE)) {
    int32_t subgroupSumVal = s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1];
    int32_t blockScanResult = subgroupSumVal + WavePrefixSum(subgroupSumVal);
    s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1] = blockScanResult;
  }
  GroupMemoryBarrierWithGroupSync();

  if (subgroupId > 0 && laneId != SUBGROUP_SIZE - 1) {
    int32_t subgroupOffset = s_data[subgroupId * SUBGROUP_SIZE - 1];
    s_data[tid] += subgroupOffset;
  }
  GroupMemoryBarrierWithGroupSync();

  if (gid < uniforms.numElements) {
    g_output[gid] = s_data[tid];
  }

  if (tid == BLOCK_SIZE - 1) {
    g_blockSums[blockId] = s_data[tid];
  }
}

[numthreads(BLOCK_SIZE_0, 1, 1)]
void scanBlockSums(uint3 groupId: SV_GroupID, uint3 localId: SV_GroupThreadID, uniform Uniforms uniforms) {
  uint tid = localId.x;
  if (tid >= BLOCK_SIZE) {
    return;
  }

  uint blockId = groupId.x;
  uint gid = blockId * BLOCK_SIZE + tid;

  s_data[tid] = (gid < uniforms.numElements) ? g_blockSums[gid] : 0;
  GroupMemoryBarrierWithGroupSync();

  uint subgroupId = tid / SUBGROUP_SIZE;
  uint laneId = tid % SUBGROUP_SIZE;

  int32_t subgroupSum = subgroupScan(s_data[tid], laneId);
  s_data[tid] = subgroupSum;
  GroupMemoryBarrierWithGroupSync();

  if (tid < (BLOCK_SIZE / SUBGROUP_SIZE)) {
    int32_t subgroupSumVal = s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1];
    int32_t blockScanResult = subgroupSumVal + WavePrefixSum(subgroupSumVal);
    s_data[tid * SUBGROUP_SIZE + SUBGROUP_SIZE - 1] = blockScanResult;
  }
  GroupMemoryBarrierWithGroupSync();

  if (subgroupId > 0 && laneId != SUBGROUP_SIZE - 1) {
    int32_t subgroupOffset = s_data[subgroupId * SUBGROUP_SIZE - 1];
    s_data[tid] += subgroupOffset;
  }
  GroupMemoryBarrierWithGroupSync();

  if (gid < uniforms.numElements) {
    g_blockSums[gid] = s_data[tid];
  }
}

[numthreads(BLOCK_SIZE_0, 1, 1)]
void addBlockOffsets(uint3 groupId: SV_GroupID, uint3 localId: SV_GroupThreadID, uniform Uniforms uniforms) {
  uint tid = localId.x;
  if (tid >= BLOCK_SIZE) {
    return;
  }

  uint blockId = groupId.x;
  uint gid = blockId * BLOCK_SIZE + tid;

  if (gid < uniforms.numElements && blockId > 0) {
    g_output[gid] += g_blockSums[blockId - 1];
  }
}

[numthreads(BLOCK_SIZE_0, 1, 1)]
void singlePassPrefixSum(uint3 localId: SV_GroupThreadID, uint3 globalId: SV_DispatchThreadID, uniform Uniforms uniforms) {
  uint tid = localId.x;
  uint gid = globalId.x;

  int32_t v_in = (gid < uniforms.numElements) ? g_input[gid] : 0;
  s_data[tid] = v_in;
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = 1; stride < BLOCK_SIZE_0; stride <<= 1) {
    if (tid % (stride * 2) == 0) {
      s_data[tid + stride * 2 - 1] += s_data[tid + stride - 1];
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (tid == 0) {
    s_data[BLOCK_SIZE_0 - 1] = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint stride = BLOCK_SIZE_0 / 2; stride > 0; stride >>= 1) {
    if (tid % (stride * 2) == 0) {
      int32_t temp = s_data[tid + stride - 1];
      s_data[tid + stride - 1] = s_data[tid + stride * 2 - 1];
      s_data[tid + stride * 2 - 1] += temp;
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (gid < uniforms.numElements) {
    g_output[gid] = s_data[tid] + v_in;
  }
}
