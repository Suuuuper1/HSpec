#ifndef HEADER_ACLRTLAUNCH_BGMV_SHRINK_HALF_H
#define HEADER_ACLRTLAUNCH_BGMV_SHRINK_HALF_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_bgmv_shrink_half(uint32_t blockDim, aclrtStream stream, void* x, void* weight, void* indices, uint32_t indicesSize, void* y, uint32_t batchSize, uint32_t numTokensPerCore, uint32_t inputHiddenDim, uint32_t maxLoRARank, float scale);
#endif
