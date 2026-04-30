#ifndef HEADER_ACLRTLAUNCH_BATCH_MATMUL_TRANSPOSE_H
#define HEADER_ACLRTLAUNCH_BATCH_MATMUL_TRANSPOSE_H
#include "acl/acl_base.h"

#ifndef ACLRT_LAUNCH_KERNEL
#define ACLRT_LAUNCH_KERNEL(kernel_func) aclrtlaunch_##kernel_func
#endif

extern "C" uint32_t aclrtlaunch_batch_matmul_transpose(uint32_t blockDim, aclrtStream stream, void* gm_a, void* gm_b, void* gm_c, void* gm_tiling_data);
#endif
