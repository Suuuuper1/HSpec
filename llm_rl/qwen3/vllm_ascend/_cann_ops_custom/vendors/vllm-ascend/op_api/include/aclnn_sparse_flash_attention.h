
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_SPARSE_FLASH_ATTENTION_H_
#define ACLNN_SPARSE_FLASH_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnSparseFlashAttentionGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * sparseIndices : required
 * blockTableOptional : optional
 * actualSeqLengthsQueryOptional : optional
 * actualSeqLengthsKvOptional : optional
 * queryRopeOptional : optional
 * keyRopeOptional : optional
 * scaleValue : required
 * sparseBlockSize : required
 * layoutQueryOptional : optional
 * layoutKvOptional : optional
 * sparseMode : optional
 * out : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseFlashAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *sparseIndices,
    const aclTensor *blockTableOptional,
    const aclTensor *actualSeqLengthsQueryOptional,
    const aclTensor *actualSeqLengthsKvOptional,
    const aclTensor *queryRopeOptional,
    const aclTensor *keyRopeOptional,
    double scaleValue,
    int64_t sparseBlockSize,
    char *layoutQueryOptional,
    char *layoutKvOptional,
    int64_t sparseMode,
    const aclTensor *out,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnSparseFlashAttention
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnSparseFlashAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
