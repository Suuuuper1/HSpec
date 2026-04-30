
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_MOE_GATING_TOP_K_H_
#define ACLNN_MOE_GATING_TOP_K_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnMoeGatingTopKGetWorkspaceSize
 * parameters :
 * x : required
 * biasOptional : optional
 * k : required
 * kGroup : optional
 * groupCount : optional
 * groupSelectMode : optional
 * renorm : optional
 * normType : optional
 * outFlag : optional
 * routedScalingFactor : optional
 * eps : optional
 * yOut : required
 * expertIdxOut : required
 * outOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeGatingTopKGetWorkspaceSize(
    const aclTensor *x,
    const aclTensor *biasOptional,
    int64_t k,
    int64_t kGroup,
    int64_t groupCount,
    int64_t groupSelectMode,
    int64_t renorm,
    int64_t normType,
    bool outFlag,
    double routedScalingFactor,
    double eps,
    const aclTensor *yOut,
    const aclTensor *expertIdxOut,
    const aclTensor *outOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnMoeGatingTopK
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnMoeGatingTopK(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
