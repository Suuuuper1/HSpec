
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_ADD_RMS_NORM_BIAS_H_
#define ACLNN_ADD_RMS_NORM_BIAS_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnAddRmsNormBiasGetWorkspaceSize
 * parameters :
 * x1 : required
 * x2 : required
 * gamma : required
 * betaOptional : optional
 * epsilon : optional
 * yOut : required
 * rstdOut : required
 * xOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddRmsNormBiasGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *gamma,
    const aclTensor *betaOptional,
    double epsilon,
    const aclTensor *yOut,
    const aclTensor *rstdOut,
    const aclTensor *xOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnAddRmsNormBias
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnAddRmsNormBias(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
