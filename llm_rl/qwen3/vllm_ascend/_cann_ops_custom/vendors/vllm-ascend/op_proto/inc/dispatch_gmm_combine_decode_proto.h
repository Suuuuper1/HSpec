#ifndef DISPATCH_GMM_COMBINE_DECODE_PROTO_H_
#define DISPATCH_GMM_COMBINE_DECODE_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(DispatchGmmCombineDecode)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(expert_ids, ge::TensorType::ALL())
    .DYNAMIC_INPUT(gmm1_permuted_weight, ge::TensorType::ALL())
    .DYNAMIC_INPUT(gmm1_permuted_weight_scale, ge::TensorType::ALL())
    .DYNAMIC_INPUT(gmm2_weight, ge::TensorType::ALL())
    .DYNAMIC_INPUT(gmm2_weight_scale, ge::TensorType::ALL())
    .INPUT(expert_scales, ge::TensorType::ALL())
    .OPTIONAL_INPUT(expert_smooth_scales, ge::TensorType::ALL())
    .OPTIONAL_INPUT(x_active_mask, ge::TensorType::ALL())
    .OUTPUT(output, ge::TensorType::ALL())
    .OUTPUT(expert_token_nums, ge::TensorType::ALL())
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_rank_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .REQUIRED_ATTR(share_expert_num, Int)
    .REQUIRED_ATTR(share_expert_rank_num, Int)
    .REQUIRED_ATTR(quant_mode, Int)
    .REQUIRED_ATTR(global_bs, Int)
    .OP_END_FACTORY_REG(DispatchGmmCombineDecode);

}

#endif
