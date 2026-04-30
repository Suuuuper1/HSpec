#ifndef MOE_COMBINE_NORMAL_PROTO_H_
#define MOE_COMBINE_NORMAL_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MoeCombineNormal)
    .INPUT(recv_x, ge::TensorType::ALL())
    .INPUT(token_src_info, ge::TensorType::ALL())
    .INPUT(ep_recv_counts, ge::TensorType::ALL())
    .INPUT(recv_topk_weights, ge::TensorType::ALL())
    .OPTIONAL_INPUT(tp_recv_counts, ge::TensorType::ALL())
    .OUTPUT(x, ge::TensorType::ALL())
    .REQUIRED_ATTR(ep_group_name, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .ATTR(tp_group_name, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(global_bs, Int, 0)
    .OP_END_FACTORY_REG(MoeCombineNormal);

}

#endif
