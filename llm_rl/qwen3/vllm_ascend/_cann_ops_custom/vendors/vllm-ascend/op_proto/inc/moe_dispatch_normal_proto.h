#ifndef MOE_DISPATCH_NORMAL_PROTO_H_
#define MOE_DISPATCH_NORMAL_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MoeDispatchNormal)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(topk_idx, ge::TensorType::ALL())
    .INPUT(send_offset, ge::TensorType::ALL())
    .INPUT(send_tokenIdx, ge::TensorType::ALL())
    .INPUT(recv_offset, ge::TensorType::ALL())
    .INPUT(recv_count, ge::TensorType::ALL())
    .OUTPUT(recv_x, ge::TensorType::ALL())
    .OUTPUT(x_scales, ge::TensorType::ALL())
    .OUTPUT(assist_info_for_combine, ge::TensorType::ALL())
    .REQUIRED_ATTR(group_ep, String)
    .REQUIRED_ATTR(ep_world_size, Int)
    .REQUIRED_ATTR(ep_rank_id, Int)
    .ATTR(group_tp, String, "")
    .ATTR(tp_world_size, Int, 0)
    .ATTR(tp_rank_id, Int, 0)
    .REQUIRED_ATTR(moe_expert_num, Int)
    .ATTR(quant_mode, Int, 0)
    .ATTR(global_bs, Int, 0)
    .OP_END_FACTORY_REG(MoeDispatchNormal);

}

#endif
