#ifndef MOE_INIT_ROUTING_CUSTOM_PROTO_H_
#define MOE_INIT_ROUTING_CUSTOM_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(MoeInitRoutingCustom)
    .INPUT(x, ge::TensorType::ALL())
    .INPUT(expert_idx, ge::TensorType::ALL())
    .OPTIONAL_INPUT(scale, ge::TensorType::ALL())
    .OPTIONAL_INPUT(offset, ge::TensorType::ALL())
    .OUTPUT(expanded_x, ge::TensorType::ALL())
    .OUTPUT(expanded_row_idx, ge::TensorType::ALL())
    .OUTPUT(expert_tokens_count_or_cumsum, ge::TensorType::ALL())
    .OUTPUT(expanded_scale, ge::TensorType::ALL())
    .ATTR(active_num, Int, -1)
    .ATTR(expert_capacity, Int, -1)
    .ATTR(expert_num, Int, -1)
    .ATTR(drop_pad_mode, Int, 0)
    .ATTR(expert_tokens_num_type, Int, 0)
    .ATTR(expert_tokens_num_flag, Bool, false)
    .ATTR(quant_mode, Int, -1)
    .ATTR(active_expert_range, ListInt, {})
    .ATTR(row_idx_type, Int, 0)
    .OP_END_FACTORY_REG(MoeInitRoutingCustom);

}

#endif
