#ifndef DISPATCH_LAYOUT_PROTO_H_
#define DISPATCH_LAYOUT_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(DispatchLayout)
    .INPUT(topkIdx, ge::TensorType::ALL())
    .OUTPUT(numTokensPerRank, ge::TensorType::ALL())
    .OUTPUT(numTokensPerExpert, ge::TensorType::ALL())
    .OUTPUT(isTokenInRank, ge::TensorType::ALL())
    .REQUIRED_ATTR(num_tokens, Int)
    .REQUIRED_ATTR(num_ranks, Int)
    .REQUIRED_ATTR(num_experts, Int)
    .REQUIRED_ATTR(num_topk, Int)
    .OP_END_FACTORY_REG(DispatchLayout);

}

#endif
