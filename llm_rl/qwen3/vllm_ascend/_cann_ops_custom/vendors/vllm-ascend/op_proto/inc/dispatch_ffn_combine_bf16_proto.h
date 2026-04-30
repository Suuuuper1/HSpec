#ifndef DISPATCH_FFN_COMBINE_BF16_PROTO_H_
#define DISPATCH_FFN_COMBINE_BF16_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(DispatchFFNCombineBF16)
    .INPUT(a, ge::TensorType::ALL())
    .DYNAMIC_INPUT(w1, ge::TensorType::ALL())
    .DYNAMIC_INPUT(w2, ge::TensorType::ALL())
    .INPUT(expertIdx, ge::TensorType::ALL())
    .DYNAMIC_INPUT(scale1, ge::TensorType::ALL())
    .DYNAMIC_INPUT(scale2, ge::TensorType::ALL())
    .INPUT(probs, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .REQUIRED_ATTR(group, String)
    .ATTR(M, Int, 0)
    .ATTR(transB, Bool, false)
    .ATTR(weightNz, Bool, false)
    .OP_END_FACTORY_REG(DispatchFFNCombineBF16);

}

#endif
