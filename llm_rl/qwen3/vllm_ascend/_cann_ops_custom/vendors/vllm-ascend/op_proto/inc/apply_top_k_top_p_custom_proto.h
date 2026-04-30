#ifndef APPLY_TOP_K_TOP_P_CUSTOM_PROTO_H_
#define APPLY_TOP_K_TOP_P_CUSTOM_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(ApplyTopKTopPCustom)
    .INPUT(sorted_value, ge::TensorType::ALL())
    .INPUT(sorted_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(p, ge::TensorType::ALL())
    .OPTIONAL_INPUT(k, ge::TensorType::ALL())
    .OUTPUT(out, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(ApplyTopKTopPCustom);

}

#endif
