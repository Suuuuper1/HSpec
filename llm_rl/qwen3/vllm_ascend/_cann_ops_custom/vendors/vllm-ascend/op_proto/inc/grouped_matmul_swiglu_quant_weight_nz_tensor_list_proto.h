#ifndef GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_PROTO_H_
#define GROUPED_MATMUL_SWIGLU_QUANT_WEIGHT_NZ_TENSOR_LIST_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(GroupedMatmulSwigluQuantWeightNzTensorList)
    .INPUT(x, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight, ge::TensorType::ALL())
    .DYNAMIC_INPUT(weight_scale, ge::TensorType::ALL())
    .INPUT(x_scale, ge::TensorType::ALL())
    .INPUT(group_list, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(y_scale, ge::TensorType::ALL())
    .OP_END_FACTORY_REG(GroupedMatmulSwigluQuantWeightNzTensorList);

}

#endif
