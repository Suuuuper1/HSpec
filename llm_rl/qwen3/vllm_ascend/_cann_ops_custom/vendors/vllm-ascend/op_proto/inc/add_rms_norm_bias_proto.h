#ifndef ADD_RMS_NORM_BIAS_PROTO_H_
#define ADD_RMS_NORM_BIAS_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(AddRmsNormBias)
    .INPUT(x1, ge::TensorType::ALL())
    .INPUT(x2, ge::TensorType::ALL())
    .INPUT(gamma, ge::TensorType::ALL())
    .OPTIONAL_INPUT(beta, ge::TensorType::ALL())
    .OUTPUT(y, ge::TensorType::ALL())
    .OUTPUT(rstd, ge::TensorType::ALL())
    .OUTPUT(x, ge::TensorType::ALL())
    .ATTR(epsilon, Float, 1e-06)
    .OP_END_FACTORY_REG(AddRmsNormBias);

}

#endif
