#ifndef NOTIFY_DISPATCH_PROTO_H_
#define NOTIFY_DISPATCH_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(NotifyDispatch)
    .INPUT(sendData, ge::TensorType::ALL())
    .INPUT(tokenPerExpertData, ge::TensorType::ALL())
    .OUTPUT(sendDataOffset, ge::TensorType::ALL())
    .OUTPUT(recvData, ge::TensorType::ALL())
    .REQUIRED_ATTR(sendCount, Int)
    .REQUIRED_ATTR(num_tokens, Int)
    .REQUIRED_ATTR(comm_group, String)
    .REQUIRED_ATTR(rank_size, Int)
    .REQUIRED_ATTR(rank_id, Int)
    .REQUIRED_ATTR(local_rank_size, Int)
    .REQUIRED_ATTR(local_rank_id, Int)
    .OP_END_FACTORY_REG(NotifyDispatch);

}

#endif
