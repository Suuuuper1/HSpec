#ifndef LIGHTNING_INDEXER_PROTO_H_
#define LIGHTNING_INDEXER_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(LightningIndexer)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(weights, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_query, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_key, ge::TensorType::ALL())
    .OPTIONAL_INPUT(block_table, ge::TensorType::ALL())
    .OUTPUT(sparse_indices, ge::TensorType::ALL())
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_key, String, "PA_BSND")
    .ATTR(sparse_count, Int, 2048)
    .ATTR(sparse_mode, Int, 3)
    .OP_END_FACTORY_REG(LightningIndexer);

}

#endif
