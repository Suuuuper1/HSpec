#ifndef SPARSE_FLASH_ATTENTION_PROTO_H_
#define SPARSE_FLASH_ATTENTION_PROTO_H_

#include "graph/operator_reg.h"
#include "register/op_impl_registry.h"

namespace ge {

REG_OP(SparseFlashAttention)
    .INPUT(query, ge::TensorType::ALL())
    .INPUT(key, ge::TensorType::ALL())
    .INPUT(value, ge::TensorType::ALL())
    .INPUT(sparse_indices, ge::TensorType::ALL())
    .OPTIONAL_INPUT(block_table, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_query, ge::TensorType::ALL())
    .OPTIONAL_INPUT(actual_seq_lengths_kv, ge::TensorType::ALL())
    .OPTIONAL_INPUT(query_rope, ge::TensorType::ALL())
    .OPTIONAL_INPUT(key_rope, ge::TensorType::ALL())
    .OUTPUT(attention_out, ge::TensorType::ALL())
    .REQUIRED_ATTR(scale_value, Float)
    .REQUIRED_ATTR(sparse_block_size, Int)
    .ATTR(layout_query, String, "BSND")
    .ATTR(layout_kv, String, "BSND")
    .ATTR(sparse_mode, Int, 3)
    .OP_END_FACTORY_REG(SparseFlashAttention);

}

#endif
