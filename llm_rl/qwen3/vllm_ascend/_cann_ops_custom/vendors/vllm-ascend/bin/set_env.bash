#!/bin/bash
export ASCEND_CUSTOM_OPP_PATH=/workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=/workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/:${LD_LIBRARY_PATH}
