# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

pkill -9 python
ray stop --force

# torch
export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="garbage_collection_threshold:0.6,max_split_size_mb:24"
# torch-npu
export TASK_QUEUE_ENABLE=1

# CANN
# The default CANN installation path. If your installation directory differs, please adjust the paths accordingly. 
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

export ASCEND_CUSTOM_OPP_PATH=./vllm_ascend/_cann_ops_custom/vendors/vllm-ascend:${ASCEND_CUSTOM_OPP_PATH}
export LD_LIBRARY_PATH=./vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/:${LD_LIBRARY_PATH}

export ASCEND_LAUNCH_BLOCKING=0             # debug usage, which seriously affects performance after use, but the error stack is accurate
export ASCEND_GLOBAL_EVENT_ENABLE=0         # whether to display the Ascend EVENT log; 1 is for display
export ASCEND_SLOG_PRINT_TO_STDOUT=0        # does the Ascend log screen out? 0 does not screen. By default, it is at /root/ascend/log. 1 directly screens
export ASCEND_GLOBAL_LOG_LEVEL=3            # log level: 3: ERROR  2:EVENT  1:INFO   0:DEBUG
export ASCEND_HOST_LOG_FILE_NUM=1000

# hydra and ray
export HYDRA_FULL_ERROR=1                   # display the accurate error stack
export RAY_DEDUP_LOGS=0                     # 0: disable ray's log folding 1: enable ray's log folding

# HCCL
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export HCCL_IF_BASE_PORT=64021
export HCCL_BUFFSIZE=300
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_ASYNC_ERROR_HANDLING=0

# vLLM
export VLLM_USE_V1=1
export VLLM_LOGGING_LEVEL=INFO

# vLLM-Ascend
# under the configuration of the vLLM log level of INFO, enable this configuration, print the time of prefill and decode
export VLLM_ASCEND_MODEL_EXECUTE_TIME_OBSERVE=0
export VLLM_ASCEND_ENABLE_NZ=0

export PYTHONUNBUFFERED=x

ulimit -n 32768
mkdir logs

NNODES=8                           # [TODO] number of nodes
NPUS_PER_NODE=16                   # the number of npus for each node
MASTER_ADDR="IP FOR MASTER NODE"   # [TODO] modify it to correspond to the IP of the master node
SOCKET_IFNAME="SOCKET IFNAME FOR CURRENT NODE"  # [TODO] use `ifconfig` to check and modify it to the communication network card of the current node.
# obtain the current node IP
CURRENT_IP=$(ifconfig $SOCKET_IFNAME | grep -Eo 'inet (addr:)?([0-9]{1,3}\.){3}[0-9]{1,3}' | awk '{print $NF}')

export MASTER_PORT=29444

export TP_SOCKET_IFNAME=$SOCKET_IFNAME
export HCCL_SOCKET_IFNAME=$SOCKET_IFNAME
export GLOO_SOCKET_IFNAME=$SOCKET_IFNAME

DEFAULT_TRAIN_SCRIPT="./internal/train_grpo_qwen3_235b_128die_true_weight.sh"
TRAIN_SCRIPT="${1:-${SCRIPT:-$DEFAULT_TRAIN_SCRIPT}}"

ENV_SCRIPT="${2:-$SCRIPT}"

if [ -n "$2" ]; then
    source "$ENV_SCRIPT"
fi

echo "Preparing to launch training script: TRAIN_SCRIPT = $TRAIN_SCRIPT"

if [[ ! -f "$TRAIN_SCRIPT" ]]; then
    echo "ERROR: Training script not found: $TRAIN_SCRIPT"
    echo "Please make sure the path is correct!"
    exit 1
fi

# Pre-compile MindSpeed Ops
python -c "import mindspeed; from mindspeed.op_builder import RotaryPositionEmbeddingOpBuilder; RotaryPositionEmbeddingOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import RingAttentionUpdateOpBuilder; RingAttentionUpdateOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import GMMOpBuilder; GMMOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import GMMV2OpBuilder; GMMV2OpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder.fused_adamw_v2_builder import FusedAdamWV2OpBuilder; FusedAdamWV2OpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import MatmulAddOpBuilder; MatmulAddOpBuilder().load()" &
python -c "import mindspeed; from mindspeed.op_builder import GroupMatmulAddOpBuilder; GroupMatmulAddOpBuilder().load()" &
wait $(jobs -rp)


if [ "$MASTER_ADDR" = "$CURRENT_IP" ]; then
  # the master node starts
  ray start --head --port $MASTER_PORT --dashboard-host=0.0.0.0 --node-ip-address=$CURRENT_IP --dashboard-port=8260 --resources='{"NPU": '$NPUS_PER_NODE'}'

  while true; do
      ray_status_output=$(ray status)
      npu_count=$(echo "$ray_status_output" | grep -oP '(?<=/)\d+\.\d+(?=\s*NPU)' | head -n 1)
      npu_count_int=$(echo "$npu_count" | awk '{print int($1)}')
      device_count=$((npu_count_int / $NPUS_PER_NODE))

      # determine whether device_count is equal to NNODES
      if [ "$device_count" -eq "$NNODES" ]; then
          echo "Ray cluster is ready with $device_count devices (from $npu_count NPU resources), starting Python script."
          ray status
          bash $TRAIN_SCRIPT
          break
      else
          echo "Waiting for Ray to allocate $NNODES devices. Current device count: $device_count"
          sleep 5
      fi
  done
else
  # the child node attempts to register ray with the master node until successful
  while true; do
      # try to connect to the Ray cluster
      ray start --address="$MASTER_ADDR:$MASTER_PORT" --resources='{"NPU": '$NPUS_PER_NODE'}' --node-ip-address=$CURRENT_IP

      # check if the connection is successful
      ray status
      if [ $? -eq 0 ]; then
          echo "Successfully connected to the Ray cluster!"
          break
      else
          echo "Failed to connect to the Ray cluster. Retrying in 5 seconds..."
          sleep 5
      fi
  done
fi

sleep 999999
