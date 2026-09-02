# Variant: regroup + rollout eager workaround for MoeDistributeDispatchV2 torchair compile issue
set -ex

export CUDA_DEVICE_MAX_CONNECTIONS=1
export PYTORCH_NPU_ALLOC_CONF="expandable_segments:True"

export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/asdsip/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi=1

export HYDRA_FULL_ERROR=1
#export ASCEND_LAUNCH_BLOCKING=1
export RAY_DEDUP_LOGS=0

export ASCEND_GLOBAL_EVENT_ENABLE=0
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export HCCL_IF_BASE_PORT=64021
export CUDA_DEVICE_MAX_CONNECTIONS=1

export MASTER_PORT=23300    # vllm port error
export D2D_DATA_TRANSFER=1

export VLLM_USE_V1=1
export PRINT_MEMORY=1
export USE_ALLTOALL_OVERLAP=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_LOGGING_LEVEL=INFO
export VLLM_ENABLE_MC2=1                     # 910C开启
export VLLM_DP_SIZE=16                        # world_size // rollout.tp_size
export HCCL_BUFFSIZE=800

export TASK_QUEUE_ENABLE=2

export VLLM_ENABLE_FIX_ROUTE=0
export VLLM_MODEL_EXECUTE_TIME_OBSERVE=0     # decode prefill的耗时打印

#extra env in qwen3_235b_env.sh
# Recipe features
export VLLM_ENABLE_GRAPH_MODE=0             # 0: eager mode, 1: graph mode
export VLLM_ENABLE_EXPERT_PARALLEL=1        # Enable EP in vLLM rollout.
export VLLM_CHUNK_MOE_SIZE=512              # The minimum block size set for prefill computation partition.
export ALL_TO_ALL_RESHARD=1                 # Enable EP to reshard parameters with AllToAllV (without communication redundancy).
export USE_ALLTOALL_OVERLAP=1               # Enable to overlap communication in EP with computation to hide MoE communication latency. Should be consistent with model conversion config.
export VLLM_ENABLE_EPLB=0                   # 0: disable eplb, 1: enable eplb
export USE_HDP=0                            # 0: disable hdp, 1: enable hdp
export ROLLOUT_REBALANCE_ENABLE=0          # 0: disable rollout rebalance, 1: enable rollout rebalance
#关闭看门狗,并控制超时时间
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_EXEC_TIMEOUT=7200
export HCCL_CONNECT_TIMEOUT=7200

#控制llm.py里的profiler
export VLLM_ASCEND_LLM_PROFILE_ENABLE=0

# Unified output root for checkpoints / rollout dumps / draft dumps / logs.
OUTPUT_ROOT=${OUTPUT_ROOT:-/workspace/cann-recipes-train/llm_rl/qwen3/outputs}
OUTPUT_SUBDIR=${OUTPUT_SUBDIR:-resample_result_16k_bs32_n16_baseline_ft}
OUTPUT_DIR="${OUTPUT_ROOT}/${OUTPUT_SUBDIR}"
ROLL_OUT_DIR="${OUTPUT_DIR}/rollout_data"
ROLL_LEN_DIR="${OUTPUT_DIR}/rollout_length"
DRAFT_DUMP_DIR="${OUTPUT_DIR}/draft_hidden"
CKPT_DIR="${OUTPUT_DIR}/checkpoints/qwen3moe_for_eagle3"
TB_DIR="${OUTPUT_DIR}/tensorboard"
LOG_DIR="${OUTPUT_DIR}/logs"

# Toggle switches:
#   SAVE_CKPT_ENABLE=1        -> save checkpoints
#   SAVE_DRAFT_HIDDEN_ENABLE=1 -> dump draft hidden states
SAVE_CKPT_ENABLE=${SAVE_CKPT_ENABLE:-0}
SAVE_DRAFT_HIDDEN_ENABLE=${SAVE_DRAFT_HIDDEN_ENABLE:-0}

mkdir -p "${ROLL_OUT_DIR}" "${ROLL_LEN_DIR}" "${TB_DIR}" "${LOG_DIR}"
if [ "${SAVE_DRAFT_HIDDEN_ENABLE}" = "1" ]; then
    mkdir -p "${DRAFT_DUMP_DIR}"
fi
if [ "${SAVE_CKPT_ENABLE}" = "1" ]; then
    mkdir -p "${CKPT_DIR}"
fi

# Draft data collection for offline Eagle3 training
export VLLM_ASCEND_ENABLE_DRAFT_TRAIN=${VLLM_ASCEND_ENABLE_DRAFT_TRAIN:-0}
export VLLM_ASCEND_DRAFT_DUMP_ENABLE=${VLLM_ASCEND_DRAFT_DUMP_ENABLE:-${SAVE_DRAFT_HIDDEN_ENABLE}}
export VLLM_ASCEND_DRAFT_DUMP_DIR=${VLLM_ASCEND_DRAFT_DUMP_DIR:-${DRAFT_DUMP_DIR}}
export VLLM_ASCEND_DRAFT_DUMP_EVERY=${VLLM_ASCEND_DRAFT_DUMP_EVERY:-1}
export VLLM_ASCEND_DRAFT_DUMP_HIDDEN_DTYPE=${VLLM_ASCEND_DRAFT_DUMP_HIDDEN_DTYPE:-bf16}
# Larger queue reduces drop risk when collecting offline draft dataset.
export VLLM_ASCEND_DRAFT_QUEUE_SIZE=${VLLM_ASCEND_DRAFT_QUEUE_SIZE:-4096}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-${TB_DIR}}

# Per-epoch auto regrouping by previous rollout response lengths.
# Keep dataloader workers at 0 when using curriculum sampler.
export VLLM_EPOCH_LENGTH_REGROUP_ENABLE=${VLLM_EPOCH_LENGTH_REGROUP_ENABLE:-1}
export VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE=${VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE:-1024}
export VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY=${VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY:-0.7}
export VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS=${VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS:-1}
# Rollout long-tail guard (default on):
# Cap per-step generation by expected response length from resampler.
export VLLM_ROLLOUT_EARLY_STOP_ENABLE=${VLLM_ROLLOUT_EARLY_STOP_ENABLE:-1}
export VLLM_ROLLOUT_EARLY_STOP_FACTOR=${VLLM_ROLLOUT_EARLY_STOP_FACTOR:-2.0}
export VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS=${VLLM_ROLLOUT_EARLY_STOP_MIN_TOKENS:-10000}
# Baseline switch:
#   1 -> eager on + no resample (disable length regroup sampler)
#   0 -> follow regroup switch below
export VLLM_EAGER_BASELINE_NO_RESAMPLE=${VLLM_EAGER_BASELINE_NO_RESAMPLE:-1}

if [ "${VLLM_EAGER_BASELINE_NO_RESAMPLE}" = "1" ]; then
    SAMPLER_CLASS_PATH=null
    SAMPLER_CLASS_NAME=null
    SAMPLER_EXTRA_ARGS=""
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
elif [ "${VLLM_EPOCH_LENGTH_REGROUP_ENABLE}" = "1" ]; then
    SAMPLER_CLASS_PATH=pkg://verl.experimental.dataset.length_bucket_sampler
    SAMPLER_CLASS_NAME=LengthAwareEpochSampler
    SAMPLER_EXTRA_ARGS="+data.sampler.bucket_size=${VLLM_EPOCH_LENGTH_REGROUP_BUCKET_SIZE} +data.sampler.ema_decay=${VLLM_EPOCH_LENGTH_REGROUP_EMA_DECAY} +data.sampler.shuffle_batch_blocks=${VLLM_EPOCH_LENGTH_REGROUP_SHUFFLE_BLOCKS}"
    # Work around torchair graph compile instability for MoE dispatch under
    # dynamically regrouped rollout batches.
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
else
    SAMPLER_CLASS_PATH=null
    SAMPLER_CLASS_NAME=null
    SAMPLER_EXTRA_ARGS=""
    ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-False}
fi

# Keep disk usage bounded for per-step save.
# 1) only save model weights in checkpoints (no optimizer/extra),
# 2) keep only the latest checkpoint(s).
ACTOR_CKPT_SAVE_CONTENTS=${ACTOR_CKPT_SAVE_CONTENTS:-[model]}
ACTOR_CKPT_LOAD_CONTENTS=${ACTOR_CKPT_LOAD_CONTENTS:-[model]}
CRITIC_CKPT_SAVE_CONTENTS=${CRITIC_CKPT_SAVE_CONTENTS:-[model]}
CRITIC_CKPT_LOAD_CONTENTS=${CRITIC_CKPT_LOAD_CONTENTS:-[model]}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
MAX_CRITIC_CKPT_TO_KEEP=${MAX_CRITIC_CKPT_TO_KEEP:-1}

if [ "${SAVE_CKPT_ENABLE}" = "1" ]; then
    TRAINER_SAVE_FREQ=1
    TRAINER_DEFAULT_LOCAL_DIR="${CKPT_DIR}"
else
    TRAINER_SAVE_FREQ=-1
    TRAINER_DEFAULT_LOCAL_DIR="${OUTPUT_DIR}"
fi

HOME=$(pwd)
MODEL_PATH=${MODEL_PATH:-"/home/data/Qwen3-30B-A3B"}
CONFIG_DIR=${CONFIG_DIR:-"${HOME}/verl/trainer/config"}
DISTCP_PATH="/home/data/Qwen3-30B-A3B_megatron"
TRAIN_FILE=${TRAIN_FILE:-"/data/deepscaler/train.parquet"}
TEST_FILE=${TEST_FILE:-"/data/deepscaler/test.parquet"}


time=$(date +%Y%m%d%H%M%S)
logfile="${LOG_DIR}/wjqwen30b-a3b-record_graph_save4eagle3_${time}.txt"

set -x

python3 -m verl.trainer.main_ppo --config-path="${CONFIG_DIR}" \
    --config-name='ppo_megatron_trainer.yaml'\
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=32 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    data.dataloader_num_workers=0 \
    data.sampler.class_path=${SAMPLER_CLASS_PATH} \
    data.sampler.class_name=${SAMPLER_CLASS_NAME} \
    ${SAMPLER_EXTRA_ARGS} \
    +data.dataset_fraction=0.005\
    custom_reward_function.path=deepscaler.py \
    custom_reward_function.name=compute_score  \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.clip_grad=10000 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=4 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=False \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.actor.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.load_weight=True \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=block \
    actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.max_num_batched_tokens=1024 \
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER} \
    actor_rollout_ref.rollout.max_num_seqs=32 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=0.9 \
    actor_rollout_ref.rollout.top_k=50 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.megatron.param_offload=True \
    actor_rollout_ref.ref.load_weight=True \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=True \
    actor_rollout_ref.ref.megatron.dist_checkpointing_path="${DISTCP_PATH}" \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.balance_batch=False \
    trainer.device=npu \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name='verl_grpo_example' \
    trainer.experiment_name='qwen3_30_verl_mindspeedllm_vllm' \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.save_freq=${TRAINER_SAVE_FREQ} \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT_TO_KEEP} \
    trainer.max_critic_ckpt_to_keep=${MAX_CRITIC_CKPT_TO_KEEP} \
    actor_rollout_ref.actor.checkpoint.save_contents="${ACTOR_CKPT_SAVE_CONTENTS}" \
    actor_rollout_ref.actor.checkpoint.load_contents="${ACTOR_CKPT_LOAD_CONTENTS}" \
    critic.checkpoint.save_contents="${CRITIC_CKPT_SAVE_CONTENTS}" \
    critic.checkpoint.load_contents="${CRITIC_CKPT_LOAD_CONTENTS}" \
    +trainer.save_epoch_freq=0 \
    trainer.default_local_dir="${TRAINER_DEFAULT_LOCAL_DIR}" \
    trainer.test_freq=-1 \
    trainer.total_epochs=3 \
    +trainer.rollout_data_dir="${ROLL_OUT_DIR}" \
    +trainer.rollout_length_dir="${ROLL_LEN_DIR}" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_flash_attn=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.pipeline_num_transformer_layers=[[11],[13],[13],[11]] \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type='alltoall' \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_alltoall_overlap_comm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_rotary_pos_emb=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_fused_swiglu=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.seq_length=2048 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_first_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.num_layers_in_last_pipeline_stage=11 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.swap_optimizer=True  $@ >> "${logfile}"
