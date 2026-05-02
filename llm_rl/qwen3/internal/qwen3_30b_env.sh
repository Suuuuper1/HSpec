export USE_HDP=0                            # 0: disable hdp, 1: enable hdp
export ROLLOUT_REBALANCE_ENABLE=0           # 0: disable rollout rebalance, 1: enable rollout rebalance
export VLLM_SPECULATIVE_BATCH_SIZE_THRE=32  # [TODO] configure SAM batch size threshold based on actual training configration. 