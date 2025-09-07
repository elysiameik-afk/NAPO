#!/bin/bash
set -x

# 激活环境
# eval "$(conda shell.bash hook)"
# conda deactivate
# conda activate verl

# 设置环境变量
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export HYDRA_FULL_ERROR=1

# ===================================================================
# FIX: 定义一个Shell变量来存储响应长度，避免'bad substitution'错误
# ===================================================================
MAX_RESPONSE_LENGTH=2048

# ===================================================================
# 结构化Mask配置说明:
# use_structure_mask: 是否启用结构化mask (true/false)
# structure_boost_factor: 正优势时格式token的增强倍数 (1.0=不增强, 2.0=双倍增强)
# 
# 启用结构化mask可以解决格式奖励和内容奖励混淆的问题:
# - 优势>0时: 格式token梯度增强 (鼓励正确格式)
# - 优势<0时: 格式token被mask掉 (避免惩罚格式)
# 
# 本脚本启用了结构化mask功能，适用于有格式要求的训练任务
# ===================================================================

# FIX: 重新使用反斜杠 `\` 来分割长命令，确保脚本可读性和正确性
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.clip_ratio_low=0.0003 \
    actor_rollout_ref.actor.clip_ratio_high=0.0004 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.policy_loss.use_structure_mask=true \
    actor_rollout_ref.actor.policy_loss.structure_boost_factor=2.0 \
    data.train_files=/root/autodl-tmp/myverl/data/kk/4ppl_few/train.parquet \
    data.val_files=/root/autodl-tmp/myverl/data/kk/4ppl_few/test.parquet \
    data.train_batch_size=16 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.shuffle=true \
    data.prompt_key=prompt \
    data.truncation='error' \
    data.filter_overlong_prompts=true \
    actor_rollout_ref.model.path=/root/autodl-tmp/verldev/Verl_RL/mymodels/qwen3-0.6b \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.entropy_checkpointing=true \
    actor_rollout_ref.actor.loss_agg_mode="seq-mean-token-mean" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=true \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    reward_model.reward_manager=logic_rl \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=false \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=2000 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    +reward_model.reward_kwargs.max_resp_len=${MAX_RESPONSE_LENGTH} \
    trainer.project_name=Qwen2.5-0.5-TokenEMA \
    trainer.experiment_name=GSPO_StructureMask \
    trainer.default_local_dir=/root/autodl-tmp/verldev/Verl_RL/ckpts/Qwen2.5-0.5/GSPO_StructureMask \
    trainer.critic_warmup=0 \
    trainer.save_freq=4 \
    trainer.test_freq=1 \
    trainer.total_epochs=4 \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.logger='["console","wandb"]' \
    trainer.val_before_train=false \
    trainer.resume_mode=auto \
    trainer.log_val_generations=2 \
    $@
