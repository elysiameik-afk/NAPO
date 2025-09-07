# A-RSIC 算法使用指南

## 概述

A-RSIC (Adaptive Risk-Sensitive Importance Correction) 算法已成功集成到 GSPO 框架中。你现在可以通过两种方式在 GSPO 和 A-RSIC 之间切换进行对比实验。

## 切换方式

### 方式一：配置文件切换（推荐）

通过修改启动脚本中的 `loss_mode` 参数：

```bash
# 使用原始 GSPO
actor_rollout_ref.actor.policy_loss.loss_mode=gspo

# 使用 A-RSIC 创新算法
actor_rollout_ref.actor.policy_loss.loss_mode=arsic
```

**示例**：修改 `exp/gspo.sh` 文件：
```bash
# 原始配置
# actor_rollout_ref.actor.policy_loss.loss_mode=gspo

# A-RSIC 配置
actor_rollout_ref.actor.policy_loss.loss_mode=arsic
```

### 方式二：手动代码切换

在 `verl/trainer/ppo/core_algos.py` 文件的 `compute_policy_loss_gspo` 函数中：

**位置**：约第 1060-1070 行

**切换到 GSPO（原始算法）**：
```python
# ==================== 原始 GSPO 实现 (可手动切换) ====================
seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
# =====================================================================

# ==================== A-RSIC 创新实现 (可手动切换) ====================
# negative_approx_kl_seq = compute_arsic_sequence_weights(
#     log_prob_current=log_prob,
#     log_prob_old=old_log_prob,
#     response_mask=response_mask
# )
# ====================================================================
```

**切换到 A-RSIC（创新算法）**：
```python
# ==================== 原始 GSPO 实现 (可手动切换) ====================
# seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
# negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths
# =====================================================================

# ==================== A-RSIC 创新实现 (可手动切换) ====================
negative_approx_kl_seq = compute_arsic_sequence_weights(
    log_prob_current=log_prob,
    log_prob_old=old_log_prob,
    response_mask=response_mask
)
# ====================================================================
```

## A-RSIC 算法参数

A-RSIC 算法包含以下可调参数（在 `compute_arsic_sequence_weights` 函数中）：

- **C** (默认: 0.02): 全局风险敏感度缩放因子（已优化为接近 GSPO 性能）
- **gamma_max** (默认: 5.0): 最大风险规避系数，用于数值稳定性
- **epsilon** (默认: 1e-8): 防止除零的极小值
- **degradation_threshold** (默认: 0.1): 风险阈值，低于此值时退化为算术平均

### 参数优化说明

**最新优化**：为了让 A-RSIC 性能接近 GSPO，我们调整了关键参数：
- `C` 从 1.0 降低到 0.02（降低风险敏感度）
- 新增 `degradation_threshold = 0.1`（大多数序列退化为算术平均）

**预期效果**：
- 90%+ 的序列：gamma < 0.1，使用算术平均（接近 GSPO 性能）
- 少数高风险序列：使用温和的 A-RSIC 逻辑

如需进一步调整参数，可修改函数调用：
```python
negative_approx_kl_seq = compute_arsic_sequence_weights(
    log_prob_current=log_prob,
    log_prob_old=old_log_prob,
    response_mask=response_mask,
    C=0.02,                      # 风险敏感度（可调整）
    gamma_max=5.0,               # 最大风险规避
    degradation_threshold=0.1    # 退化阈值（可调整）
)
```

## 对比实验建议

1. **基线实验**：先运行原始 GSPO 获得基线结果
2. **A-RSIC 实验**：切换到 A-RSIC 进行对比
3. **关键指标**：
   - 训练稳定性（损失曲线平滑度）
   - 收敛速度
   - 最终模型性能
   - 内存使用情况

## 预期效果

A-RSIC 算法预期在以下场景中表现更好：
- 高方差的训练环境
- 长序列任务
- 大模型训练中的稳定性问题

## 故障排除

如果遇到问题：
1. 检查配置文件语法是否正确
2. 确认 `loss_mode` 参数拼写无误
3. 查看训练日志中的错误信息
4. 如有数值不稳定，可尝试降低 `gamma_max` 参数

### 常见错误修复

**错误**: `RuntimeError: a Tensor with 4 elements cannot be converted to Scalar`
**原因**: 返回值格式不匹配
**解决**: 已修复，A-RSIC 现在返回与 GSPO 完全一致的四个标量值：
- `pg_loss`: 策略梯度损失
- `pg_clipfrac`: 裁剪比例
- `ppo_kl`: KL 散度
- `pg_clipfrac_lower`: 下界裁剪比例（兼容性，固定为0）

## 技术细节

A-RSIC 的核心创新：
1. **风险量化**：使用变异系数 (CV) 评估序列内权重波动
2. **自适应调整**：根据风险动态调整风险规避系数 γ
3. **确定性等价物**：通过数值稳定的 Log-Sum-Exp 计算最终权重

这种方法能够自动识别高风险序列并采用更保守的权重计算，从而提升训练稳定性。
