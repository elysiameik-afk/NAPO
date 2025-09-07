# 结构化Mask功能使用指南

## 🎯 **功能概述**

结构化Mask功能解决了格式奖励和内容奖励混淆的问题，通过智能的梯度调整来解耦格式学习和内容学习。

### **核心思想**：
- **优势 > 0**：格式token梯度增强（鼓励正确格式）
- **优势 < 0**：格式token被mask掉（避免惩罚格式）
- **结果**：格式学习和内容学习互不干扰

## 🔧 **配置方法**

### **启用结构化Mask**

在训练脚本中添加以下参数：

```bash
# 启用结构化mask
actor_rollout_ref.actor.policy_loss.use_structure_mask=true

# 设置增强倍数（可选，默认1.0表示不增强）
actor_rollout_ref.actor.policy_loss.structure_boost_factor=2.0

# 自定义格式token（可选，已有默认值）
actor_rollout_ref.actor.policy_loss.structure_tokens.think_start=[151667]
actor_rollout_ref.actor.policy_loss.structure_tokens.think_end=[151668]
actor_rollout_ref.actor.policy_loss.structure_tokens.answer_start=[27,9217,29]
actor_rollout_ref.actor.policy_loss.structure_tokens.answer_end=[522,9217,29]
```

### **完整示例**

```bash
python -m verl.trainer.main_ppo \
    data=gsm8k_data \
    data.train_files=[train.jsonl] \
    data.val_files=[val.jsonl] \
    actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
    actor_rollout_ref.actor.policy_loss.use_structure_mask=true \
    actor_rollout_ref.actor.policy_loss.structure_boost_factor=2.0
```

## ⚙️ **参数说明**

### **use_structure_mask** (bool)
- **默认值**: `false`
- **作用**: 是否启用结构化mask功能
- **建议**: 如果训练中有格式token，建议设为 `true`

### **structure_boost_factor** (float)
- **默认值**: `1.0`
- **作用**: 正优势时格式token的增强倍数
- **建议值**:
  - `1.0`: 不增强（只mask负优势）
  - `2.0`: 适度增强
  - `3.0`: 强力增强

### **structure_tokens** (dict)
- **默认值**: 已配置常用格式token
- **作用**: 定义哪些token被视为格式token
- **格式**: `token_name: [token_id1, token_id2, ...]`

## 🎛️ **不同算法的兼容性**

结构化Mask功能对所有算法都生效：

### **GSPO + 结构化Mask**
```bash
actor_rollout_ref.actor.policy_loss.loss_mode=gspo \
actor_rollout_ref.actor.policy_loss.use_structure_mask=true
```

### **A-RSIC + 结构化Mask**
```bash
actor_rollout_ref.actor.policy_loss.loss_mode=arsic \
actor_rollout_ref.actor.policy_loss.use_structure_mask=true
```

### **Vanilla PPO + 结构化Mask**
```bash
actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
actor_rollout_ref.actor.policy_loss.use_structure_mask=true
```

## 📊 **预期效果**

### **启用前的问题**：
- 答案错误时，格式正确也被惩罚
- 模型对格式token产生混乱的学习信号
- 格式一致性差，训练不稳定

### **启用后的改进**：
- ✅ **格式一致性提升**：模型更稳定地生成正确格式
- ✅ **学习效率提升**：避免格式和内容的负面干扰
- ✅ **训练稳定性**：减少混乱的梯度信号

## 🔍 **监控指标**

启用结构化Mask后，重点观察：

1. **格式一致性**：生成的文本是否更稳定地包含正确格式
2. **训练稳定性**：损失曲线是否更平滑
3. **性能指标**：整体任务性能是否提升
4. **梯度范数**：是否在合理范围内

## ⚠️ **注意事项**

### **计算开销**
- **额外开销**: < 1%，几乎可以忽略
- **内存影响**: 微乎其微
- **训练速度**: 基本无影响

### **参数调优建议**
1. **首次使用**: 先用默认参数 `structure_boost_factor=1.0`
2. **如需增强**: 逐步提高到 `2.0` 或 `3.0`
3. **过度增强**: 如果格式过于强化而忽略内容，降低倍数

### **适用场景**
- ✅ **有明确格式要求的任务**（如思维链、结构化输出）
- ✅ **格式和内容奖励混合的训练**
- ❌ **纯内容生成任务**（无格式要求时不需要启用）

## 🚀 **快速开始**

### **最简配置**（推荐新手）：
```bash
actor_rollout_ref.actor.policy_loss.use_structure_mask=true
```

### **进阶配置**（需要强化格式）：
```bash
actor_rollout_ref.actor.policy_loss.use_structure_mask=true \
actor_rollout_ref.actor.policy_loss.structure_boost_factor=2.0
```

### **自定义格式token**：
```bash
actor_rollout_ref.actor.policy_loss.use_structure_mask=true \
actor_rollout_ref.actor.policy_loss.structure_tokens.custom_start=[12345] \
actor_rollout_ref.actor.policy_loss.structure_tokens.custom_end=[67890]
```

## 🔧 **故障排除**

### **常见问题**

**Q: 启用后性能下降？**
A: 尝试降低 `structure_boost_factor` 或检查格式token定义是否正确

**Q: 格式还是不稳定？**
A: 提高 `structure_boost_factor` 到 2.0-3.0

**Q: 如何验证功能是否生效？**
A: 检查训练日志，观察格式一致性的变化

### **调试技巧**
1. 先在小数据集上测试
2. 对比启用前后的格式一致性
3. 监控梯度范数是否异常

## 📝 **总结**

结构化Mask是一个轻量级但强大的功能，能够有效解决格式学习和内容学习的冲突问题。通过简单的配置就能显著提升模型在结构化任务上的表现。
