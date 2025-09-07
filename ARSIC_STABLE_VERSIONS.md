# A-RSIC 稳定版本切换指南

本文档说明如何在两个更稳定的 A-RSIC 版本之间切换。

## 🔧 **两个版本说明**

### **版本1：截断几何平均（推荐）**
- **稳定性**：比几何平均更稳定
- **原理**：去掉最极端的10%值，然后取几何平均
- **优势**：去除异常值影响，保持合理的学习速度
- **适用**：大多数情况下的首选

### **版本2：调和平均（最稳定）**
- **稳定性**：最稳定，对异常值最不敏感
- **原理**：调和平均 = n / Σ(1/w_i)
- **优势**：极度保守，几乎不受大值影响
- **适用**：训练极不稳定时的最后手段

## 🚀 **如何切换**

### **当前激活：版本1（截断几何平均）**

在 `verl/trainer/ppo/core_algos.py` 的 `compute_arsic_sequence_weights` 函数中：

```python
# ==================== 版本1：截断几何平均（更稳定） ====================
# 去掉最极端的10%值，然后取几何平均
def trimmed_geometric_mean(weights, mask, trim_ratio=0.1):
    # ... 实现代码 ...

trimmed_mean_log = trimmed_geometric_mean(token_weights, response_mask)
arsic_log_weights = torch.where(is_low_risk, trimmed_mean_log, lse_result)
# ====================================================================

# ==================== 版本2：调和平均（最稳定） ====================
# # 调和平均：n / Σ(1/w_i)，对异常值最不敏感
# def harmonic_mean_log(weights, mask):
#     # ... 实现代码 ...
# 
# harmonic_mean_log_result = harmonic_mean_log(token_weights, response_mask)
# arsic_log_weights = torch.where(is_low_risk, harmonic_mean_log_result, lse_result)
# ====================================================================
```

### **切换到版本2（调和平均）**

1. **注释掉版本1的代码**：
```python
# ==================== 版本1：截断几何平均（更稳定） ====================
# # 去掉最极端的10%值，然后取几何平均
# def trimmed_geometric_mean(weights, mask, trim_ratio=0.1):
#     # ... 实现代码 ...
# 
# trimmed_mean_log = trimmed_geometric_mean(token_weights, response_mask)
# arsic_log_weights = torch.where(is_low_risk, trimmed_mean_log, lse_result)
# ====================================================================
```

2. **取消注释版本2的代码**：
```python
# ==================== 版本2：调和平均（最稳定） ====================
# 调和平均：n / Σ(1/w_i)，对异常值最不敏感
def harmonic_mean_log(weights, mask):
    # 计算调和平均的对数
    valid_weights = weights * mask + (1 - mask) * 1.0  # padding位置设为1避免除零
    reciprocal_sum = (1.0 / (valid_weights + epsilon) * mask).sum(dim=-1)
    seq_lengths = mask.sum(dim=-1).clamp(min=1)
    harmonic_mean = seq_lengths / reciprocal_sum
    return torch.log(harmonic_mean + epsilon)

harmonic_mean_log_result = harmonic_mean_log(token_weights, response_mask)
arsic_log_weights = torch.where(is_low_risk, harmonic_mean_log_result, lse_result)
# ====================================================================
```

## 📊 **预期效果对比**

### **版本1（截断几何平均）**
- **Score**：应该接近或略优于 GSPO
- **稳定性**：比原始几何平均更稳定
- **探索性**：保持合理的探索能力
- **收敛**：正常收敛速度

### **版本2（调和平均）**
- **Score**：可能略低于 GSPO（更保守）
- **稳定性**：最高稳定性，几乎不会崩溃
- **探索性**：较低，更专注于已知好策略
- **收敛**：可能收敛较慢但更稳定

## 🎯 **使用建议**

### **推荐流程**：
1. **先试版本1**：截断几何平均，平衡性能和稳定性
2. **如果不够稳定**：切换到版本2（调和平均）
3. **如果过于保守**：回到版本1，或调整 `degradation_threshold`

### **参数调优**：
- **版本1**：可调整 `trim_ratio`（默认0.1，即去掉10%极值）
- **版本2**：主要通过 `degradation_threshold` 控制激活频率

## 🔍 **监控指标**

切换版本后，重点观察：
- **critic/score/mean**：性能指标
- **actor/entropy**：探索性（应该比 GSPO 稍低）
- **actor/pg_clipfrac**：裁剪比例（应该接近 GSPO）
- **训练稳定性**：损失曲线是否平滑

## ⚠️ **注意事项**

1. **同时修改测试文件**：`test_arsic.py` 中也有相同的代码需要同步修改
2. **重新训练**：切换版本后需要重新开始训练
3. **保存结果**：建议保存每个版本的训练日志用于对比
4. **渐进测试**：可以先在小数据集上快速验证效果

## 🚀 **快速验证**

切换版本后，可以运行测试脚本快速验证：
```bash
python test_arsic.py
```

观察输出中的退化比例和数值稳定性。
