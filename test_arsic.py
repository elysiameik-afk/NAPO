#!/usr/bin/env python3
"""
A-RSIC 算法数值稳定性测试脚本

用于验证 A-RSIC 实现的正确性和数值稳定性。
"""

import torch
import numpy as np

def compute_arsic_sequence_weights(
    log_prob_current: torch.Tensor,
    log_prob_old: torch.Tensor,
    response_mask: torch.Tensor,
    C: float = 0.02,
    gamma_max: float = 5.0,
    epsilon: float = 1e-8,
    degradation_threshold: float = 0.1
) -> torch.Tensor:
    """A-RSIC 算法实现（复制自 core_algos.py）"""
    
    # Step 1: Compute token-level importance weights
    log_token_weights = log_prob_current - log_prob_old
    token_weights = torch.exp(log_token_weights) * response_mask
    
    # Step 2: Compute sequence statistics
    seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
    
    masked_weights_sum = (token_weights * response_mask).sum(dim=-1)
    seq_mean = masked_weights_sum / seq_lengths
    
    seq_var = (token_weights.pow(2) * response_mask).sum(dim=-1) / seq_lengths - seq_mean.pow(2)
    seq_var = torch.clamp(seq_var, min=0.0)
    seq_std = torch.sqrt(seq_var + epsilon)
    
    # Step 3: Compute adaptive risk aversion coefficient
    cv = seq_std / (seq_mean + epsilon)
    gamma = torch.clamp(C * cv, min=0.0, max=gamma_max)
    
    # Step 4: Compute A-RSIC weights using numerically stable LSE
    is_low_risk = (gamma < degradation_threshold)

    gamma_w = gamma.unsqueeze(-1) * token_weights

    gamma_w_masked = torch.where(
        response_mask.bool(),
        gamma_w,
        torch.tensor(-torch.inf, device=gamma_w.device, dtype=gamma_w.dtype)
    )
    max_gamma_w, _ = torch.max(gamma_w_masked, dim=-1, keepdim=True)

    exp_term = torch.exp(gamma_w - max_gamma_w) * response_mask
    mean_exp_term = exp_term.sum(dim=-1) / seq_lengths

    log_mean_exp_term = torch.log(mean_exp_term + epsilon)
    lse_result = (max_gamma_w.squeeze(-1) + log_mean_exp_term) / (gamma + epsilon)

    # Step 5: Choose between LSE result and stable mean based on risk level

    # ==================== 版本1：截断几何平均（更稳定） ====================
    # 去掉最极端的10%值，然后取几何平均
    def trimmed_geometric_mean(weights, mask, trim_ratio=0.1):
        # 只对有效token计算
        valid_weights = weights * mask
        batch_size, seq_len = weights.shape

        trimmed_results = []
        for i in range(batch_size):
            seq_weights = valid_weights[i][mask[i].bool()]
            if len(seq_weights) <= 2:  # 序列太短，直接几何平均
                log_mean = torch.log(seq_weights + epsilon).mean()
            else:
                # 排序并去掉极值
                sorted_weights, _ = torch.sort(seq_weights)
                trim_count = max(1, int(len(seq_weights) * trim_ratio))
                trimmed_weights = sorted_weights[trim_count:-trim_count] if trim_count < len(seq_weights)//2 else sorted_weights
                log_mean = torch.log(trimmed_weights + epsilon).mean()
            trimmed_results.append(log_mean)

        return torch.stack(trimmed_results)

    trimmed_mean_log = trimmed_geometric_mean(token_weights, response_mask)
    arsic_log_weights = torch.where(is_low_risk, trimmed_mean_log, lse_result)
    # ====================================================================

    # ==================== 版本2：调和平均（最稳定） ====================
    # # 调和平均：n / Σ(1/w_i)，对异常值最不敏感
    # def harmonic_mean_log(weights, mask):
    #     # 计算调和平均的对数
    #     valid_weights = weights * mask + (1 - mask) * 1.0  # padding位置设为1避免除零
    #     reciprocal_sum = (1.0 / (valid_weights + epsilon) * mask).sum(dim=-1)
    #     seq_lengths = mask.sum(dim=-1).clamp(min=1)
    #     harmonic_mean = seq_lengths / reciprocal_sum
    #     return torch.log(harmonic_mean + epsilon)
    #
    # harmonic_mean_log_result = harmonic_mean_log(token_weights, response_mask)
    # arsic_log_weights = torch.where(is_low_risk, harmonic_mean_log_result, lse_result)
    # ====================================================================
    
    return arsic_log_weights


def test_arsic_numerical_stability():
    """测试 A-RSIC 的数值稳定性"""
    print("=== A-RSIC 数值稳定性测试 ===")
    
    # 测试用例 1: 正常情况
    print("\n1. 正常情况测试")
    batch_size, seq_len = 4, 8
    log_prob_current = torch.randn(batch_size, seq_len) * 0.1
    log_prob_old = torch.randn(batch_size, seq_len) * 0.1
    response_mask = torch.ones(batch_size, seq_len)
    
    result = compute_arsic_sequence_weights(log_prob_current, log_prob_old, response_mask)
    print(f"结果形状: {result.shape}")
    print(f"结果范围: [{result.min().item():.4f}, {result.max().item():.4f}]")
    print(f"是否包含 NaN: {torch.isnan(result).any().item()}")
    print(f"是否包含 Inf: {torch.isinf(result).any().item()}")
    
    # 测试用例 2: 高方差情况
    print("\n2. 高方差情况测试")
    log_prob_current = torch.randn(batch_size, seq_len) * 2.0  # 更大的方差
    log_prob_old = torch.randn(batch_size, seq_len) * 2.0
    
    result = compute_arsic_sequence_weights(log_prob_current, log_prob_old, response_mask)
    print(f"结果范围: [{result.min().item():.4f}, {result.max().item():.4f}]")
    print(f"是否包含 NaN: {torch.isnan(result).any().item()}")
    print(f"是否包含 Inf: {torch.isinf(result).any().item()}")
    
    # 测试用例 3: 极端情况
    print("\n3. 极端情况测试")
    log_prob_current = torch.tensor([[10.0, -10.0, 5.0, -5.0] * 2]).float()  # 极端值
    log_prob_old = torch.tensor([[-10.0, 10.0, -5.0, 5.0] * 2]).float()
    response_mask = torch.ones(1, 8)
    
    result = compute_arsic_sequence_weights(log_prob_current, log_prob_old, response_mask)
    print(f"结果: {result.item():.4f}")
    print(f"是否包含 NaN: {torch.isnan(result).any().item()}")
    print(f"是否包含 Inf: {torch.isinf(result).any().item()}")
    
    # 测试用例 4: 带 padding 的情况
    print("\n4. 带 padding 的情况测试")
    response_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 0, 0],
                                  [1, 1, 1, 1, 1, 1, 1, 1],
                                  [1, 1, 0, 0, 0, 0, 0, 0]]).float()
    
    result = compute_arsic_sequence_weights(log_prob_current[:4], log_prob_old[:4], response_mask)
    print(f"结果: {result}")
    print(f"是否包含 NaN: {torch.isnan(result).any().item()}")
    print(f"是否包含 Inf: {torch.isinf(result).any().item()}")


def compare_with_gspo():
    """对比 A-RSIC 和 GSPO 的结果"""
    print("\n=== A-RSIC vs GSPO 对比测试 ===")
    
    batch_size, seq_len = 3, 6
    log_prob_current = torch.randn(batch_size, seq_len) * 0.5
    log_prob_old = torch.randn(batch_size, seq_len) * 0.5
    response_mask = torch.ones(batch_size, seq_len)
    
    # GSPO 计算
    negative_approx_kl = log_prob_current - log_prob_old
    seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
    gspo_result = (negative_approx_kl * response_mask).sum(dim=-1) / seq_lengths
    
    # A-RSIC 计算
    arsic_result = compute_arsic_sequence_weights(log_prob_current, log_prob_old, response_mask)
    
    print("GSPO 结果:", gspo_result)
    print("A-RSIC 结果:", arsic_result)
    print("差异:", (arsic_result - gspo_result).abs())
    
    # 当权重方差很小时，A-RSIC 应该接近 GSPO
    print("\n低方差情况下的对比:")
    log_prob_current = torch.randn(batch_size, seq_len) * 0.01  # 很小的方差
    log_prob_old = torch.randn(batch_size, seq_len) * 0.01
    
    negative_approx_kl = log_prob_current - log_prob_old
    gspo_result = (negative_approx_kl * response_mask).sum(dim=-1) / seq_lengths
    arsic_result = compute_arsic_sequence_weights(log_prob_current, log_prob_old, response_mask)
    
    print("GSPO 结果:", gspo_result)
    print("A-RSIC 结果:", arsic_result)
    print("差异:", (arsic_result - gspo_result).abs())
    print("差异是否 < 0.01:", ((arsic_result - gspo_result).abs() < 0.01).all().item())


def test_degradation_behavior():
    """测试新参数下的退化行为"""
    print("\n=== 测试退化行为（新参数 C=0.02, threshold=0.1）===")

    batch_size, seq_len = 5, 6

    # 测试不同方差水平的序列
    test_cases = [
        ("低方差", torch.randn(batch_size, seq_len) * 0.1),
        ("中等方差", torch.randn(batch_size, seq_len) * 0.5),
        ("高方差", torch.randn(batch_size, seq_len) * 1.5),
    ]

    response_mask = torch.ones(batch_size, seq_len)

    for case_name, log_prob_current in test_cases:
        log_prob_old = torch.randn(batch_size, seq_len) * 0.1

        # 计算 A-RSIC
        arsic_result = compute_arsic_sequence_weights(log_prob_current, log_prob_old, response_mask)

        # 计算算术平均（用于对比）
        log_token_weights = log_prob_current - log_prob_old
        token_weights = torch.exp(log_token_weights) * response_mask
        seq_lengths = response_mask.sum(dim=-1).clamp(min=1)
        masked_weights_sum = (token_weights * response_mask).sum(dim=-1)
        seq_mean = masked_weights_sum / seq_lengths
        arithmetic_mean_log = torch.log(seq_mean + 1e-8)

        # 计算差异
        diff = (arsic_result - arithmetic_mean_log).abs()

        print(f"\n{case_name}:")
        print(f"  A-RSIC 结果: {arsic_result.mean().item():.6f}")
        print(f"  算术平均: {arithmetic_mean_log.mean().item():.6f}")
        print(f"  平均差异: {diff.mean().item():.6f}")
        print(f"  最大差异: {diff.max().item():.6f}")

        # 检查是否大多数序列退化为算术平均
        close_to_arithmetic = (diff < 0.01).float().mean()
        print(f"  退化比例: {close_to_arithmetic.item():.1%}")


if __name__ == "__main__":
    torch.manual_seed(42)  # 确保可重现性

    test_arsic_numerical_stability()
    compare_with_gspo()
    test_degradation_behavior()

    print("\n=== 测试完成 ===")
    print("如果所有测试都通过（无 NaN/Inf），则 A-RSIC 实现数值稳定。")
    print("新参数应该让大多数序列退化为算术平均，接近 GSPO 性能。")
