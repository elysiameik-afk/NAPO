#!/usr/bin/env python3
"""
验证 A-RSIC 修复的脚本

检查 A-RSIC 函数是否返回正确的标量值，避免之前的 Tensor 转换错误。
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath('.'))

try:
    from verl.trainer.ppo.core_algos import compute_policy_loss_arsic
    from verl.workers.config import ActorConfig
    print("✅ 成功导入 A-RSIC 函数")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def test_arsic_return_format():
    """测试 A-RSIC 返回值格式是否正确"""
    print("\n=== 测试 A-RSIC 返回值格式 ===")
    
    # 创建测试数据
    batch_size, seq_len = 4, 8
    old_log_prob = torch.randn(batch_size, seq_len) * 0.1
    log_prob = torch.randn(batch_size, seq_len) * 0.1
    advantages = torch.randn(batch_size, seq_len) * 0.5
    response_mask = torch.ones(batch_size, seq_len)
    
    # 创建配置
    config = ActorConfig()
    config.clip_ratio = 0.2
    config.clip_ratio_low = 0.0003
    config.clip_ratio_high = 0.0004
    
    try:
        # 调用 A-RSIC 函数
        result = compute_policy_loss_arsic(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=config
        )
        
        print(f"✅ A-RSIC 函数调用成功")
        print(f"返回值数量: {len(result)}")
        
        # 检查返回值
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = result
        
        print(f"pg_loss: {type(pg_loss)}, shape: {pg_loss.shape if hasattr(pg_loss, 'shape') else 'scalar'}")
        print(f"pg_clipfrac: {type(pg_clipfrac)}, shape: {pg_clipfrac.shape if hasattr(pg_clipfrac, 'shape') else 'scalar'}")
        print(f"ppo_kl: {type(ppo_kl)}, shape: {ppo_kl.shape if hasattr(ppo_kl, 'shape') else 'scalar'}")
        print(f"pg_clipfrac_lower: {type(pg_clipfrac_lower)}, shape: {pg_clipfrac_lower.shape if hasattr(pg_clipfrac_lower, 'shape') else 'scalar'}")
        
        # 测试是否可以转换为标量
        try:
            pg_loss_scalar = pg_loss.detach().item()
            pg_clipfrac_scalar = pg_clipfrac.detach().item()
            ppo_kl_scalar = ppo_kl.detach().item()
            pg_clipfrac_lower_scalar = pg_clipfrac_lower.detach().item()
            
            print(f"✅ 所有返回值都可以转换为标量:")
            print(f"  pg_loss: {pg_loss_scalar:.6f}")
            print(f"  pg_clipfrac: {pg_clipfrac_scalar:.6f}")
            print(f"  ppo_kl: {ppo_kl_scalar:.6f}")
            print(f"  pg_clipfrac_lower: {pg_clipfrac_lower_scalar:.6f}")
            
        except RuntimeError as e:
            print(f"❌ 标量转换失败: {e}")
            return False
            
    except Exception as e:
        print(f"❌ A-RSIC 函数调用失败: {e}")
        return False
    
    return True


def test_numerical_stability():
    """测试数值稳定性"""
    print("\n=== 测试数值稳定性 ===")
    
    # 极端情况测试
    batch_size, seq_len = 2, 4
    
    # 极大值
    old_log_prob = torch.tensor([[10.0, -10.0, 5.0, -5.0],
                                 [8.0, -8.0, 3.0, -3.0]])
    log_prob = torch.tensor([[-10.0, 10.0, -5.0, 5.0],
                             [-8.0, 8.0, -3.0, 3.0]])
    advantages = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    config = ActorConfig()
    config.clip_ratio = 0.2
    config.clip_ratio_low = 0.0003
    config.clip_ratio_high = 0.0004
    
    try:
        result = compute_policy_loss_arsic(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=advantages,
            response_mask=response_mask,
            config=config
        )
        
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = result
        
        # 检查是否有 NaN 或 Inf
        values = [pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower]
        names = ['pg_loss', 'pg_clipfrac', 'ppo_kl', 'pg_clipfrac_lower']
        
        all_good = True
        for name, value in zip(names, values):
            if torch.isnan(value).any():
                print(f"❌ {name} 包含 NaN")
                all_good = False
            elif torch.isinf(value).any():
                print(f"❌ {name} 包含 Inf")
                all_good = False
            else:
                print(f"✅ {name} 数值正常: {value.item():.6f}")
        
        return all_good
        
    except Exception as e:
        print(f"❌ 数值稳定性测试失败: {e}")
        return False


if __name__ == "__main__":
    print("开始验证 A-RSIC 修复...")
    
    success1 = test_arsic_return_format()
    success2 = test_numerical_stability()
    
    if success1 and success2:
        print("\n🎉 所有测试通过！A-RSIC 修复成功。")
        print("现在可以安全地使用 loss_mode=arsic 进行训练。")
    else:
        print("\n❌ 测试失败，需要进一步修复。")
        sys.exit(1)
