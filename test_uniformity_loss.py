"""
测试mask_uniformity_loss是否能有效防止slot坍塌
"""
import torch
import sys
sys.path.append('/home/pbw/data1/3D_PointCloud_Segmentation/PLSG_Net/Model_Code/src/DINOSAUR')

from models.losses import DINOSAURLoss

def test_uniformity_loss():
    """测试均匀性损失"""
    print("=" * 60)
    print("测试 Mask Uniformity Loss")
    print("=" * 60)
    
    # 创建损失函数
    loss_weights = {
        'reconstruction': 1.0,
        'mask_entropy': 0.15,
        'slot_diversity': 0.08,
        'mask_sparsity': 0.05,
        'mask_uniformity': 0.5
    }
    criterion = DINOSAURLoss(loss_weights)
    
    B, S, N = 2, 16, 512  # batch=2, slots=16, points=512
    
    # 情况1：所有点都分配给第一个slot（最坏情况）
    print("\n【情况1】所有点塌缩到slot_0:")
    masks_collapsed = torch.zeros(B, S, N)
    masks_collapsed[:, 0, :] = 1.0  # 所有点都在slot_0
    
    slots = torch.randn(B, S, 256)  # 随机slot特征
    reconstruction = torch.randn(B, N, 768)
    target = torch.randn(B, N, 768)
    
    loss_uniformity_1 = criterion.mask_uniformity_loss(masks_collapsed)
    print(f"  Uniformity Loss: {loss_uniformity_1.item():.4f}")
    
    # 情况2：均匀分布（理想情况）
    print("\n【情况2】均匀分布（理想情况）:")
    masks_uniform = torch.ones(B, S, N) / S  # 每个slot分配 N/S 个点
    
    loss_uniformity_2 = criterion.mask_uniformity_loss(masks_uniform)
    print(f"  Uniformity Loss: {loss_uniformity_2.item():.4f}")
    
    # 情况3：部分slot被使用（中等情况）
    print("\n【情况3】部分slot被使用（4个slot各占25%）:")
    masks_partial = torch.zeros(B, S, N)
    masks_partial[:, 0:4, :] = 0.25  # 前4个slot各占25%
    
    loss_uniformity_3 = criterion.mask_uniformity_loss(masks_partial)
    print(f"  Uniformity Loss: {loss_uniformity_3.item():.4f}")
    
    # 对比
    print("\n" + "=" * 60)
    print("损失对比:")
    print(f"  塌缩情况: {loss_uniformity_1.item():.4f} (应该最大)")
    print(f"  均匀分布: {loss_uniformity_2.item():.4f} (应该最小)")
    print(f"  部分使用: {loss_uniformity_3.item():.4f} (应该中等)")
    print("=" * 60)
    
    # 验证：塌缩情况的损失应该远大于均匀分布
    assert loss_uniformity_1 > loss_uniformity_2, "❌ 塌缩损失应该大于均匀分布损失！"
    assert loss_uniformity_1 > loss_uniformity_3, "❌ 塌缩损失应该大于部分使用损失！"
    print("\n✅ 测试通过：损失函数能正确惩罚slot坍塌！")
    
    # 测试完整损失计算
    print("\n" + "=" * 60)
    print("测试完整损失计算:")
    print("=" * 60)
    
    total_loss_1, loss_dict_1 = criterion(reconstruction, target, slots, masks_collapsed)
    total_loss_2, loss_dict_2 = criterion(reconstruction, target, slots, masks_uniform)
    
    print(f"\n塌缩情况总损失: {total_loss_1.item():.4f}")
    print(f"  各项损失: {loss_dict_1}")
    
    print(f"\n均匀分布总损失: {total_loss_2.item():.4f}")
    print(f"  各项损失: {loss_dict_2}")
    
    print(f"\n损失差异: {total_loss_1.item() - total_loss_2.item():.4f}")
    print("✅ 完整损失计算正常！")

if __name__ == "__main__":
    test_uniformity_loss()

