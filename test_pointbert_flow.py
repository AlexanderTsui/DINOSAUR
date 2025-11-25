"""
测试 PointBERT + DINOSAUR 完整训练流程

验证内容:
1. 数据加载
2. 模型创建
3. 前向传播
4. 损失计算
5. 反向传播
6. 维度检查
"""

import os
import sys
import yaml
import torch
import torch.nn as nn

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '../PointBERT'))

from data.s3dis_dataset_pointbert import S3DISPointBERTDataset, collate_fn_pointbert
from models.pointbert_wrapper import create_pointbert_dinosaur_model
from models.losses import DINOSAURLoss
from torch.utils.data import DataLoader


def test_data_loading(config):
    """测试数据加载"""
    print("\n" + "=" * 60)
    print("[1/5] 测试数据加载")
    print("=" * 60)
    
    # 创建数据集
    train_dataset = S3DISPointBERTDataset(
        root_dir=config['data']['s3dis_root'],
        areas=config['data']['train_areas'],
        target_points=config['data']['target_points'],
        augment=True,
        aug_config=config['augmentation']
    )
    
    val_dataset = S3DISPointBERTDataset(
        root_dir=config['data']['s3dis_root'],
        areas=config['data']['val_areas'],
        target_points=config['data']['target_points'],
        augment=False
    )
    
    print(f"✓ 训练集: {len(train_dataset)} 个房间")
    print(f"✓ 验证集: {len(val_dataset)} 个房间")
    
    # 测试单个样本
    sample = train_dataset[0]
    print(f"✓ 样本形状: {sample['xyzrgb'].shape}")
    print(f"✓ XYZ范围: [{sample['xyzrgb'][:, :3].min():.3f}, {sample['xyzrgb'][:, :3].max():.3f}]")
    print(f"✓ RGB范围: [{sample['xyzrgb'][:, 3:].min():.3f}, {sample['xyzrgb'][:, 3:].max():.3f}]")
    
    # 测试DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn_pointbert
    )
    
    batch = next(iter(train_loader))
    print(f"✓ Batch形状: {batch['xyzrgb'].shape}")
    
    return train_loader, val_dataset


def test_model_creation(config, device):
    """测试模型创建"""
    print("\n" + "=" * 60)
    print("[2/5] 测试模型创建")
    print("=" * 60)
    
    model = create_pointbert_dinosaur_model(config, device=device)
    model.eval()
    
    print("✓ 模型创建成功")
    
    return model


def test_forward_pass(model, train_loader, device):
    """测试前向传播"""
    print("\n" + "=" * 60)
    print("[3/5] 测试前向传播")
    print("=" * 60)
    
    batch = next(iter(train_loader))
    xyzrgb = batch['xyzrgb'].to(device)
    
    print(f"输入形状: {xyzrgb.shape}")
    
    with torch.no_grad():
        reconstruction, slots, masks, sp_feats_proj = model(xyzrgb)
    
    print(f"\n输出形状:")
    print(f"  - reconstruction: {reconstruction.shape}")
    print(f"  - slots: {slots.shape}")
    print(f"  - masks: {masks.shape}")
    print(f"  - sp_feats_proj: {sp_feats_proj.shape}")
    
    # 验证形状
    B = xyzrgb.shape[0]
    assert reconstruction.shape == (B, 512, 768), f"reconstruction形状错误: {reconstruction.shape}"
    assert slots.shape == (B, 16, 256), f"slots形状错误: {slots.shape}"
    assert masks.shape == (B, 16, 512), f"masks形状错误: {masks.shape}"
    assert sp_feats_proj.shape == (B, 512, 768), f"sp_feats_proj形状错误: {sp_feats_proj.shape}"
    
    print("✓ 所有输出形状正确")
    
    return reconstruction, slots, masks, sp_feats_proj


def test_loss_computation(reconstruction, slots, masks, sp_feats_proj, config):
    """测试损失计算"""
    print("\n" + "=" * 60)
    print("[4/5] 测试损失计算")
    print("=" * 60)
    
    criterion = DINOSAURLoss(config['loss']['weights'])
    
    loss, loss_dict = criterion(reconstruction, sp_feats_proj, slots, masks)
    
    print(f"总损失: {loss.item():.6f}")
    print(f"  - reconstruction: {loss_dict['reconstruction']:.6f}")
    print(f"  - mask_entropy: {loss_dict['mask_entropy']:.6f}")
    print(f"  - slot_diversity: {loss_dict['slot_diversity']:.6f}")
    print(f"  - mask_sparsity: {loss_dict['mask_sparsity']:.6f}")
    
    # 验证loss是有限的
    assert torch.isfinite(loss), "Loss包含NaN或Inf"
    
    print("✓ 损失计算正常")
    
    return loss


def test_backward_pass(model, train_loader, criterion, device):
    """测试反向传播"""
    print("\n" + "=" * 60)
    print("[5/5] 测试反向传播")
    print("=" * 60)
    
    model.train()
    
    # 创建优化器
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(),
        lr=0.0001,
        weight_decay=0.05
    )
    
    # 获取一个batch
    batch = next(iter(train_loader))
    xyzrgb = batch['xyzrgb'].to(device)
    
    # 前向传播
    reconstruction, slots, masks, sp_feats_proj = model(xyzrgb)
    
    # 计算损失
    loss, loss_dict = criterion(reconstruction, sp_feats_proj, slots, masks)
    
    print(f"训练损失: {loss.item():.6f}")
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    
    # 检查梯度
    grad_norm = torch.nn.utils.clip_grad_norm_(model.get_trainable_params(), 1.0)
    print(f"梯度范数: {grad_norm:.6f}")
    
    # 验证梯度是有限的
    assert torch.isfinite(grad_norm), "梯度包含NaN或Inf"
    
    # 优化器步进
    optimizer.step()
    
    print("✓ 反向传播正常")
    print("✓ 梯度更新正常")


def main():
    """主测试流程"""
    print("=" * 60)
    print("PointBERT + DINOSAUR 完整流程测试")
    print("=" * 60)
    
    # 加载配置
    config_path = os.path.join(current_dir, 'config/config_train_pointbert.yaml')
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n使用设备: {device}")
    
    try:
        # 1. 测试数据加载
        train_loader, val_dataset = test_data_loading(config)
        
        # 2. 测试模型创建
        model = test_model_creation(config, device)
        
        # 3. 测试前向传播
        reconstruction, slots, masks, sp_feats_proj = test_forward_pass(
            model, train_loader, device
        )
        
        # 4. 测试损失计算
        loss = test_loss_computation(
            reconstruction, slots, masks, sp_feats_proj, config
        )
        
        # 5. 测试反向传播
        criterion = DINOSAURLoss(config['loss']['weights'])
        test_backward_pass(model, train_loader, criterion, device)
        
        # 总结
        print("\n" + "=" * 60)
        print("✅ 所有测试通过！")
        print("=" * 60)
        print("\n可以开始完整训练:")
        print("  python train_3d_pointbert.py --test_run")
        print("=" * 60)
        
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"❌ 测试失败: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


