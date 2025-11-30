"""
DINOSAUR 3D训练脚本 (SPFormer特征输入)

训练流程:
1. S3DIS数据加载 → 超点生成
2. SPFormer提取超点特征 (冻结)
3. 特征投影 32→384维
4. DINOSAUR ISA处理
5. 4项损失计算
6. TensorBoard监控
"""

import os
import sys
import time
import argparse
import yaml
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# TensorBoard导入（可选）
try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    print("警告: TensorBoard未安装，将跳过TensorBoard日志")
    HAS_TENSORBOARD = False
    SummaryWriter = None

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '../SPFormer'))

# 导入自定义模块
from data.s3dis_dataset import S3DISDataset, collate_fn
from models.model import DINOSAURpp
from models.spformer_wrapper import FeatureProjector, SPFormerDINOSAUR
from models.losses import DINOSAURLoss

# 导入可视化工具（使用完整路径避免与原utils.py冲突）
sys.path.insert(0, os.path.join(current_dir, 'utils'))
from visualizer import visualize_slot_assignment, visualize_reconstruction_error, visualize_slot_statistics

# 导入SPFormer
from test_3d_isa import TestSPFormerExtractor


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_distributed():
    """设置分布式训练"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        
        return rank, world_size, local_rank
    else:
        # 单机单卡
        return 0, 1, 0


def create_model(config):
    """创建模型"""
    import gorilla
    
    # 配置DINOSAUR参数
    class Args:
        def __init__(self):
            self.num_slots = config['model']['num_slots']
            self.slot_dim = config['model']['slot_dim']
            self.slot_att_iter = config['model']['slot_att_iter']
            self.query_opt = config['model']['query_opt']
            self.ISA = config['model']['ISA']
            self.token_num = config['model']['token_num']
            self.num_points = config['model']['num_points']
            self.point_feature_dim = config['model']['din_feature_dim']
    
    args = Args()
    
    # 创建SPFormer提取器
    spformer_config_path = os.path.join(
        current_dir,
        config['data']['spformer_config']
    )
    
    # 检查是否有预训练权重
    spformer_checkpoint = config['data'].get('spformer_checkpoint', None)
    
    if spformer_checkpoint:
        # 使用预训练权重
        spformer_checkpoint_path = os.path.join(current_dir, spformer_checkpoint)
        
        if os.path.exists(spformer_checkpoint_path):
            print(f"[Info] 加载SPFormer预训练权重: {spformer_checkpoint_path}")
            
            # 导入SPFormer相关模块
            from spformer.model import SPFormer
            from test_3d_isa import TestSPFormerExtractor
            
            # 创建提取器（会加载配置）
            spformer_extractor = TestSPFormerExtractor(
                spformer_config_path,
                device='cuda'
            )
            
            # 加载预训练权重
            gorilla.load_checkpoint(
                spformer_extractor.model, 
                spformer_checkpoint_path,
                strict=False
            )
            
            # 冻结SPFormer
            for param in spformer_extractor.model.parameters():
                param.requires_grad = False
            spformer_extractor.model.eval()
            
            frozen_params = sum(1 for p in spformer_extractor.model.parameters() if not p.requires_grad)
            total_params = sum(1 for p in spformer_extractor.model.parameters())
            print(f"[Info] ✓ SPFormer权重加载成功")
            print(f"[Info] ✓ SPFormer已冻结 ({frozen_params}/{total_params} 参数)")
        else:
            print(f"[Warning] 预训练权重不存在: {spformer_checkpoint_path}")
            print(f"[Warning] 使用随机初始化")
            spformer_extractor = TestSPFormerExtractor(
                spformer_config_path,
                device='cuda'
            )
    else:
        # 随机初始化
        print(f"[Info] SPFormer使用随机初始化（无预训练权重）")
        spformer_extractor = TestSPFormerExtractor(
            spformer_config_path,
            device='cuda'
        )
    
    # 创建投影层
    projector = FeatureProjector(
        in_dim=config['model']['spformer_dim'],
        out_dim=config['model']['din_feature_dim']
    )
    
    # 创建DINOSAUR
    dinosaur = DINOSAURpp(args)
    
    # 封装完整模型
    model = SPFormerDINOSAUR(spformer_extractor, projector, dinosaur)
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, epoch, config, writer, rank):
    """训练一个epoch"""
    model.train()
    
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'mask_entropy': 0.0,
        'slot_diversity': 0.0,
        'mask_uniformity': 0.0
    }
    
    if rank == 0:
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    else:
        pbar = train_loader
    
    for iter_idx, batch in enumerate(pbar):
        sp_coords = batch['sp_coords'].cuda()
        xyz_full = batch['xyz_full']
        rgb_full = batch['rgb_full']
        sp_labels = batch['sp_labels']
        
        optimizer.zero_grad()
        
        # 混合精度前向传播
        with autocast(enabled=config['train']['use_amp']):
            try:
                reconstruction, slots, masks, sp_feats_proj = model(
                    xyz_full, rgb_full, sp_labels, sp_coords
                )
                
                # 检查NaN
                if torch.isnan(reconstruction).any() or torch.isnan(slots).any():
                    print(f"[警告] Epoch {epoch}, Iter {iter_idx}: 检测到NaN，跳过此batch")
                    continue
                
                # 计算损失
                loss, loss_dict = criterion(reconstruction, sp_feats_proj, slots, masks)
                
                # 检查loss是否为NaN
                if torch.isnan(loss):
                    print(f"[警告] Epoch {epoch}, Iter {iter_idx}: Loss为NaN，跳过此batch")
                    continue
                    
            except AssertionError as e:
                print(f"[错误] Epoch {epoch}, Iter {iter_idx}: {e}")
                print("[跳过此batch并继续训练]")
                continue
        
        # 反向传播
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        
        # 梯度裁剪并记录
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.get_trainable_params(),
            config['train']['grad_clip_norm']
        )
        
        # 检查梯度异常
        if torch.isnan(grad_norm) or grad_norm > 100:
            print(f"[警告] Epoch {epoch}, Iter {iter_idx}: 梯度异常 (norm={grad_norm:.2f})，跳过更新")
            optimizer.zero_grad()
            continue
        
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # 累积损失
        for key in total_losses.keys():
            total_losses[key] += loss_dict[key]
        
        # 日志记录
        if rank == 0:
            if iter_idx % config['log_interval'] == 0:
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix({
                    'loss': f"{loss_dict['total']:.4f}",
                    'recon': f"{loss_dict['reconstruction']:.4f}",
                    'lr': f"{current_lr:.6f}"
                })
                
                # TensorBoard
                if writer is not None:
                    global_step = epoch * len(train_loader) + iter_idx
                    writer.add_scalar('Train/total_loss', loss_dict['total'], global_step)
                    writer.add_scalar('Train/reconstruction_loss', loss_dict['reconstruction'], global_step)
                    writer.add_scalar('Train/mask_entropy', loss_dict['mask_entropy'], global_step)
                    writer.add_scalar('Train/slot_diversity', loss_dict['slot_diversity'], global_step)
                    writer.add_scalar('Train/mask_uniformity', loss_dict['mask_uniformity'], global_step)
                    writer.add_scalar('Train/learning_rate', current_lr, global_step)
    
    # 平均损失
    avg_losses = {k: v / len(train_loader) for k, v in total_losses.items()}
    
    return avg_losses


@torch.no_grad()
def validate(model, val_loader, criterion, epoch, config, writer, rank):
    """验证 - 计算所有损失项"""
    model.eval()
    
    total_losses = {
        'total': 0.0,
        'reconstruction': 0.0,
        'mask_entropy': 0.0,
        'slot_diversity': 0.0,
        'mask_uniformity': 0.0
    }
    
    if rank == 0:
        pbar = tqdm(val_loader, desc=f'Validation')
    else:
        pbar = val_loader
    
    with torch.no_grad():
        for batch in pbar:
            sp_coords = batch['sp_coords'].cuda()
            xyz_full = batch['xyz_full']
            rgb_full = batch['rgb_full']
            sp_labels = batch['sp_labels']
            
            reconstruction, slots, masks, sp_feats_proj = model(
                xyz_full, rgb_full, sp_labels, sp_coords
            )
            
            # 计算所有损失项
            loss, loss_dict = criterion(reconstruction, sp_feats_proj, slots, masks)
            
            # 累积所有损失
            for key in total_losses.keys():
                if key in loss_dict:
                    total_losses[key] += loss_dict[key]
    
    # 计算平均损失
    avg_losses = {key: val / len(val_loader) for key, val in total_losses.items()}
    
    if rank == 0:
        print(f"\n验证损失:")
        print(f"  总损失: {avg_losses['total']:.6f}")
        print(f"  重建损失: {avg_losses['reconstruction']:.6f}")
        print(f"  Mask熵损失: {avg_losses['mask_entropy']:.6f}")
        print(f"  Slot多样性损失: {avg_losses['slot_diversity']:.6f}")
        print(f"  Mask均匀性损失: {avg_losses['mask_uniformity']:.6f}")
        
        if writer is not None:
            writer.add_scalar('Val/total_loss', avg_losses['total'], epoch)
            writer.add_scalar('Val/reconstruction_loss', avg_losses['reconstruction'], epoch)
            writer.add_scalar('Val/mask_entropy_loss', avg_losses['mask_entropy'], epoch)
            writer.add_scalar('Val/slot_diversity_loss', avg_losses['slot_diversity'], epoch)
            writer.add_scalar('Val/mask_uniformity_loss', avg_losses['mask_uniformity'], epoch)
    
    return avg_losses['total']  # 返回总损失用于最佳模型判断


@torch.no_grad()
def visualize_samples(model, vis_samples, epoch, config, output_dir):
    """可视化样本"""
    model.eval()
    
    vis_dir = os.path.join(output_dir, f'epoch_{epoch:03d}')
    os.makedirs(vis_dir, exist_ok=True)
    
    for idx, sample in enumerate(vis_samples):
        sp_coords = sample['sp_coords'].unsqueeze(0).cuda()
        xyz_full = [sample['xyz_full'].cuda()]
        rgb_full = [sample['rgb_full'].cuda()]
        sp_labels = [sample['sp_labels'].cuda()]
        
        reconstruction, slots, masks, sp_feats_proj = model(
            xyz_full, rgb_full, sp_labels, sp_coords
        )
        
        # 转换为numpy
        xyz_np = xyz_full[0].cpu().numpy()
        sp_labels_np = sp_labels[0].cpu().numpy()
        masks_np = masks[0].cpu().numpy()
        recon_np = reconstruction[0].cpu().numpy()
        sp_feats_np = sp_feats_proj[0].cpu().numpy()
        
        # 可视化
        visualize_slot_assignment(
            xyz_np, sp_labels_np, masks_np,
            os.path.join(vis_dir, f'sample_{idx}_slot_assignment.png'),
            num_slots=config['model']['num_slots']
        )
        
        visualize_reconstruction_error(
            xyz_np, sp_labels_np, recon_np, sp_feats_np,
            os.path.join(vis_dir, f'sample_{idx}_recon_error.png')
        )
        
        visualize_slot_statistics(
            masks_np,
            os.path.join(vis_dir, f'sample_{idx}_slot_stats.png')
        )
    
    print(f"✓ 可视化已保存到: {vis_dir}")


def save_checkpoint(epoch, model, optimizer, scheduler, val_loss, config, output_dir, is_best=False):
    """保存checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': {
            'projector': model.projector.state_dict(),
            'dinosaur': model.dinosaur.state_dict()
        },
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
        'config': config
    }
    
    # 常规保存
    if epoch % config['checkpoint']['save_interval'] == 0:
        save_path = os.path.join(output_dir, f'epoch_{epoch:03d}.pth')
        torch.save(checkpoint, save_path)
        print(f"✓ Checkpoint已保存: {save_path}")
    
    # 最佳模型
    if is_best:
        best_path = os.path.join(output_dir, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ 最佳模型已保存: {best_path}")


def main():
    # 解析参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config_train_spformer.yaml')
    parser.add_argument('--test_run', action='store_true', help='测试运行模式（仅2个epoch）')
    args = parser.parse_args()
    
    # 加载配置
    config_path = os.path.join(current_dir, args.config)
    config = load_config(config_path)
    
    # GPU绑定（使用配置文件指定的卡）
    gpu_ids = config.get('gpu_ids')
    if gpu_ids:
        gpu_ids_str = ','.join(str(g) for g in gpu_ids)
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids_str
        print(f"[Info] 使用GPU: {gpu_ids_str}")
    
    if args.test_run:
        print("\n⚠️  测试运行模式：仅训练2个epoch")
        config['train']['epochs'] = 2
        config['validation']['val_interval'] = 1
        config['checkpoint']['save_interval'] = 1
    
    # 设置随机种子
    torch.manual_seed(config['seed'])
    
    # 分布式设置
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"DINOSAUR 3D训练 (SPFormer特征)")
        print(f"{'='*60}")
        print(f"配置: {args.config}")
        print(f"设备数: {world_size}")
        print(f"Slot数量: {config['model']['num_slots']}")
        print(f"超点数量: {config['model']['num_points']}")
    
    # 创建输出目录
    output_dir = config['checkpoint']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
    
    # TensorBoard
    writer = None
    if rank == 0 and HAS_TENSORBOARD:
        log_dir = os.path.join(output_dir, 'logs')
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard日志: {log_dir}")
    
    # 创建数据集
    train_dataset = S3DISDataset(
        root_dir=config['data']['s3dis_root'],
        areas=config['data']['train_areas'],
        n_superpoints=config['data']['n_superpoints'],
        augment=True,
        aug_config=config['augmentation'],
        ignore_warnings=config.get('ignore_sklearn_warnings', True)
    )
    
    val_dataset = S3DISDataset(
        root_dir=config['data']['s3dis_root'],
        areas=config['data']['val_areas'],
        n_superpoints=config['data']['n_superpoints'],
        augment=False,
        ignore_warnings=config.get('ignore_sklearn_warnings', True)
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['train']['batch_size_per_gpu'],
        shuffle=True,
        num_workers=config['train']['num_workers'],
        pin_memory=config['train']['pin_memory'],
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    # 准备可视化样本
    vis_samples = [val_dataset[i] for i in range(min(3, len(val_dataset)))]
    
    if rank == 0:
        print(f"\n数据集:")
        print(f"  训练: {len(train_dataset)} 个房间")
        print(f"  验证: {len(val_dataset)} 个房间")
    
    # 创建模型
    model = create_model(config).cuda()
    
    if world_size > 1:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            find_unused_parameters=config['distributed']['find_unused_parameters']
        )
    
    # 损失函数
    criterion = DINOSAURLoss(config['loss']['weights'])
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.get_trainable_params() if world_size == 1 else model.module.get_trainable_params(),
        lr=config['train']['optimizer']['lr'],
        weight_decay=config['train']['optimizer']['weight_decay'],
        betas=tuple(config['train']['optimizer']['betas'])
    )
    
    # 学习率调度器 (PolyLR)
    total_iters = len(train_loader) * config['train']['epochs']
    warmup_iters = len(train_loader) * config['train']['warmup_epochs']
    
    def lr_lambda(current_iter):
        if current_iter < warmup_iters:
            return current_iter / warmup_iters
        else:
            progress = (current_iter - warmup_iters) / (total_iters - warmup_iters)
            return (1 - progress) ** config['train']['scheduler']['power']
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 混合精度
    scaler = GradScaler(enabled=config['train']['use_amp'])
    
    if rank == 0:
        print(f"\n模型创建完成")
        print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # 训练循环
    best_val_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(config['train']['epochs']):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Epoch {epoch+1}/{config['train']['epochs']}")
            print(f"{'='*60}")
        
        # 训练
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler,
            epoch, config, writer, rank
        )
        
        if rank == 0:
            print(f"\n训练损失: {train_losses['total']:.6f}")
            if writer is not None:
                writer.add_scalar('Epoch/train_loss', train_losses['total'], epoch)
        
        # 验证
        if (epoch + 1) % config['validation']['val_interval'] == 0:
            val_loss = validate(
                model, val_loader, criterion, epoch, config, writer, rank
            )
            
            # 保存最佳模型
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            if rank == 0:
                save_checkpoint(
                    epoch, model.module if world_size > 1 else model,
                    optimizer, scheduler, val_loss, config, output_dir, is_best
                )
        
        # 可视化
        if rank == 0 and (epoch + 1) % config['checkpoint']['save_interval'] == 0:
            visualize_samples(
                model.module if world_size > 1 else model,
                vis_samples, epoch, config,
                os.path.join(output_dir, 'visualizations')
            )
    
    # 训练完成
    total_time = time.time() - start_time
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"总时间: {total_time/3600:.2f} 小时")
        print(f"最佳验证损失: {best_val_loss:.6f}")
        print(f"{'='*60}\n")
        if writer is not None:
            writer.close()
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

