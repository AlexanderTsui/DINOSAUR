"""
S3DIS数据集 - Mask3D版本
为Mask3D特征提取优化的数据加载器

说明：
1. 采样到更多点（80000），因为Mask3D内部会体素化
2. 不需要严格的点数要求
3. 返回分离的xyz和rgb（Mask3D需要分开输入）
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import pickle


class S3DISMask3DDataset(Dataset):
    """
    S3DIS数据集 - 为Mask3D特征提取优化
    
    数据流：
    1. 加载房间点云 (xyz, rgb)
    2. 重采样到固定点数（如80000）
    3. 归一化xyz和rgb
    4. 返回分离的xyz和rgb
    """
    
    def __init__(
        self,
        root_dir,
        areas=[1, 2, 3, 4, 5, 6],
        max_points=80000,
        augment=False,
        aug_config=None
    ):
        """
        参数:
            root_dir: S3DIS根目录
            areas: 使用的Area列表
            max_points: 最大采样点数
            augment: 是否数据增强
            aug_config: 增强参数配置
        """
        self.root_dir = root_dir
        self.areas = areas
        self.max_points = max_points
        self.augment = augment
        self.aug_config = aug_config or {}
        
        # 使用缓存文件加速加载
        cache_filename = f"s3dis_mask3d_areas_{'_'.join(map(str, sorted(areas)))}.pkl"
        cache_path = os.path.join(root_dir, cache_filename)
        
        if os.path.exists(cache_path):
            print(f"[S3DIS-Mask3D] 从缓存加载文件列表: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.room_files = pickle.load(f)
        else:
            print("[S3DIS-Mask3D] 正在扫描文件列表 (首次运行)...")
            self.room_files = []
            for area in areas:
                area_path = os.path.join(root_dir, f'Area_{area}')
                if not os.path.exists(area_path):
                    print(f"[Warning] Area {area} 不存在: {area_path}")
                    continue
                
                pattern = os.path.join(area_path, '*', '*.txt')
                area_files = glob.glob(pattern)
                
                for f in area_files:
                    if 'ReadMe' in f or 'Annotations' in f:
                        continue
                    parent_name = os.path.basename(os.path.dirname(f))
                    file_name = os.path.basename(f).replace('.txt', '')
                    if parent_name == file_name:
                        self.room_files.append(f)
            
            print(f"[S3DIS-Mask3D] 扫描完成，正在保存缓存: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.room_files, f)
        
        if len(self.room_files) == 0:
            raise ValueError(f"未找到有效的S3DIS房间文件！检查路径: {root_dir}")
        
        print(f"[S3DIS-Mask3D] 找到 {len(self.room_files)} 个房间 (Areas: {areas})")
    
    def __len__(self):
        return len(self.room_files)
    
    def load_room(self, file_path):
        """
        加载S3DIS房间文件
        返回: xyz (N, 3), rgb (N, 3)
        """
        try:
            data = np.loadtxt(file_path, dtype=np.float32)
        except Exception as e:
            print(f"[Warning] 文件损坏，跳过: {file_path}")
            print(f"[Warning] 错误信息: {e}")
            return None, None
        
        xyz = data[:, 0:3]
        rgb = data[:, 3:6]
        
        # RGB归一化到[0, 1]
        if rgb.max() > 1.1:
            rgb = rgb / 255.0
        
        return xyz, rgb
    
    def resample_points(self, xyz, rgb):
        """
        重采样到固定点数
        """
        n_points = len(xyz)
        
        if n_points > self.max_points:
            # 下采样
            idx = np.random.choice(n_points, self.max_points, replace=False)
        elif n_points < self.max_points:
            # 上采样（重复采样）
            idx = np.random.choice(n_points, self.max_points, replace=True)
        else:
            idx = np.arange(n_points)
        
        return xyz[idx], rgb[idx]
    
    def augment_point_cloud(self, xyz, rgb):
        """
        数据增强
        """
        if not self.augment:
            return xyz, rgb
        
        # 1. 随机旋转（绕z轴）
        if self.aug_config.get('rotation_range'):
            angle_range = self.aug_config['rotation_range']
            angle = np.random.uniform(angle_range[0], angle_range[1]) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            xyz = xyz @ rotation_matrix.T
        
        # 2. 随机缩放
        if self.aug_config.get('scale_range'):
            scale_range = self.aug_config['scale_range']
            scale = np.random.uniform(scale_range[0], scale_range[1])
            xyz = xyz * scale
        
        # 3. 随机平移
        if self.aug_config.get('translation'):
            translation = self.aug_config['translation']
            xyz = xyz + np.random.uniform(-translation, translation, size=(3,))
        
        # 4. 坐标抖动
        if self.aug_config.get('jitter_sigma'):
            sigma = self.aug_config['jitter_sigma']
            xyz = xyz + np.random.normal(0, sigma, size=xyz.shape).astype(np.float32)
        
        # 5. RGB颜色抖动
        if self.aug_config.get('color_jitter'):
            jitter = self.aug_config['color_jitter']
            rgb = rgb + np.random.uniform(-jitter, jitter, size=rgb.shape).astype(np.float32)
            rgb = np.clip(rgb, 0, 1)
        
        return xyz, rgb
    
    def __getitem__(self, idx):
        """
        返回格式:
        {
            'xyz': Tensor (max_points, 3)  # 坐标
            'rgb': Tensor (max_points, 3)  # 颜色
            'room_name': str               # 房间名（用于可视化）
            'area': str                    # Area名称
        }
        """
        # 1. 加载房间（带重试机制）
        max_retries = 10
        for retry in range(max_retries):
            try:
                file_path = self.room_files[(idx + retry) % len(self.room_files)]
                xyz, rgb = self.load_room(file_path)
                
                # 检查是否加载失败
                if xyz is None or rgb is None:
                    continue
                
                # 提取房间信息
                room_name = os.path.basename(file_path).replace('.txt', '')
                area_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
                
                # 2. 重采样
                xyz, rgb = self.resample_points(xyz, rgb)
                
                # 3. 数据增强
                xyz, rgb = self.augment_point_cloud(xyz, rgb)
                
                # 转换为Tensor
                xyz_tensor = torch.from_numpy(xyz.astype(np.float32))
                rgb_tensor = torch.from_numpy(rgb.astype(np.float32))
                
                return {
                    'xyz': xyz_tensor,      # (max_points, 3)
                    'rgb': rgb_tensor,      # (max_points, 3)
                    'room_name': room_name,
                    'area': area_name
                }
            except Exception as e:
                print(f"[Warning] 加载样本{idx}失败 (尝试{retry+1}/{max_retries}): {e}")
                continue
        
        # 如果所有重试都失败，返回随机数据
        print(f"[Error] 无法加载有效数据，返回随机点云")
        xyz = np.random.randn(self.max_points, 3).astype(np.float32)
        rgb = np.random.rand(self.max_points, 3).astype(np.float32)
        return {
            'xyz': torch.from_numpy(xyz),
            'rgb': torch.from_numpy(rgb),
            'room_name': 'random',
            'area': 'random'
        }


def collate_fn_mask3d(batch):
    """
    Collate函数
    
    输入: batch = [{'xyz': Tensor(N, 3), 'rgb': Tensor(N, 3), ...}, ...]
    输出: {'xyz': Tensor(B, N, 3), 'rgb': Tensor(B, N, 3), ...}
    """
    xyz_list = [item['xyz'] for item in batch]
    rgb_list = [item['rgb'] for item in batch]
    room_names = [item['room_name'] for item in batch]
    areas = [item['area'] for item in batch]
    
    # 直接stack（因为所有样本点数已固定）
    xyz_batch = torch.stack(xyz_list, dim=0)  # (B, N, 3)
    rgb_batch = torch.stack(rgb_list, dim=0)  # (B, N, 3)
    
    return {
        'xyz': xyz_batch,
        'rgb': rgb_batch,
        'room_name': room_names,
        'area': areas
    }


if __name__ == "__main__":
    """
    测试数据集加载
    """
    print("=" * 60)
    print("测试 S3DIS Mask3D Dataset")
    print("=" * 60)
    
    # 配置
    s3dis_root = '/home/pbw/data1/3D_PointCloud_Segmentation/PLSG_Net/dataset/S3DIS/Stanford_Large-Scale_Indoor_Spaces_3D_Dataset/Stanford3dDataset_v1.2_Aligned_Version'
    
    # 创建数据集
    print("\n[1/3] 创建训练集...")
    train_dataset = S3DISMask3DDataset(
        root_dir=s3dis_root,
        areas=[1, 2, 3, 4, 6],
        max_points=80000,
        augment=True,
        aug_config={
            'rotation_range': [-5, 5],
            'scale_range': [0.95, 1.05],
            'translation': 0.1,
            'jitter_sigma': 0.01,
            'color_jitter': 0.05
        }
    )
    print(f"✓ 训练集房间数: {len(train_dataset)}")
    
    print("\n[2/3] 创建验证集...")
    val_dataset = S3DISMask3DDataset(
        root_dir=s3dis_root,
        areas=[5],
        max_points=80000,
        augment=False
    )
    print(f"✓ 验证集房间数: {len(val_dataset)}")
    
    # 测试单个样本
    print("\n[3/3] 测试数据加载...")
    sample = train_dataset[0]
    print(f"✓ XYZ形状: {sample['xyz'].shape}")
    print(f"✓ RGB形状: {sample['rgb'].shape}")
    print(f"✓ 房间名: {sample['room_name']}")
    print(f"✓ Area: {sample['area']}")
    print(f"✓ XYZ范围: [{sample['xyz'].min():.3f}, {sample['xyz'].max():.3f}]")
    print(f"✓ RGB范围: [{sample['rgb'].min():.3f}, {sample['rgb'].max():.3f}]")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    print("\n测试 DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn_mask3d
    )
    
    batch = next(iter(train_loader))
    print(f"✓ Batch XYZ形状: {batch['xyz'].shape}")
    print(f"✓ Batch RGB形状: {batch['rgb'].shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！数据集可以正常使用")
    print("=" * 60)

