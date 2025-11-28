"""
S3DIS数据集 - PointBERT版本
简化版：不需要预计算超点标签，PointBERT内部处理超点生成

与SPFormer版本的主要差异：
1. 固定采样到8192点（PointBERT输入要求）
2. 不需要生成超点标签和KNN图
3. collate_fn更简单（直接stack）
4. 返回xyzrgb合并格式
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import pickle


class S3DISPointBERTDataset(Dataset):
    """
    S3DIS数据集 - 为PointBERT特征提取优化
    
    数据流：
    1. 加载房间点云 (xyz, rgb)
    2. 重采样到固定8192点
    3. 归一化xyz和rgb
    4. 合并为[N, 6]格式
    """
    
    def __init__(
        self,
        root_dir,
        areas=[1, 2, 3, 4, 5, 6],
        target_points=8192,
        augment=False,
        aug_config=None
    ):
        """
        参数:
            root_dir: S3DIS根目录
            areas: 使用的Area列表
            target_points: 重采样点数（PointBERT默认8192）
            augment: 是否数据增强
            aug_config: 增强参数配置
        """
        self.root_dir = root_dir
        self.areas = areas
        self.target_points = target_points
        self.augment = augment
        self.aug_config = aug_config or {}
        
        # 使用缓存文件加速加载
        cache_filename = f"s3dis_pointbert_areas_{'_'.join(map(str, sorted(areas)))}.pkl"
        cache_path = os.path.join(root_dir, cache_filename)
        
        if os.path.exists(cache_path):
            print(f"[S3DIS-PointBERT] 从缓存加载文件列表: {cache_path}")
            with open(cache_path, 'rb') as f:
                self.room_files = pickle.load(f)
        else:
            print("[S3DIS-PointBERT] 正在扫描文件列表 (首次运行)...")
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
            
            print(f"[S3DIS-PointBERT] 扫描完成，正在保存缓存: {cache_path}")
            with open(cache_path, 'wb') as f:
                pickle.dump(self.room_files, f)
        
        if len(self.room_files) == 0:
            raise ValueError(f"未找到有效的S3DIS房间文件！检查路径: {root_dir}")
        
        print(f"[S3DIS-PointBERT] 找到 {len(self.room_files)} 个房间 (Areas: {areas})")
    
    def __len__(self):
        return len(self.room_files)
    
    def load_room(self, file_path):
        """
        加载S3DIS房间文件
        返回: xyz (N, 3), rgb (N, 3)
        """
        # 读取点云数据
        try:
            data = np.loadtxt(file_path, dtype=np.float32)
        except Exception as e:
            print(f"[Warning] 文件损坏，跳过: {file_path}")
            print(f"[Warning] 错误信息: {e}")
            # 返回None，在__getitem__中处理
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
        
        if n_points > self.target_points:
            # 下采样
            idx = np.random.choice(n_points, self.target_points, replace=False)
        elif n_points < self.target_points:
            # 上采样（重复采样）
            idx = np.random.choice(n_points, self.target_points, replace=True)
        else:
            idx = np.arange(n_points)
        
        return xyz[idx], rgb[idx]
    
    def normalize_xyz(self, xyz):
        """
        归一化坐标到单位球
        1. 中心化
        2. 缩放到[-1, 1]
        """
        # 中心化
        centroid = np.mean(xyz, axis=0)
        xyz = xyz - centroid
        
        # 缩放到单位球
        max_dist = np.max(np.sqrt(np.sum(xyz**2, axis=1)))
        xyz = xyz / (max_dist + 1e-8)
        
        return xyz
    
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
            'xyzrgb': Tensor (8192, 6)  # 合并的xyz+rgb
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
                
                # 2. 重采样
                xyz, rgb = self.resample_points(xyz, rgb)
                
                # 3. 数据增强（在归一化之前）
                xyz, rgb = self.augment_point_cloud(xyz, rgb)
                
                # 4. 归一化xyz
                xyz = self.normalize_xyz(xyz)
                
                # 5. 合并xyz和rgb
                xyzrgb = np.concatenate([xyz, rgb], axis=1).astype(np.float32)
                
                return {
                    'xyzrgb': torch.from_numpy(xyzrgb)  # (8192, 6)
                }
            except Exception as e:
                print(f"[Warning] 加载样本{idx}失败 (尝试{retry+1}/{max_retries}): {e}")
                continue
        
        # 如果所有重试都失败，返回随机数据
        print(f"[Error] 无法加载有效数据，返回随机点云")
        xyzrgb = np.random.randn(8192, 6).astype(np.float32)
        xyzrgb[:, 3:] = np.clip(xyzrgb[:, 3:], 0, 1)  # RGB范围
        return {
            'xyzrgb': torch.from_numpy(xyzrgb)
        }


def collate_fn_pointbert(batch):
    """
    简化的collate函数
    
    输入: batch = [{'xyzrgb': Tensor(8192, 6)}, ...]
    输出: {'xyzrgb': Tensor(B, 8192, 6)}
    """
    xyzrgb_list = [item['xyzrgb'] for item in batch]
    
    # 直接stack（因为所有样本都是8192点）
    xyzrgb_batch = torch.stack(xyzrgb_list, dim=0)  # (B, 8192, 6)
    
    return {
        'xyzrgb': xyzrgb_batch
    }


if __name__ == "__main__":
    """
    测试数据集加载
    """
    print("=" * 60)
    print("测试 S3DIS PointBERT Dataset")
    print("=" * 60)
    
    # 配置
    s3dis_root = '/home/pbw/data1/3D_PointCloud_Segmentation/PLSG_Net/dataset/S3DIS/Stanford_Large-Scale_Indoor_Spaces_3D_Dataset/Stanford3dDataset_v1.2_Aligned_Version'
    
    # 创建数据集
    print("\n[1/3] 创建训练集...")
    train_dataset = S3DISPointBERTDataset(
        root_dir=s3dis_root,
        areas=[1, 2, 3, 4, 6],
        target_points=8192,
        augment=True,
        aug_config={
            'rotation_range': [-5, 5],
            'scale_range': [0.95, 1.05],
            'translation': 0.05,
            'jitter_sigma': 0.01,
            'color_jitter': 0.05
        }
    )
    print(f"✓ 训练集房间数: {len(train_dataset)}")
    
    print("\n[2/3] 创建验证集...")
    val_dataset = S3DISPointBERTDataset(
        root_dir=s3dis_root,
        areas=[5],
        target_points=8192,
        augment=False
    )
    print(f"✓ 验证集房间数: {len(val_dataset)}")
    
    # 测试单个样本
    print("\n[3/3] 测试数据加载...")
    sample = train_dataset[0]
    print(f"✓ 样本形状: {sample['xyzrgb'].shape}")
    print(f"✓ XYZ范围: [{sample['xyzrgb'][:, :3].min():.3f}, {sample['xyzrgb'][:, :3].max():.3f}]")
    print(f"✓ RGB范围: [{sample['xyzrgb'][:, 3:].min():.3f}, {sample['xyzrgb'][:, 3:].max():.3f}]")
    
    # 测试DataLoader
    from torch.utils.data import DataLoader
    
    print("\n测试 DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn_pointbert
    )
    
    batch = next(iter(train_loader))
    print(f"✓ Batch形状: {batch['xyzrgb'].shape}")
    
    print("\n" + "=" * 60)
    print("测试完成！数据集可以正常使用")
    print("=" * 60)

