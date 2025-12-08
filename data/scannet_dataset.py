"""
ScanNet数据集 - DINOSAUR训练版本
仅返回 xyz/rgb，不依赖 SupervisedSeg 模块

数据格式:
- 点云: scene*_vh_clean_2.ply
- 语义标签: scene*_vh_clean_2.labels.ply (可选)
- 实例标签: aggregation.json + segs.json (可选)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

# 尝试导入PLY读取库
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    try:
        from plyfile import PlyData
        HAS_PLYFILE = True
    except ImportError:
        HAS_PLYFILE = False


def read_ply_points(filepath):
    """读取PLY文件的点和颜色"""
    if HAS_OPEN3D:
        pcd = o3d.io.read_point_cloud(filepath)
        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32)
        return points, colors
    elif HAS_PLYFILE:
        plydata = PlyData.read(filepath)
        vertex = plydata['vertex']
        points = np.stack([vertex['x'], vertex['y'], vertex['z']], axis=-1).astype(np.float32)
        if 'red' in vertex:
            colors = np.stack([vertex['red'], vertex['green'], vertex['blue']], axis=-1).astype(np.float32)
            if colors.max() > 1.0:
                colors = colors / 255.0
        else:
            colors = np.zeros((len(points), 3), dtype=np.float32)
        return points, colors
    else:
        raise ImportError("需要安装open3d或plyfile库来读取PLY文件: pip install open3d 或 pip install plyfile")


class ScanNetDataset(Dataset):
    """
    ScanNet数据集 - 仅返回 xyz/rgb，用于 DINOSAUR 无监督训练
    
    每个样本返回:
    - xyz: (N, 3) 点坐标
    - rgb: (N, 3) 颜色 [0, 1]
    - scene_name: str
    """
    
    def __init__(
        self,
        root_dir,
        split='train',
        max_points=80000,
        augment=False,
        aug_config=None
    ):
        """
        Args:
            root_dir: ScanNet数据根目录 (scans目录或其父目录)
            split: 'train', 'val', 或 'test'
            max_points: 最大采样点数
            augment: 是否数据增强
            aug_config: 增强配置
        """
        self.root_dir = root_dir
        self.split = split
        self.max_points = max_points
        self.augment = augment
        self.aug_config = aug_config or {}
        
        # 兼容 root 直接指向 scans 目录或其父目录
        scans_basename = os.path.basename(self.root_dir.rstrip(os.sep))
        self.scans_dir = self.root_dir if scans_basename == 'scans' else os.path.join(self.root_dir, 'scans')
        self.split_dir = self.root_dir if os.path.isfile(os.path.join(self.root_dir, f'scannetv2_{self.split}.txt')) else os.path.dirname(self.scans_dir)
        
        # 获取场景列表
        self.scene_list = self._get_scene_list()
        
        print(f"[ScanNet] {split}集: {len(self.scene_list)} 个场景")
    
    def _get_scene_list(self):
        """获取场景列表"""
        # 检查是否有split文件
        split_file = os.path.join(self.split_dir, f'scannetv2_{self.split}.txt')
        
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                scenes = [line.strip() for line in f if line.strip()]
        else:
            # 没有split文件，使用所有场景
            if not os.path.exists(self.scans_dir):
                print(f"[警告] scans目录不存在: {self.scans_dir}")
                return []
            scenes = [
                d for d in os.listdir(self.scans_dir)
                if os.path.isdir(os.path.join(self.scans_dir, d)) and d.startswith('scene')
            ]
            scenes = sorted(scenes)
            
            # 简单划分: 前80%训练，后20%验证
            n = len(scenes)
            if self.split == 'train':
                scenes = scenes[:int(n * 0.8)]
            elif self.split == 'val':
                scenes = scenes[int(n * 0.8):]
        
        # 过滤掉不存在的场景
        valid_scenes = []
        for scene in scenes:
            ply_path = os.path.join(self.scans_dir, scene, f'{scene}_vh_clean_2.ply')
            if os.path.exists(ply_path):
                valid_scenes.append(scene)
        
        return valid_scenes
    
    def _load_scene(self, scene_name):
        """
        加载单个场景（仅 xyz + rgb）
        
        Returns:
            xyz: (N, 3) 点坐标
            rgb: (N, 3) 颜色 [0, 1]
        """
        scene_dir = os.path.join(self.scans_dir, scene_name)
        ply_path = os.path.join(scene_dir, f'{scene_name}_vh_clean_2.ply')
        xyz, rgb = read_ply_points(ply_path)
        return xyz, rgb
    
    def _augment_point_cloud(self, xyz, rgb):
        """数据增强"""
        if not self.augment:
            return xyz, rgb
        
        # 随机旋转（绕z轴）
        if self.aug_config.get('rotation_range'):
            angle_range = self.aug_config['rotation_range']
            angle = np.random.uniform(angle_range[0], angle_range[1]) * np.pi / 180
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation = np.array([[cos_a, -sin_a, 0], [sin_a, cos_a, 0], [0, 0, 1]], dtype=np.float32)
            xyz = xyz @ rotation.T
        
        # 随机缩放
        if self.aug_config.get('scale_range'):
            scale = np.random.uniform(*self.aug_config['scale_range'])
            xyz = xyz * scale
        
        # 随机平移
        if self.aug_config.get('translation'):
            t = self.aug_config['translation']
            xyz = xyz + np.random.uniform(-t, t, size=(3,))
        
        # 坐标抖动
        if self.aug_config.get('jitter_sigma'):
            xyz = xyz + np.random.normal(0, self.aug_config['jitter_sigma'], xyz.shape)
        
        # 颜色抖动
        if self.aug_config.get('color_jitter'):
            j = self.aug_config['color_jitter']
            rgb = rgb + np.random.uniform(-j, j, rgb.shape)
            rgb = np.clip(rgb, 0, 1)
        
        return xyz.astype(np.float32), rgb.astype(np.float32)
    
    def __len__(self):
        return len(self.scene_list)
    
    def __getitem__(self, idx):
        """
        返回:
        {
            'xyz': Tensor (N, 3)
            'rgb': Tensor (N, 3)
            'scene_name': str
        }
        """
        # 加载场景（带重试机制）
        max_retries = 10
        for retry in range(max_retries):
            try:
                scene_name = self.scene_list[(idx + retry) % len(self.scene_list)]
                xyz, rgb = self._load_scene(scene_name)
                
                if xyz is None or len(xyz) == 0:
                    continue
                
                # 采样到固定点数
                N = len(xyz)
                if N > self.max_points:
                    indices = np.random.choice(N, self.max_points, replace=False)
                elif N < self.max_points:
                    indices = np.random.choice(N, self.max_points, replace=True)
                else:
                    indices = np.arange(N)
                
                xyz = xyz[indices]
                rgb = rgb[indices]
                
                # 数据增强
                xyz, rgb = self._augment_point_cloud(xyz, rgb)
                
                return {
                    'xyz': torch.from_numpy(xyz),
                    'rgb': torch.from_numpy(rgb),
                    'scene_name': scene_name
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
            'scene_name': 'random'
        }


def collate_fn_scannet(batch):
    """
    Collate函数
    
    输入: batch = [{'xyz': Tensor(N, 3), 'rgb': Tensor(N, 3), ...}, ...]
    输出: {'xyz': Tensor(B, N, 3), 'rgb': Tensor(B, N, 3), ...}
    """
    xyz_list = [item['xyz'] for item in batch]
    rgb_list = [item['rgb'] for item in batch]
    scene_names = [item['scene_name'] for item in batch]
    
    # 直接stack（因为所有样本点数已固定）
    xyz_batch = torch.stack(xyz_list, dim=0)  # (B, N, 3)
    rgb_batch = torch.stack(rgb_list, dim=0)  # (B, N, 3)
    
    return {
        'xyz': xyz_batch,
        'rgb': rgb_batch,
        'scene_name': scene_names
    }


if __name__ == "__main__":
    """测试数据集"""
    print("=" * 60)
    print("测试 ScanNet 数据集 (DINOSAUR版本)")
    print("=" * 60)
    
    # 配置
    scannet_root = '/home/pbw/data/scannetv2/download/scans'
    
    # 创建数据集
    dataset = ScanNetDataset(
        root_dir=scannet_root,
        split='train',
        max_points=80000,
        augment=False
    )
    
    print(f"\n数据集大小: {len(dataset)}")
    
    if len(dataset) > 0:
        # 测试单个样本
        sample = dataset[0]
        print(f"\n样本信息:")
        print(f"  - xyz: {sample['xyz'].shape}")
        print(f"  - rgb: {sample['rgb'].shape}")
        print(f"  - scene_name: {sample['scene_name']}")
        
        # 测试DataLoader
        from torch.utils.data import DataLoader
        
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn_scannet)
        batch = next(iter(loader))
        
        print(f"\nBatch信息:")
        print(f"  - xyz: {batch['xyz'].shape}")
        print(f"  - rgb: {batch['rgb'].shape}")
    
    print("\n测试完成！")

