"""
S3DIS数据集加载器
用于DINOSAUR 3D训练
"""

import os
import glob
import numpy as np
import torch
import warnings
from torch.utils.data import Dataset
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph


class S3DISDataset(Dataset):
    """
    S3DIS数据集
    
    数据格式: 每个房间一个txt文件，每行6个值 (x, y, z, r, g, b)
    """
    
    def __init__(self, root_dir, areas, n_superpoints=512, augment=False, aug_config=None, ignore_warnings=True):
        """
        Args:
            root_dir: S3DIS根目录
            areas: 区域列表，如[1,2,3,4,6]
            n_superpoints: 固定超点数量
            augment: 是否使用数据增强
            aug_config: 增强配置dict
            ignore_warnings: 是否忽略sklearn警告
        """
        self.root_dir = root_dir
        self.areas = areas
        self.n_superpoints = n_superpoints
        self.augment = augment
        self.aug_config = aug_config or {}
        self.ignore_warnings = ignore_warnings
        
        # 设置警告过滤
        if self.ignore_warnings:
            warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        
        # 收集所有房间文件
        self.room_files = []
        for area_id in areas:
            area_dir = os.path.join(root_dir, f'Area_{area_id}')
            if not os.path.exists(area_dir):
                print(f"警告: 区域目录不存在: {area_dir}")
                continue
            
            # 查找所有房间的Annotations目录
            room_dirs = glob.glob(os.path.join(area_dir, '*'))
            for room_dir in room_dirs:
                if not os.path.isdir(room_dir):
                    continue
                
                anno_dir = os.path.join(room_dir, 'Annotations')
                if not os.path.exists(anno_dir):
                    continue
                
                # 收集房间所有物体的txt文件
                txt_files = glob.glob(os.path.join(anno_dir, '*.txt'))
                if len(txt_files) > 0:
                    self.room_files.append((room_dir, txt_files))
        
        print(f"加载了 {len(self.room_files)} 个房间 从区域 {areas}")
    
    def __len__(self):
        return len(self.room_files)
    
    def load_room(self, room_dir, txt_files):
        """
        加载一个房间的所有点云数据
        
        Returns:
            xyz: (N, 3)
            rgb: (N, 3)
        """
        all_points = []
        
        for txt_file in txt_files:
            try:
                data = np.loadtxt(txt_file)
                if data.ndim == 1:
                    data = data.reshape(1, -1)
                all_points.append(data)
            except:
                continue
        
        if len(all_points) == 0:
            # 返回一个小的空房间
            return np.zeros((100, 3), dtype=np.float32), np.zeros((100, 3), dtype=np.float32)
        
        all_points = np.concatenate(all_points, axis=0)
        
        xyz = all_points[:, :3].astype(np.float32)
        rgb = all_points[:, 3:6].astype(np.float32)
        
        # RGB归一化到[-1, 1]
        rgb = rgb / 127.5 - 1.0
        
        return xyz, rgb
    
    def generate_superpoints(self, xyz, rgb):
        """
        使用层次聚类生成固定数量的超点
        """
        n_points = len(xyz)
        
        # 如果点数太少，直接返回
        if n_points < self.n_superpoints:
            # 重复采样到n_superpoints
            indices = np.random.choice(n_points, self.n_superpoints, replace=True)
            labels = np.arange(self.n_superpoints)
            return labels[np.argsort(indices)]
        
        # 对于大场景，先降采样构建图
        if n_points > 30000:
            idx = np.random.choice(n_points, 30000, replace=False)
            xyz_sub = xyz[idx]
            rgb_sub = rgb[idx]
        else:
            idx = np.arange(n_points)
            xyz_sub = xyz
            rgb_sub = rgb
        
        # 构建KNN图
        connectivity = kneighbors_graph(xyz_sub, n_neighbors=10, include_self=False)
        
        # 层次聚类
        cluster = AgglomerativeClustering(
            n_clusters=self.n_superpoints,
            connectivity=connectivity,
            linkage='ward'
        )
        
        labels_sub = cluster.fit_predict(np.concatenate([xyz_sub, rgb_sub], axis=1))
        
        # 如果降采样了，传播标签
        if n_points > 30000:
            from sklearn.neighbors import KNeighborsClassifier
            knn = KNeighborsClassifier(n_neighbors=1)
            knn.fit(xyz_sub, labels_sub)
            labels = knn.predict(xyz)
        else:
            labels = labels_sub
        
        return labels
    
    def augment_data(self, xyz, rgb):
        """
        数据增强
        """
        if not self.augment:
            return xyz, rgb
        
        # 随机旋转（绕z轴）
        if 'rotation_range' in self.aug_config:
            angle_range = self.aug_config['rotation_range']
            angle = np.random.uniform(angle_range[0], angle_range[1])
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            rot_matrix = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=np.float32)
            xyz = xyz @ rot_matrix.T
        
        # 随机缩放
        if 'scale_range' in self.aug_config:
            scale_range = self.aug_config['scale_range']
            scale = np.random.uniform(scale_range[0], scale_range[1])
            xyz = xyz * scale
        
        # 随机平移
        if 'translation' in self.aug_config:
            trans_std = self.aug_config['translation']
            translation = np.random.uniform(-trans_std, trans_std, size=3)
            xyz = xyz + translation
        
        # 坐标抖动
        if 'jitter_sigma' in self.aug_config:
            jitter_sigma = self.aug_config['jitter_sigma']
            xyz = xyz + np.random.normal(0, jitter_sigma, xyz.shape).astype(np.float32)
        
        return xyz, rgb
    
    def __getitem__(self, idx):
        """
        返回:
            sp_coords: (512, 3) 超点中心坐标
            sp_features_placeholder: (512, 3) 用RGB作为占位（实际特征由SPFormer提取）
            xyz_full: (N, 3) 完整点云（用于SPFormer）
            rgb_full: (N, 3) 完整RGB（用于SPFormer）
            sp_labels: (N,) 超点标签
        """
        room_dir, txt_files = self.room_files[idx]
        
        # 加载房间
        xyz, rgb = self.load_room(room_dir, txt_files)
        
        # 数据增强
        xyz, rgb = self.augment_data(xyz, rgb)
        
        # 生成超点
        sp_labels = self.generate_superpoints(xyz, rgb)
        
        # 计算超点中心
        sp_coords = np.zeros((self.n_superpoints, 3), dtype=np.float32)
        for sp_id in range(self.n_superpoints):
            mask = (sp_labels == sp_id)
            if mask.sum() > 0:
                sp_coords[sp_id] = xyz[mask].mean(axis=0)
        
        # 归一化超点坐标到[-1, 1]
        coords_min = sp_coords.min(axis=0)
        coords_max = sp_coords.max(axis=0)
        sp_coords_norm = (sp_coords - coords_min) / (coords_max - coords_min + 1e-8)
        sp_coords_norm = sp_coords_norm * 2 - 1
        
        return {
            'sp_coords': torch.from_numpy(sp_coords_norm).float(),
            'xyz_full': torch.from_numpy(xyz).float(),
            'rgb_full': torch.from_numpy(rgb).float(),
            'sp_labels': torch.from_numpy(sp_labels).long(),
            'room_name': os.path.basename(room_dir)
        }


def collate_fn(batch):
    """
    Batch collate函数
    """
    sp_coords = torch.stack([item['sp_coords'] for item in batch])
    
    # xyz_full和rgb_full长度不同，不能直接stack
    # 返回list
    xyz_full = [item['xyz_full'] for item in batch]
    rgb_full = [item['rgb_full'] for item in batch]
    sp_labels = [item['sp_labels'] for item in batch]
    room_names = [item['room_name'] for item in batch]
    
    return {
        'sp_coords': sp_coords,  # [B, 512, 3]
        'xyz_full': xyz_full,    # List[Tensor(N_i, 3)]
        'rgb_full': rgb_full,    # List[Tensor(N_i, 3)]
        'sp_labels': sp_labels,  # List[Tensor(N_i,)]
        'room_names': room_names
    }

