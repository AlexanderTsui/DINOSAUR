"""
特征提取和可视化分析脚本

功能：
1. 提取Encoder特征（projected features）和Slot特征
2. PCA/t-SNE可视化特征分布
3. 分析特征质量（类内/类间距离、可分性）
4. 可视化slot绑定情况
5. 定位slot绑定问题的根源

用法：
python analyze_features.py --config config/config_train_concerto_scannet.yaml \
    --checkpoint checkpoints/checkpoints_concerto/concerto_scannet_origin/epoch_200.pth \
    --dataset scannet \
    --num_samples 20 \
    --output_dir analysis_results/
"""

import os
import sys
import argparse
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
# import seaborn as sns  # Not required, using matplotlib only
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import importlib.util
from tqdm import tqdm
import pickle

# 设置matplotlib支持中文
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)


def _import_module_from_path(module_name, file_path):
    """从指定路径导入模块"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# 导入必要模块
_mask3d_wrapper = _import_module_from_path(
    'dinosaur_mask3d_wrapper',
    os.path.join(current_dir, 'models', 'mask3d_wrapper.py')
)
create_mask3d_dinosaur_model = _mask3d_wrapper.create_mask3d_dinosaur_model
create_logosp_dinosaur_model = _mask3d_wrapper.create_logosp_dinosaur_model
create_concerto_dinosaur_model = _mask3d_wrapper.create_concerto_dinosaur_model

# 导入数据集
_s3dis_dataset = _import_module_from_path(
    's3dis_dataset_module',
    os.path.join(current_dir, 'data', 's3dis_dataset_mask3d.py')
)
S3DISDataset = _s3dis_dataset.S3DISMask3DDataset  # Correct class name

_scannet_dataset = _import_module_from_path(
    'scannet_dataset_module',
    os.path.join(current_dir, 'data', 'scannet_dataset.py')
)
ScanNetDataset = _scannet_dataset.ScanNetDataset


class FeatureAnalyzer:
    """特征分析器"""

    def __init__(self, model, dataset, device='cuda', max_samples=20):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.max_samples = max_samples

        # 存储提取的特征
        self.encoder_features = []  # projected features (B, N, 768)
        self.slot_features = []  # slot representations (B, S, D_slot)
        self.slot_masks = []  # slot masks (B, S, N)
        self.point_coords = []  # point coordinates (B, N, 3)
        self.point_labels = []  # ground truth labels (B, N)
        self.sample_names = []

    def extract_features(self):
        """提取特征"""
        self.model.eval()
        print(f"\n开始提取特征（最多{self.max_samples}个样本）...")

        with torch.no_grad():
            for i in tqdm(range(min(self.max_samples, len(self.dataset)))):
                try:
                    sample = self.dataset[i]
                    xyz = sample['xyz'].unsqueeze(0).to(self.device)  # (1, N, 3)
                    rgb = sample['rgb'].unsqueeze(0).to(self.device)  # (1, N, 3)
                    labels = sample.get('semantic_label', None)  # (N,)

                    # 前向传播
                    reconstruction, slots, masks, sp_feats_proj, sampled_coords = self.model(xyz, rgb)

                    # 存储特征
                    self.encoder_features.append(sp_feats_proj[0].cpu().numpy())  # (N, 768)
                    self.slot_features.append(slots[0].cpu().numpy())  # (S, D_slot)
                    self.slot_masks.append(masks[0].cpu().numpy())  # (S, N)
                    self.point_coords.append(sampled_coords[0].cpu().numpy())  # (N, 3)

                    # 标签需要映射到采样点
                    if labels is not None:
                        # 使用KD-tree找最近邻标签
                        from scipy.spatial import cKDTree
                        tree = cKDTree(xyz[0].cpu().numpy())
                        _, nn_idx = tree.query(sampled_coords[0].cpu().numpy(), k=1)
                        sampled_labels = labels[nn_idx]
                        self.point_labels.append(sampled_labels.numpy())
                    else:
                        self.point_labels.append(None)

                    self.sample_names.append(sample.get('scene_name', f'sample_{i}'))

                except Exception as e:
                    print(f"\n[警告] 样本 {i} 提取失败: {e}")
                    continue

        print(f"✓ 成功提取 {len(self.encoder_features)} 个样本的特征")

    def analyze_feature_distribution(self, output_dir):
        """分析特征分布"""
        print("\n=== 1. 特征分布分析 ===")
        os.makedirs(output_dir, exist_ok=True)

        # 合并所有样本的特征
        all_encoder_feats = np.concatenate(self.encoder_features, axis=0)  # (N_total, 768)
        all_labels = []
        for labels in self.point_labels:
            if labels is not None:
                all_labels.append(labels)
        if len(all_labels) > 0:
            all_labels = np.concatenate(all_labels, axis=0)  # (N_total,)
            has_labels = True
        else:
            all_labels = None
            has_labels = False

        print(f"  总点数: {all_encoder_feats.shape[0]}")
        print(f"  特征维度: {all_encoder_feats.shape[1]}")
        if has_labels:
            print(f"  类别数: {len(np.unique(all_labels))}")

        # 1.1 PCA降维可视化
        print("\n  → PCA降维...")
        pca = PCA(n_components=min(50, all_encoder_feats.shape[1]))
        feats_pca = pca.fit_transform(all_encoder_feats)

        # 绘制PCA方差解释比例
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(np.cumsum(pca.explained_variance_ratio_))
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('Cumulative Explained Variance')
        axes[0].set_title('PCA Variance Explained')
        axes[0].grid(True)

        # 绘制前2个主成分
        if has_labels:
            unique_labels = np.unique(all_labels)
            for label in unique_labels:
                mask = all_labels == label
                axes[1].scatter(feats_pca[mask, 0], feats_pca[mask, 1],
                              alpha=0.3, s=1, label=f'Class {label}')
            axes[1].legend(markerscale=5, fontsize=8, ncol=2)
        else:
            axes[1].scatter(feats_pca[:, 0], feats_pca[:, 1], alpha=0.3, s=1)
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title('PCA Projection (First 2 Components)')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_pca_analysis.png'), dpi=150)
        plt.close()
        print(f"  ✓ 保存: 01_pca_analysis.png")

        # 1.2 t-SNE降维可视化（采样以加速）
        print("\n  → t-SNE降维（采样5000点）...")
        sample_size = min(5000, all_encoder_feats.shape[0])
        sample_idx = np.random.choice(all_encoder_feats.shape[0], sample_size, replace=False)
        feats_sampled = all_encoder_feats[sample_idx]
        labels_sampled = all_labels[sample_idx] if has_labels else None

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        feats_tsne = tsne.fit_transform(feats_sampled)

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        if has_labels:
            unique_labels = np.unique(labels_sampled)
            for label in unique_labels:
                mask = labels_sampled == label
                ax.scatter(feats_tsne[mask, 0], feats_tsne[mask, 1],
                          alpha=0.5, s=2, label=f'Class {label}')
            ax.legend(markerscale=5, fontsize=8, ncol=2)
        else:
            ax.scatter(feats_tsne[:, 0], feats_tsne[:, 1], alpha=0.5, s=2)
        ax.set_title('t-SNE Projection of Encoder Features')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_tsne_analysis.png'), dpi=150)
        plt.close()
        print(f"  ✓ 保存: 02_tsne_analysis.png")

        # 1.3 特征质量指标
        if has_labels:
            print("\n  → 计算特征质量指标...")
            # Silhouette Score（轮廓系数）
            # 使用PCA降维后的特征计算（加速）
            sil_score = silhouette_score(feats_pca[:, :10], all_labels, sample_size=5000)
            print(f"    Silhouette Score: {sil_score:.4f} ([-1, 1], 越大越好)")

            # 类内距离 vs 类间距离
            intra_class_dists = []
            inter_class_dists = []
            unique_labels = np.unique(all_labels)

            for label in unique_labels:
                mask = all_labels == label
                class_feats = all_encoder_feats[mask]

                # 类内距离：同类别点之间的平均距离
                if class_feats.shape[0] > 1:
                    # 采样以加速
                    sample_size_class = min(500, class_feats.shape[0])
                    sample_idx_class = np.random.choice(class_feats.shape[0], sample_size_class, replace=False)
                    class_feats_sampled = class_feats[sample_idx_class]

                    # 计算pairwise距离
                    dists = np.linalg.norm(
                        class_feats_sampled[:, None, :] - class_feats_sampled[None, :, :],
                        axis=2
                    )
                    intra_dist = dists[np.triu_indices_from(dists, k=1)].mean()
                    intra_class_dists.append(intra_dist)

            # 类间距离：不同类别中心之间的距离
            class_centers = []
            for label in unique_labels:
                mask = all_labels == label
                center = all_encoder_feats[mask].mean(axis=0)
                class_centers.append(center)
            class_centers = np.array(class_centers)  # (n_classes, D)

            for i in range(len(class_centers)):
                for j in range(i + 1, len(class_centers)):
                    inter_dist = np.linalg.norm(class_centers[i] - class_centers[j])
                    inter_class_dists.append(inter_dist)

            intra_mean = np.mean(intra_class_dists)
            inter_mean = np.mean(inter_class_dists)
            separation_ratio = inter_mean / (intra_mean + 1e-8)

            print(f"    类内平均距离: {intra_mean:.4f}")
            print(f"    类间平均距离: {inter_mean:.4f}")
            print(f"    分离比 (类间/类内): {separation_ratio:.4f} (越大越好)")

            # 保存指标
            metrics = {
                'silhouette_score': float(sil_score),
                'intra_class_distance': float(intra_mean),
                'inter_class_distance': float(inter_mean),
                'separation_ratio': float(separation_ratio),
            }
            with open(os.path.join(output_dir, '00_feature_metrics.txt'), 'w') as f:
                f.write("=== Encoder Feature Quality Metrics ===\n\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value:.4f}\n")
            print(f"  ✓ 保存: 00_feature_metrics.txt")

    def analyze_slot_binding(self, output_dir):
        """分析slot绑定情况"""
        print("\n=== 2. Slot绑定分析 ===")
        os.makedirs(output_dir, exist_ok=True)

        # 统计每个样本的slot占用率
        slot_occupancies = []
        for masks in self.slot_masks:  # masks: (S, N)
            # 计算每个slot的平均mask值（占用率）
            occupancy = masks.mean(axis=1)  # (S,)
            slot_occupancies.append(occupancy)

        slot_occupancies = np.array(slot_occupancies)  # (n_samples, S)
        mean_occupancy = slot_occupancies.mean(axis=0)  # (S,)
        std_occupancy = slot_occupancies.std(axis=0)  # (S,)

        # 2.1 绘制slot占用率分布
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # 每个slot的平均占用率
        axes[0].bar(range(len(mean_occupancy)), mean_occupancy, yerr=std_occupancy,
                   alpha=0.7, capsize=5)
        axes[0].set_xlabel('Slot Index')
        axes[0].set_ylabel('Mean Occupancy')
        axes[0].set_title('Slot Occupancy Across All Samples')
        axes[0].axhline(y=1.0 / len(mean_occupancy), color='r', linestyle='--',
                       label=f'Uniform ({1.0/len(mean_occupancy):.3f})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Heatmap: 样本 x Slot
        im = axes[1].imshow(slot_occupancies.T, aspect='auto', cmap='viridis',
                           interpolation='nearest')
        axes[1].set_xlabel('Sample Index')
        axes[1].set_ylabel('Slot Index')
        axes[1].set_title('Slot Occupancy Heatmap (Samples x Slots)')
        plt.colorbar(im, ax=axes[1], label='Occupancy')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '03_slot_occupancy.png'), dpi=150)
        plt.close()
        print(f"  ✓ 保存: 03_slot_occupancy.png")

        # 2.2 分析slot collapse（多个slots关注同一区域）
        print("\n  → 分析slot collapse...")
        slot_overlap_scores = []
        for masks in self.slot_masks:  # masks: (S, N)
            S, N = masks.shape
            # 计算slot之间的overlap（余弦相似度）
            masks_norm = masks / (np.linalg.norm(masks, axis=1, keepdims=True) + 1e-8)
            overlap_matrix = masks_norm @ masks_norm.T  # (S, S)

            # 只取上三角（不包括对角线）
            overlap_scores = overlap_matrix[np.triu_indices(S, k=1)]
            slot_overlap_scores.append(overlap_scores.mean())

        mean_overlap = np.mean(slot_overlap_scores)
        print(f"    平均Slot Overlap: {mean_overlap:.4f} (越小越好，说明slots关注不同区域)")

        # 2.3 分析background vs foreground占用（如果是Two-Stage）
        has_two_stage = self.check_two_stage()
        if has_two_stage:
            print("\n  → 检测到Two-Stage模型，分析背景/前景分离...")
            bg_occupancies = slot_occupancies[:, 0]  # slot 0是背景
            fg_occupancies = slot_occupancies[:, 1:].sum(axis=1)  # 前景slots总和

            print(f"    背景slot平均占用率: {bg_occupancies.mean():.4f}")
            print(f"    前景slots平均占用率: {fg_occupancies.mean():.4f}")

            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            x = np.arange(len(bg_occupancies))
            width = 0.35
            ax.bar(x - width/2, bg_occupancies, width, label='Background (Slot 0)', alpha=0.7)
            ax.bar(x + width/2, fg_occupancies, width, label='Foreground (Slots 1-N)', alpha=0.7)
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Occupancy')
            ax.set_title('Background vs Foreground Occupancy')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '04_bg_fg_separation.png'), dpi=150)
            plt.close()
            print(f"  ✓ 保存: 04_bg_fg_separation.png")

        # 保存slot统计
        with open(os.path.join(output_dir, '05_slot_statistics.txt'), 'w') as f:
            f.write("=== Slot Binding Statistics ===\n\n")
            f.write(f"Number of slots: {len(mean_occupancy)}\n")
            f.write(f"Number of samples: {len(slot_occupancies)}\n\n")
            f.write(f"Mean slot overlap: {mean_overlap:.4f}\n\n")
            f.write("Per-slot mean occupancy:\n")
            for i, occ in enumerate(mean_occupancy):
                f.write(f"  Slot {i}: {occ:.4f} ± {std_occupancy[i]:.4f}\n")

            if has_two_stage:
                f.write(f"\nTwo-Stage Analysis:\n")
                f.write(f"  Background (Slot 0): {bg_occupancies.mean():.4f} ± {bg_occupancies.std():.4f}\n")
                f.write(f"  Foreground (Slots 1-N): {fg_occupancies.mean():.4f} ± {fg_occupancies.std():.4f}\n")
        print(f"  ✓ 保存: 05_slot_statistics.txt")

    def visualize_slot_features(self, output_dir):
        """可视化slot特征分布"""
        print("\n=== 3. Slot特征分析 ===")
        os.makedirs(output_dir, exist_ok=True)

        # 合并所有样本的slot特征
        all_slot_feats = np.concatenate(self.slot_features, axis=0)  # (n_samples*S, D_slot)
        n_samples = len(self.slot_features)
        S = self.slot_features[0].shape[0]
        D_slot = self.slot_features[0].shape[1]

        print(f"  总slot数: {all_slot_feats.shape[0]}")
        print(f"  Slot特征维度: {D_slot}")

        # 3.1 PCA降维
        print("\n  → PCA降维slot特征...")
        pca = PCA(n_components=min(10, D_slot))
        slot_feats_pca = pca.fit_transform(all_slot_feats)

        # 可视化：按slot index着色
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 按slot index着色
        slot_indices = np.tile(np.arange(S), n_samples)  # 每个样本的slot index
        scatter = axes[0].scatter(slot_feats_pca[:, 0], slot_feats_pca[:, 1],
                                 c=slot_indices, cmap='tab20', alpha=0.6, s=20)
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title('Slot Features PCA (Colored by Slot Index)')
        plt.colorbar(scatter, ax=axes[0], label='Slot Index')

        # 按样本着色
        sample_indices = np.repeat(np.arange(n_samples), S)
        scatter = axes[1].scatter(slot_feats_pca[:, 0], slot_feats_pca[:, 1],
                                 c=sample_indices, cmap='viridis', alpha=0.6, s=20)
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].set_title('Slot Features PCA (Colored by Sample Index)')
        plt.colorbar(scatter, ax=axes[1], label='Sample Index')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '06_slot_features_pca.png'), dpi=150)
        plt.close()
        print(f"  ✓ 保存: 06_slot_features_pca.png")

        # 3.2 Slot多样性分析
        print("\n  → 分析slot多样性...")
        # 计算每个样本内slot之间的余弦相似度
        slot_similarities = []
        for slot_feats in self.slot_features:  # (S, D_slot)
            # 归一化
            slot_feats_norm = slot_feats / (np.linalg.norm(slot_feats, axis=1, keepdims=True) + 1e-8)
            # 余弦相似度矩阵
            sim_matrix = slot_feats_norm @ slot_feats_norm.T  # (S, S)
            # 只取上三角
            sim_scores = sim_matrix[np.triu_indices(S, k=1)]
            slot_similarities.append(sim_scores)

        slot_similarities = np.concatenate(slot_similarities)
        mean_similarity = slot_similarities.mean()
        print(f"    Slot特征平均相似度: {mean_similarity:.4f} (越小越好，说明slots学到不同表征)")

        # 可视化相似度分布
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.hist(slot_similarities, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(x=mean_similarity, color='r', linestyle='--',
                  label=f'Mean: {mean_similarity:.4f}')
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Pairwise Slot Feature Similarities')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '07_slot_similarity_distribution.png'), dpi=150)
        plt.close()
        print(f"  ✓ 保存: 07_slot_similarity_distribution.png")

    def check_two_stage(self):
        """检查是否使用Two-Stage DINOSAUR"""
        return hasattr(self.model, 'dinosaur') and self.model.dinosaur.__class__.__name__ == 'TwoStageDINOSAURpp'

    def generate_report(self, output_dir):
        """生成分析报告"""
        print("\n=== 4. 生成诊断报告 ===")
        report_path = os.path.join(output_dir, '00_DIAGNOSIS_REPORT.txt')

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DINOSAUR 特征分析诊断报告\n")
            f.write("=" * 80 + "\n\n")

            f.write("## 1. 模型配置\n")
            f.write(f"  - Backbone: {self.model.__class__.__name__}\n")
            f.write(f"  - Two-Stage: {'是' if self.check_two_stage() else '否'}\n")
            f.write(f"  - 样本数: {len(self.encoder_features)}\n\n")

            f.write("## 2. 问题诊断\n\n")

            # 读取之前保存的指标
            metrics = {}
            if os.path.exists(os.path.join(output_dir, '00_feature_metrics.txt')):
                with open(os.path.join(output_dir, '00_feature_metrics.txt'), 'r') as mf:
                    for line in mf:
                        if ':' in line:
                            key, value = line.split(':', 1)
                            try:
                                metrics[key.strip()] = float(value.strip())
                            except:
                                pass

            # 诊断1: Encoder特征质量
            f.write("### 2.1 Encoder特征质量\n")
            if 'silhouette_score' in metrics:
                sil = metrics['silhouette_score']
                if sil > 0.3:
                    f.write(f"  ✓ Silhouette Score: {sil:.4f} - 特征可分性良好\n")
                elif sil > 0.1:
                    f.write(f"  ⚠ Silhouette Score: {sil:.4f} - 特征可分性一般\n")
                else:
                    f.write(f"  ✗ Silhouette Score: {sil:.4f} - 特征可分性较差\n")

            if 'separation_ratio' in metrics:
                ratio = metrics['separation_ratio']
                if ratio > 2.0:
                    f.write(f"  ✓ 分离比: {ratio:.4f} - 类间距离远大于类内距离\n")
                elif ratio > 1.0:
                    f.write(f"  ⚠ 分离比: {ratio:.4f} - 类间/类内距离接近\n")
                else:
                    f.write(f"  ✗ 分离比: {ratio:.4f} - 类内距离大于类间距离（特征混乱）\n")

            f.write("\n### 2.2 Slot绑定问题\n")

            # 读取slot统计
            if os.path.exists(os.path.join(output_dir, '05_slot_statistics.txt')):
                with open(os.path.join(output_dir, '05_slot_statistics.txt'), 'r') as sf:
                    content = sf.read()
                    if 'Mean slot overlap' in content:
                        import re
                        match = re.search(r'Mean slot overlap: ([\d.]+)', content)
                        if match:
                            overlap = float(match.group(1))
                            if overlap < 0.3:
                                f.write(f"  ✓ Slot Overlap: {overlap:.4f} - Slots关注不同区域\n")
                            elif overlap < 0.5:
                                f.write(f"  ⚠ Slot Overlap: {overlap:.4f} - 部分slots可能collapse\n")
                            else:
                                f.write(f"  ✗ Slot Overlap: {overlap:.4f} - 严重的slot collapse\n")

            if self.check_two_stage():
                f.write("\n### 2.3 Two-Stage分析\n")
                f.write("  → 请查看 04_bg_fg_separation.png 确认背景/前景分离质量\n")
                f.write("  → 如果背景slot占用率过低，可能导致前景slots绑定背景\n")

            f.write("\n## 3. 改进建议\n\n")

            if not self.check_two_stage():
                f.write("  [1] 启用Two-Stage DINOSAUR\n")
                f.write("      → 配置文件中设置 two_stage: true\n")
                f.write("      → 强制前景-背景分离，避免slots浪费在背景上\n\n")

            f.write("  [2] 使用对比学习损失\n")
            f.write("      → 已集成到losses.py中\n")
            f.write("      → 配置文件中设置 contrastive_compact/separate/fg_bg 权重\n\n")

            f.write("  [3] 调整聚类策略\n")
            f.write("      → 启用size和spatial特征辅助聚类\n")
            f.write("      → 如果使用Two-Stage，聚类时排除背景slot\n\n")

            f.write("  [4] 可视化检查\n")
            f.write("      → 查看 02_tsne_analysis.png 确认特征是否按语义聚类\n")
            f.write("      → 查看 03_slot_occupancy.png 确认slot使用是否均衡\n")
            f.write("      → 查看 06_slot_features_pca.png 确认slot表征是否多样\n\n")

        print(f"  ✓ 保存: 00_DIAGNOSIS_REPORT.txt")
        print(f"\n完整报告已生成: {report_path}")


def main():
    parser = argparse.ArgumentParser(description='特征提取和可视化分析')
    parser.add_argument('--config', type=str, required=True, help='训练配置文件')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型checkpoint路径')
    parser.add_argument('--dataset', type=str, default='scannet', choices=['s3dis', 'scannet'],
                       help='数据集类型')
    parser.add_argument('--num_samples', type=int, default=20, help='分析的样本数量')
    parser.add_argument('--output_dir', type=str, default='analysis_results/',
                       help='输出目录')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("DINOSAUR 特征分析")
    print("=" * 80)
    print(f"\n配置文件: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"数据集: {args.dataset}")
    print(f"样本数: {args.num_samples}")
    print(f"输出目录: {args.output_dir}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 创建模型
    backbone = config['model'].get('backbone', 'mask3d').lower()
    print(f"\n加载模型 (backbone={backbone})...")
    if backbone == 'mask3d':
        model = create_mask3d_dinosaur_model(config).to(args.device)
    elif backbone == 'logosp':
        model = create_logosp_dinosaur_model(config).to(args.device)
    elif backbone == 'concerto':
        model = create_concerto_dinosaur_model(config).to(args.device)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    # 加载checkpoint
    print(f"加载checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if 'model_state_dict' in checkpoint:
        model.projector.load_state_dict(checkpoint['model_state_dict']['projector'])
        model.dinosaur.load_state_dict(checkpoint['model_state_dict']['dinosaur'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print("✓ 模型加载完成")

    # 加载数据集
    print(f"\n加载数据集: {args.dataset}...")
    if args.dataset == 's3dis':
        dataset = S3DISDataset(
            root_dir=config['data']['s3dis_root'],
            areas=config['data']['val_areas'],
            max_points=config['data']['max_points']
        )
    elif args.dataset == 'scannet':
        dataset = ScanNetDataset(
            root_dir=config['data']['scannet_root'],
            split='val',
            max_points=config['data']['max_points']
        )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"✓ 数据集加载完成，共 {len(dataset)} 个样本")

    # 创建分析器
    analyzer = FeatureAnalyzer(model, dataset, device=args.device, max_samples=args.num_samples)

    # 提取特征
    analyzer.extract_features()

    # 分析特征分布
    analyzer.analyze_feature_distribution(args.output_dir)

    # 分析slot绑定
    analyzer.analyze_slot_binding(args.output_dir)

    # 可视化slot特征
    analyzer.visualize_slot_features(args.output_dir)

    # 生成报告
    analyzer.generate_report(args.output_dir)

    print("\n" + "=" * 80)
    print("分析完成！")
    print("=" * 80)
    print(f"\n所有结果已保存到: {args.output_dir}")
    print(f"请查看诊断报告: {os.path.join(args.output_dir, '00_DIAGNOSIS_REPORT.txt')}")


if __name__ == '__main__':
    main()
