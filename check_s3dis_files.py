"""
检查S3DIS数据集中的损坏文件
"""
import os
import glob
import numpy as np
from tqdm import tqdm

def check_file(file_path):
    """检查单个文件是否可以正常加载"""
    try:
        data = np.loadtxt(file_path, dtype=np.float32)
        return True, None
    except Exception as e:
        return False, str(e)

def main():
    s3dis_root = '/home/pbw/data1/3D_PointCloud_Segmentation/PLSG_Net/dataset/S3DIS/Stanford_Large-Scale_Indoor_Spaces_3D_Dataset/Stanford3dDataset_v1.2_Aligned_Version'
    
    # 查找所有房间文件
    pattern = os.path.join(s3dis_root, 'Area_*', '*', '*.txt')
    all_files = glob.glob(pattern)
    
    # 过滤有效文件
    valid_files = []
    for f in all_files:
        if 'ReadMe' in f or 'Annotations' in f:
            continue
        parent_name = os.path.basename(os.path.dirname(f))
        file_name = os.path.basename(f).replace('.txt', '')
        if parent_name == file_name:
            valid_files.append(f)
    
    print(f"总共找到 {len(valid_files)} 个房间文件")
    print("\n开始检查...")
    
    corrupted_files = []
    
    for file_path in tqdm(valid_files):
        is_ok, error = check_file(file_path)
        if not is_ok:
            corrupted_files.append((file_path, error))
    
    print(f"\n检查完成！")
    print(f"损坏文件数: {len(corrupted_files)}")
    
    if corrupted_files:
        print("\n损坏的文件列表:")
        for file_path, error in corrupted_files:
            print(f"\n文件: {file_path}")
            print(f"错误: {error[:100]}...")  # 只显示前100字符
    
    print(f"\n有效文件数: {len(valid_files) - len(corrupted_files)}")
    print(f"损坏文件占比: {len(corrupted_files) / len(valid_files) * 100:.2f}%")

if __name__ == "__main__":
    main()


