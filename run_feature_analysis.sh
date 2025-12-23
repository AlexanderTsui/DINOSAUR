#!/bin/bash
# 特征分析运行脚本

echo "================================"
echo "DINOSAUR 特征分析"
echo "================================"
echo ""

# 设置默认参数
CONFIG_FILE="config/config_train_concerto_scannet.yaml"
CHECKPOINT="checkpoints/checkpoints_concerto/concerto_scannet_origin/epoch_200.pth"
DATASET="scannet"
NUM_SAMPLES=20
OUTPUT_DIR="analysis_results/single_stage_analysis"
DEVICE="cuda"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --num_samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "未知参数: $1"
      exit 1
      ;;
  esac
done

# 显示配置
echo "配置信息："
echo "  配置文件: $CONFIG_FILE"
echo "  Checkpoint: $CHECKPOINT"
echo "  数据集: $DATASET"
echo "  样本数: $NUM_SAMPLES"
echo "  输出目录: $OUTPUT_DIR"
echo "  设备: $DEVICE"
echo ""

# 检查文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
  echo "错误: 配置文件不存在: $CONFIG_FILE"
  exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
  echo "错误: Checkpoint文件不存在: $CHECKPOINT"
  exit 1
fi

# 运行分析
echo "开始分析..."
python analyze_features.py \
  --config "$CONFIG_FILE" \
  --checkpoint "$CHECKPOINT" \
  --dataset "$DATASET" \
  --num_samples "$NUM_SAMPLES" \
  --output_dir "$OUTPUT_DIR" \
  --device "$DEVICE"

# 检查退出状态
if [ $? -eq 0 ]; then
  echo ""
  echo "================================"
  echo "分析完成！"
  echo "================================"
  echo ""
  echo "结果保存在: $OUTPUT_DIR"
  echo "诊断报告: $OUTPUT_DIR/00_DIAGNOSIS_REPORT.txt"
  echo ""
  echo "生成的可视化文件："
  echo "  - 01_pca_analysis.png: PCA特征分析"
  echo "  - 02_tsne_analysis.png: t-SNE特征可视化"
  echo "  - 03_slot_occupancy.png: Slot占用率分析"
  echo "  - 04_bg_fg_separation.png: 背景/前景分离（Two-Stage）"
  echo "  - 06_slot_features_pca.png: Slot特征PCA"
  echo "  - 07_slot_similarity_distribution.png: Slot相似度分布"
  echo ""
else
  echo ""
  echo "分析失败，请检查错误信息"
  exit 1
fi
