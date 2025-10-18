#!/bin/bash
# 完整训练启动脚本

echo "============================================================"
echo "🚀 开始完整的AnoVox异常检测训练"
echo "============================================================"
echo ""
echo "📊 训练配置:"
echo "   - 模型: MileAnomalyDetection (跨模态注意力融合)"
echo "   - 数据集: AnoVox (4200样本)"
echo "   - Epochs: 50"
echo "   - Batch Size: 4"
echo "   - GPU: RTX 4090"
echo "   - 预计时间: 8-12小时"
echo ""
echo "💡 监控方式:"
echo "   1. 实时日志: tail -f full_training.log"
echo "   2. TensorBoard: tensorboard --logdir lightning_logs"
echo "   3. GPU监控: watch -n 1 nvidia-smi"
echo ""
echo "============================================================"
echo ""

# 开始训练
python train_anovox.py \
    --config muvo/configs/anovox_training.yml \
    --batch-size 4 \
    --epochs 50 \
    --gpus 1 \
    2>&1 | tee full_training.log

echo ""
echo "============================================================"
echo "🎉 训练完成！"
echo "============================================================"
