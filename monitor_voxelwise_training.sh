#!/bin/bash
# 监控体素级训练进度

echo "🔍 实时监控体素级训练（按Ctrl+C退出）"
echo "=================================="
echo ""

while true; do
    clear
    echo "🚀 体素级异常检测训练监控"
    echo "=================================="
    echo ""
    echo "📊 已完成的Epoch摘要:"
    echo "---"
    tail -1000 voxelwise_training_v3.log | grep "📊 Epoch" -A 5 | tail -30
    echo ""
    echo "---"
    echo "⏰ 最后更新时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "🔄 每10秒自动刷新..."
    sleep 10
done

