#!/bin/bash
# 实时监控训练进度

echo "========================================"
echo "🔥 MUVO 训练监控面板"
echo "========================================"
echo ""

while true; do
    clear
    echo "========================================"
    echo "🔥 MUVO 训练实时监控"
    echo "========================================"
    echo ""
    
    # 最新的Epoch汇总
    echo "📊 最近的Epoch总结："
    tail -2000 voxelwise_training_final.log | grep "📊 Epoch" -A 5 | tail -30
    
    echo ""
    echo "========================================" 
    echo "🔄 每5秒自动刷新... (Ctrl+C退出)"
    echo "========================================"
    
    sleep 5
done

