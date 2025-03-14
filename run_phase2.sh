#!/usr/bin/env bash
# 绕过 source 命令，直接执行 conda 的初始化
eval "$(/home/yida/anaconda3/bin/conda shell.bash hook)"
# 切换到目标环境
conda activate pytorch112
# 训练脚本
echo "运行成功"
echo "tail -f ./Logs/nohup/$(date +%Y-%m%d-%H%M)_time.log"
nohup python -u train_phase2.py --batch_size=4 --dataset='RoadScene_train'  --num_epochs=4000 --img_size 304 400 --size_mode='resize' --learning_rate=0.0001 --scheduler 'True' '800' --phase2_model_id='24-1119-1001' --phase2_ModelPath='./Model/Parameters/24-1119-1001/RegImageFusModel-best.pth' > ./Logs/nohup/$(date +%Y-%m%d-%H%M)_time.log 2>&1 &







