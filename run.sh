#!/usr/bin/env bash
# 绕过 source 命令，直接执行 conda 的初始化
eval "$(/home/yida/anaconda3/bin/conda shell.bash hook)"
# 切换到目标环境
conda activate pytorch112
# 训练脚本
echo "运行成功"
echo "tail -f ./Logs/nohup/$(date +%Y-%m%d-%H%M)_time.log"
nohup python -u train.py --batch_size=10 --dataset='RoadScene_train' --loss_weight 1 5 20 4 --num_epochs=2000 --img_size 128 128 --size_mode='crop' --learning_rate=0.01 --scheduler 'True' '600' > ./Logs/nohup/$(date +%Y-%m%d-%H%M)_time.log 2>&1 &
# 像素、SSIM、梯度 解码一致性损失 对比损失 融合特征一致性损失 重建融合特征红外与可见光图像损失 重建公有+私有的红外与可见光图像损失
