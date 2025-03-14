#!/usr/bin/env bash
# 绕过 source 命令，直接执行 conda 的初始化
eval "$(/home/yida/anaconda3/bin/conda shell.bash hook)"
# 切换到目标环境
conda activate pytorch112
# 训练脚本
echo "运行开始"

# 定义一个函数来运行训练命令并保存日志
run_training() {
    batch_size=$1
    loss_weight=$2
    num_epochs=$3
    img_size1=$4
    img_size2=$5
    size_mode=$6
    log_file="./Logs/nohup/$(date +%Y-%m%d-%H%M)_time.log"
    echo "开始训练：batch_size=${batch_size}, loss_weight=${loss_weight}, num_epochs=${num_epochs}"
    echo "日志文件：tail -f ${log_file}"
    nohup python -u train.py \
        --batch_size=${batch_size} \
        --dataset='RoadScene_train' \
        --loss_weight ${loss_weight} \
        --num_epochs=${num_epochs} \
        --img_size ${img_size1} ${img_size2} \
        --size_mode=${size_mode} \
        --learning_rate=0.01 \
        --scheduler 'True' '600' \
        > ${log_file} 2>&1
}

# 依次执行训练命令
run_training 10 "1 15 10 4" 2000 128 128 'resize'
run_training 10 "1 20 10 4" 2000 128 128 'resize'

echo "所有训练已提交"