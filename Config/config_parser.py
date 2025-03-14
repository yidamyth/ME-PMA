"""
Author: YidaChen
Time is: 2023/9/5
this Code: 参数设置
"""
# config_parser.py
import argparse
import time
import yaml


class ConfigManager:
    def __init__(self, config_path='./config.yaml'):
        self.config_path = config_path
        self.load_config()
        self.args = None
        self.load_config()

    def load_config(self):
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', type=str, default=config.get('dataset'),
                            choices=['RoadScene_train'], help='数据集名称')
        parser.add_argument('--testset', type=str, default=config.get('testset'),
                            choices=['Test'], help='数据集名称')

        parser.add_argument('--num_epochs', type=int, default=config.get('num_epochs'), help='模型训练epoch次数')
        parser.add_argument('--batch_size', type=int, default=config.get('batch_size'), help='模型训练的Batch_size')

        parser.add_argument('--num_workers', type=int, default=config.get('num_workers'),
                            help='训练的多线程数num_workers')

        parser.add_argument('--learning_rate', type=float, default=config.get('learning_rate'),
                            help='模型训练的学习率learning_rate')

        parser.add_argument('--weight_decay', type=float, default=config.get('weight_decay'),
                            help='模型训练的正则化设置，避免模型过拟合以及梯度消失、')
        parser.add_argument('--seed', type=int, default=config.get('seed'), help='固定随机数种子，便于复现实验')
        parser.add_argument('--id', type=str, default=time.strftime('%y-%m%d-%H%M'), help='模型的唯一标识符')
        parser.add_argument('--img_size', nargs='+', type=int, default=config.get('img_size'), help='transform的图像大小(h,w)')
        parser.add_argument('--loss_weight', nargs='+', type=float, default=config.get('loss_weight'),
                            help='损失函数的权重系数，默认均为 1')
        parser.add_argument('--opt_type', type=str, default=config.get('opt_type'), choices=['Adam', 'RMSprop'],
                            help='设置优化器类型')
        parser.add_argument('--scheduler', nargs='+', default=config.get('scheduler'), help="['True', '30'] 放入字符串，是否使用学习率衰减策略，每隔多少轮学习率下降0.1")
        parser.add_argument('--img_mode', type=str, default=config.get('img_mode'), choices=['gray', 'rgb'],
                            help='设置加载图像模型，如果为 gray 就 ir 和 vi 转到灰度图操作，如果为 rgb 就把 vis 图像转到 YCrCb 通道基于 Y 通道操作；最后实现一个后处理算法融合 CrCb')
        parser.add_argument('--registration', action='store_false', default=config.get('registration'),
                            help='默认为True,配准模式：输入对齐的图像对， 调用--registration关闭为False')
        parser.add_argument('--use_bn', action='store_false', default=config.get('use_bn'),
                            help='是否使用 BN 层, 默认为True, 调用--use_bn关闭为False')
        parser.add_argument('--num_step', type=int, default=config.get('num_step'),
                            help='每隔多少轮在模型上测试一次，边训练边观测测试结果，默认为1')
        parser.add_argument('--train_paths', type=str, default=config.get('train_paths'),
                            help='训练集路径 key:value train_paths:数据集名称')
        # 第二阶段训练参数
        parser.add_argument('--phase2_ModelPath', type=str, default=config.get('phase2_ModelPath'),
                            help='训练第二阶段加载，已经完成训练的模型参数路径')
        parser.add_argument('--phase2_model_id', type=str, default=config.get('phase2_model_id'),
                            help='训练唯一标识符，与第一阶段相同；保存模型在同一个文件夹下')


        parser.add_argument('--size_mode', type=str, default=config.get('size_mode'), choices=['crop', 'resize'],
                            help='设置加载图像的方式，如果为 crop 就 在原图上进行随机裁剪，如果为 resize 就对原图进行rezie；')
        args = parser.parse_args()
        self.args = args


if __name__ == '__main__':
    config = ConfigManager()
    print(config.args)
    args = ConfigManager(config_path='./config.yaml').args

