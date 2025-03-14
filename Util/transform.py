"""
Author: YidaChen
Time is: 2023/9/4
this Code: 数据变换类
"""

# 将上级目录添加到系统目录中，可导入上级目录中的类与函数
import sys

import torch
from torchvision import transforms

sys.path.append("../Config")
from Config.config_parser import ConfigManager


class DataTransform:
    def __init__(self, img_size=[256, 256], train_mode='train', size_mode='resize'):
        # 设置变换模式
        self.mode = train_mode
        # 选择采用resize或者随机裁剪
        self.size_mode = size_mode
        # 新增长和宽
        self.image_size_h = img_size[0]
        self.image_size_w = img_size[1]

        # 训练模式
        if self.mode == 'train':
            if self.image_size_h<=0 and self.image_size_w <= 0:
                self.transform = transforms.Compose([
                    # transforms.Resize((img_size, img_size)),
                    # transforms.CenterCrop(self.image_size),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5,), (0.5,))
                ])
            else:
                if size_mode == 'resize':
                    # self.image_size = img_size
                    self.transform = transforms.Compose([
                        transforms.Resize((self.image_size_h, self.image_size_w)),
                        # transforms.Resize(self.image_size_h),
                        # transforms.RandomCrop((img_size, img_size), pad_if_needed=True),
                        # transforms.CenterCrop(self.image_size),
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomVerticalFlip(),
                        # transforms.ColorJitter(brightness=(0.8, 1.2)),

                        transforms.ToTensor(),
                        # transforms.Normalize((0.5,), (0.5,))
                    ])

                elif size_mode == 'crop':
                    # self.image_size = img_size
                    self.transform = transforms.Compose([
                        # transforms.Resize((img_size, img_size)),
                        transforms.RandomCrop((self.image_size_h, self.image_size_w), pad_if_needed=True),
                        # transforms.CenterCrop(self.image_size),
                        # transforms.RandomHorizontalFlip(),
                        # transforms.RandomVerticalFlip(),
                        # transforms.ColorJitter(brightness=(0.8, 1.2)),

                        transforms.ToTensor(),
                        # transforms.Normalize((0.5,), (0.5,))
                    ])



        # 测试模式
        elif self.mode == 'test':
            if img_size == 'original':
                # self.image_size = img_size
                self.transform = transforms.Compose([
                    # transforms.Resize((img_size, img_size)),
                    # transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5,), (0.5,))
                ])
            else:
                # self.image_size = img_size
                self.transform = transforms.Compose([
                    # transforms.Resize((300, 400)),
                    # transforms.Resize(self.image_size_h),
                    transforms.Resize((self.image_size_h, self.image_size_w)),
                    # transforms.CenterCrop(self.image_size),
                    transforms.ToTensor(),
                    # transforms.Normalize((0.5,), (0.5,))
                ])

    def get_transform(self, ir_img, vis_img):
        # 训练模式
        if self.mode == 'train':
            # 保证红外与可见光图像的变换一致
            seed = torch.randint(0, 2 ** 32, (1,)).item()

            # 设置随机数种子，并应用于红外图像
            torch.manual_seed(seed)
            ir_transformed = self.transform(ir_img)

            # 设置同样的随机数种子，并应用于可见光图像
            torch.manual_seed(seed)
            vis_transformed = self.transform(vis_img)
            return ir_transformed, vis_transformed

        # 测试模式
        elif self.mode == 'test':
            ir_transformed = self.transform(ir_img)
            vis_transformed = self.transform(vis_img)
            return ir_transformed, vis_transformed


if __name__ == '__main__':
    # 获取默认参数
    args = ConfigManager(config_path='../Config/config.yaml').args
    img_size = args.img_size
    transform = DataTransform(img_size)
    print(transform.get_transform())
