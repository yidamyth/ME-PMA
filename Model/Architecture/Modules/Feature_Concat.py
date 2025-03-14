"""
Author: YidaChen
Time is: 2023/10/27
this Code: 从新实现的特征拼接模块，特征拼接后基于公有特征、私有特征、解码特征做自注意力，最后一个卷积层降低维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.swin_transformer import SwinTransformerBlock


# 特征融合模块
class Feature_Concat_Model(nn.Module):
    def __init__(self, in_channels=16, use_bn=True):
        super(Feature_Concat_Model, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(2 * in_channels, 2 * in_channels, 3, 1, 1),
            nn.BatchNorm2d(2 * in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

        )

        self.layer2 = nn.Sequential(
            # 充分融合
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
            # 降维
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, common, private):
        # 融合特征
        x = torch.concat([common, private], dim=1)
        x = self.layer1(x) + x
        x = self.layer2(x) + common + private
        x = self.layer3(x) + x

        return x


if __name__ == '__main__':
    # 初始化模型和输入
    channels = 16  # 假设输入特征的通道数为64
    model = Feature_Concat_Model(in_channels=channels, use_bn=False)
    print(model)

    input1 = torch.rand((1, channels, 256, 256))  # 假设每个输入特征都有形状 (1, 64, 224, 224)
    input2 = torch.rand((1, channels, 256, 256))

    # 使用交叉模态注意力模块
    output = model(input1, input2)
    print("Output feature shape:", output.shape)  # 应该与输入特征有相同的形状
