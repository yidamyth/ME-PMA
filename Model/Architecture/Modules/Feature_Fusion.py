# Author: yida
# Time is: 2023/9/3 0:36
# this Code: 特征融合模块，用于融合红外与可见光特征


import torch
import torch.nn as nn
from timm.models.swin_transformer import SwinTransformerBlock

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 特征融合模块
class Feature_Fusion_Model(nn.Module):
    def __init__(self, in_channels, use_bn=True):
        super(Feature_Fusion_Model, self).__init__()
        self.layer1 = nn.Sequential(
            # 充分融合
            nn.Conv2d(2 * in_channels, 2 * in_channels, 3, 1, 1),
            nn.BatchNorm2d(2 * in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer2 = nn.Sequential(
            # 降维
            nn.Conv2d(2 * in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            # 充分融合
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, ir_f, vis_f):
        x = torch.concat([ir_f, vis_f], dim=1)
        f_max = torch.max(ir_f, vis_f)

        x = self.layer1(x) + x

        x = self.layer2(x) + f_max

        x = self.layer3(x) + ir_f + vis_f

        return x


if __name__ == '__main__':
    # 初始化模型和输入
    channels = 16  # 假设输入特征的通道数为64
    model = Feature_Fusion_Model(in_channels=16, use_bn=False)
    print(model)

    input1 = torch.rand((1, channels, 256, 256))  # 假设每个输入特征都有形状 (1, 64, 224, 224)
    input2 = torch.rand((1, channels, 256, 256))

    # 使用交叉模态注意力模块
    output = model(input1, input2)
    print("Output feature shape:", output.shape)  # 应该与输入特征有相同的形状
