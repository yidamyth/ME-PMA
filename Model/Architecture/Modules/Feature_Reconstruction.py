# Author: yida
# Time is: 2023/9/3 1:51
# this Code: 特征重建模块，得到融合的可见光图像

import torch
import torch.nn as nn


class FeatureReconstructionModule(nn.Module):
    def __init__(self, in_channels, use_bn=True):
        super(FeatureReconstructionModule, self).__init__()

        self.final_layers1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.final_layers2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels, 1, kernel_size=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, x_ir, x_vis):
        x = self.final_layers1(x) + x
        x = self.final_layers2(x)
        x = self.sigmoid(x)
        return x


if __name__ == '__main__':
    # 初始化模型和输入
    channels = 16  # 假设输入特征的通道数为64
    x = torch.randn(1, 1, 256, 256)
    model = FeatureReconstructionModule(in_channels=16, use_bn=False)
    print(model)

    input_features = torch.rand((1, channels, 256, 256))  # 假设每个输入特征都有形状 (1, 64, 224, 224)

    # 使用特征重建模块
    reconstructed_image = model(input_features, x, x)
    print("Reconstructed image shape:", reconstructed_image.shape)  # 应该为 (1, 3, 224, 224)
