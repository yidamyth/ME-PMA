"""
Author: YidaChen
Time is: 2024/2/18
this Code: 输入红外/可见光的融合特征用于特征重建 -> 对应融合图像的梯度信息 目的是为了第二阶段用于变形场的预测
"""
import torch
import torch.nn as nn


# 基于实现编解码器
class Encoder_Decoder_ReGradient(nn.Module):
    def __init__(self, in_channels, use_bn=True):
        super(Encoder_Decoder_ReGradient, self).__init__()
        # c = 32 16 8 1
        c = [in_channels, in_channels // 2, in_channels // 4, 1]

        self.layer = nn.Sequential(
            # 16 -> 8
            nn.Conv2d(c[0], c[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c[0]) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c[0], c[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(c[1]) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 -> 4
            nn.Conv2d(c[1], c[1], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c[1]) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c[1], c[2], kernel_size=1, stride=1),
            nn.BatchNorm2d(c[2]) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 -> 1
            nn.Conv2d(c[2], c[2], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(c[2]) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(c[2], c[3], kernel_size=1, stride=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layer(x)
        return x


if __name__ == '__main__':
    model = Encoder_Decoder_ReGradient(in_channels=16, use_bn=True)
    inputs = torch.rand(10, 16, 256, 256)

    print(model)
    outputs = model(inputs)
    print(outputs.shape)
