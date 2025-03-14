# Author: yida
# Time is: 2023/9/2 23:37
# this Code: 共享特征提取
import torch
import torch.nn as nn
from torchsummary import summary

# 浅层特征提取模块
class ShallowFeatureExtractor(nn.Module):
    def __init__(self, up_channels, use_bn=True):
        super(ShallowFeatureExtractor, self).__init__()

        # layer1 1->16 Conv-BN-ReLU
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=up_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(up_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=up_channels, out_channels=up_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(up_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # layer2 16->16 Conv-BN-ReLU
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=up_channels, out_channels=up_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(up_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=up_channels, out_channels=up_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(up_channels) if use_bn else nn.Identity(),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        # layer1
        x = self.layer1(x)
        # layer2
        x = self.layer2(x) + x

        return x


if __name__ == '__main__':
    # 初始化模型
    model = ShallowFeatureExtractor(up_channels=16, use_bn=True)
    print(model)

    # 生成ir 和 vis 图像
    visible_img = torch.randn(10, 1, 128, 128)  # 单 通道可见光图像
    infrared_img = torch.randn(10, 1, 128, 128)  # 单 通道红外图像

    # 输出
    vis_features = model(visible_img)
    ir_features = model(infrared_img)

    print("Visible features shape:", vis_features.shape)
    print("Infrared features shape:", vis_features.shape)
    summary(model, (1, 224, 224))
