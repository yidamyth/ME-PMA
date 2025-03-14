# Author: yida
# Time is: 2023/9/2 22:50
# this Code: 端对端的配准与融合网络模型实现
import warnings

import kornia.utils as KU
import torch
import torch.nn as nn
import torch.nn.init as init

from Model.Architecture.Modules.CNN_Based import CNN_Based_Model

from Model.Architecture.Modules.Deformable_Field import DeformableFieldPredictor
# 新增梯度重建模块
from Model.Architecture.Modules.Encoder_Decoder_ReGradient import Encoder_Decoder_ReGradient
# 新增重建模块
from Model.Architecture.Modules.Feature_Concat import Feature_Concat_Model
from Model.Architecture.Modules.Feature_Fusion import Feature_Fusion_Model
from Model.Architecture.Modules.Feature_Reconstruction import FeatureReconstructionModule
from Model.Architecture.Modules.Shallow_Feature import ShallowFeatureExtractor
from Model.Architecture.Modules.Transformer_Based import Transformer_Based_Model
# 测试损失是否能够计算
from Util.loss_function import FusionLoss
import kornia
import torch.nn.functional as F
from torchsummary import summary
import thop

import torch
import torchvision.models as models






device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

warnings.simplefilter(action='ignore', category=Warning)  # 忽略警告


# 主网络
class RegImageFusModel(nn.Module):
    def __init__(self, channels_up=32, registration=True, use_bn=True):
        """

        :param channels_up:
        :param registration:
        :param use_bn:
        :param distil: 新增加一个用于蒸馏的参数，目的是蒸馏第二阶段变形场的encoder
        """
        super(RegImageFusModel, self).__init__()
        # 是否为配准模式, 为否时需要调整解码器输入 、预测变形场作用与变形特征
        self.registration = registration
        # 共享浅层特征  Shallow feature extraction module
        self.shallow = ShallowFeatureExtractor(channels_up, use_bn=use_bn)
        # self.shallow_ir = ShallowFeatureExtractor(channels_up, use_bn=use_bn)
        # self.shallow_vi = ShallowFeatureExtractor(channels_up, use_bn=use_bn)


        # 基于CNN卷积细节特征提取模块
        self.cnn_based = CNN_Based_Model(channels_up, use_bn=use_bn)

        # 基于Transformer的全局特征提取模块
        self.transformer_based = Transformer_Based_Model(channels_up)

        # 特征拼接模块:用不共享的模块来共同处理各个分支
        self.feature_concat_ir = Feature_Concat_Model(channels_up, use_bn=use_bn)
        self.feature_concat_vi = Feature_Concat_Model(channels_up, use_bn=use_bn)

        #
        # 预测变形场模块
        self.deformable_transformation = DeformableFieldPredictor(channels_up, use_bn=use_bn)
        # self.ir_concat_align_decoder = IRALign_Decoder_Model(in_channels=channels_up, use_bn=use_bn)
        # self.fus_feature_align_decoder = IRALign_Decoder_Model(in_channels=channels_up, use_bn=use_bn)

        # 特征融合模块
        self.feature_fusion = Feature_Fusion_Model(in_channels=channels_up, use_bn=use_bn)

        # 特征重建模块
        self.feature_reconstruction = FeatureReconstructionModule(channels_up, use_bn=use_bn)

        # 获取融合图像结果，便于调用访问
        self.interface_fus_images = None


        # 初始化网络模型
        self._initialize_weights()

        # 新增梯度重建模块
        self.reconstruction_gradient_ir = Encoder_Decoder_ReGradient(channels_up, use_bn=use_bn)
        self.reconstruction_gradient_vi = Encoder_Decoder_ReGradient(channels_up, use_bn=use_bn)

    def forward(self, ir_img, vis_img):
        # 浅层特征提取: 共享Shallow feature extraction module
        ir_shallow_features = self.shallow(ir_img)
        vis_shallow_features = self.shallow(vis_img)

        # 卷积细节特征提取
        ir_detail_features = self.cnn_based(ir_shallow_features)
        vis_detail_features = self.cnn_based(vis_shallow_features)

        # 共享的全局特征提取器
        ir_global_features = self.transformer_based(ir_shallow_features)
        vis_global_features = self.transformer_based(vis_shallow_features)

        # 特征拼接模块
        ir_concat_features = self.feature_concat_ir(ir_detail_features, ir_global_features)
        vis_concat_features = self.feature_concat_vi(vis_detail_features, vis_global_features)

        # 输入对齐图像对
        if self.registration:
            # 特征重建
            recon_ir_gradient = self.reconstruction_gradient_ir(ir_concat_features)
            recon_vis_gradient = self.reconstruction_gradient_vi(vis_concat_features)

            # 特征融合
            fusion_features = self.feature_fusion(ir_concat_features, vis_concat_features)

            # 特征重建模块
            fus_images = self.feature_reconstruction(fusion_features, ir_img, vis_img)

            return {"ir_img": ir_img, "vis_img": vis_img, "ir_detail_features": ir_detail_features, "ir_global_features": ir_global_features, "vis_detail_features": vis_detail_features, "vis_global_features": vis_global_features,
                    "ir_concat_features": ir_concat_features,
                    "vis_concat_features": vis_concat_features, "fusion_features": fusion_features, "fus_img": fus_images, 'recon_gradient': {'recon_ir_gradient': recon_ir_gradient, 'recon_vis_gradient': recon_vis_gradient},
                    'ir_shallow_features': ir_shallow_features, 'vis_shallow_features': vis_shallow_features}


        else:
            # TODO:特征对齐，变形场参数预测

            reg_ir_concat_features, flow, x_fix, x_mov = self.deformable_transformation(ir_concat_features=ir_concat_features, vis_concat_features=vis_concat_features)

            reg_recon_ir_gradient = self.reconstruction_gradient_ir(reg_ir_concat_features)
            reg_recon_vis_gradient = self.reconstruction_gradient_vi(vis_concat_features)

            # 特征融合模块
            reg_fusion_features = self.feature_fusion(reg_ir_concat_features, vis_concat_features)

            # 特征重建模块
            reg_fus_images = self.feature_reconstruction(reg_fusion_features, reg_recon_ir_gradient, vis_img)

            return {"ir_img": ir_img, "vis_img": vis_img, "deformabled_features": {"ir_concat_features": reg_ir_concat_features, "fusion_features": reg_fusion_features}, "flow": flow, "unreg_fus_img": reg_fus_images, "reg_recon_ir_gradient": reg_recon_ir_gradient, "distil": {"x_fix": x_fix, "x_mov": x_mov}}

    def _initialize_weights(self):
        """
        使用凯明初始化模型参数
        :return:
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)


if __name__ == '__main__':
    # 生成随机输入图像
    ir_img = torch.randn(1, 1, 256, 256).to(device)  # 单通道红外图像
    vis_img = torch.randn(1, 1, 256, 256).to(device)  # 三通道可见光图像

    # 初始化模型
    model = RegImageFusModel(registration=False, use_bn=True).to(device)

    # 打印模型
    print(model)

    # 训练步骤
    # 输入对齐图像对
    model_dict = model(ir_img=ir_img, vis_img=vis_img)
    # fus_img = model_dict['fus_img']
    # print(fus_img.shape)

    # summary(model, [(1, 256, 256), (1, 256, 256)])
    import thop
    flops, params = thop.profile(model, inputs=(ir_img, vis_img,))
    # 格式化输出
    print("%-10s | %-12s | %-12s" % ("Model", "Params(M)", "FLOPs(G)"))
    print("-----------|--------------|--------------")
    print("%-10s | %-12.3f | %-12.3f" % ("MyModel", params / 1e6, flops / 1e9))



