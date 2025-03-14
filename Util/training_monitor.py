"""
Author: YidaChen
Time is: 2023/9/24
this Code:  封装类，把损失和模型保存等封装起来在调用
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.utils as vutils
from torchvision.utils import save_image

from Util.metrics import ImageMetrics


class Tracker:
    def __init__(self):
        self.loss_values = []

    def add_loss(self, loss):
        self.loss_values.append(loss.detach().cpu().numpy()) # loss.item()

    def plot_loss(self, save_path, now_epoch, dataset, best_epoch):
        """
        :param save_path:
        :param now_epoch: 当前的 epoch
        :param best_epoch: 最好的epoch
        :return:
        """
        save_path = './Figures/ModelResult/' + save_path
        plt.plot(self.loss_values)
        plt.title(f'{dataset} now={now_epoch} best={best_epoch}e Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.savefig(save_path)
        plt.close()

    def save_model(self, model, model_path='model.pth'):
        """
        保存模型
        :param model:
        :param model_path:
        :return:
        """
        # model_path = './Model/Parameters/' + model_path
        model_path = model_path
        # 仅保存模型参数
        # torch.save(model.state_dict(), model_path)
        # 加载： ->
        # model = RegImageFusModel(registration=True, use_bn=True)
        # model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

        # 保存整个模型
        torch.save(model, model_path)
        # 加载 -> : model = torch.load(args.model_path, map_location=device)

    def save_fus(self, fus_img, img_path):
        """
        保存融合图像
        :param fus_img:
        :param img_path:
        :return:
        """
        # 保存成2行2列
        grid_images = vutils.make_grid(fus_img[:25, :, :, :], nrow=5)
        img_path = './Figures/ModelResult/' + img_path
        save_image(grid_images, img_path)

    def get_metrics(self, ir_img, vis_img, fus_img):
        """
        测试图像的batch为1, 维度为[1, ,1, h, w] 的tensor
        :param ir_img:
        :param vis_img:
        :param fus_img:
        :return:
        """
        # 压缩维度-> (h,w) * 255 截断然后转换成uint8
        ir_img = (ir_img.squeeze() * 255).clamp(0, 255)
        vis_img = (vis_img.squeeze() * 255).clamp(0, 255)
        fus_img = (fus_img.squeeze() * 255).clamp(0, 255)
        # to numpy 以float64避免过多损失
        ir_img = ir_img.cpu().numpy().astype(np.float64)
        vis_img = vis_img.cpu().numpy().astype(np.float64)
        fus_img = fus_img.cpu().numpy().astype(np.float64)
        # 初始化指标计算类
        metrics = ImageMetrics()
        EN = metrics.calculate_EN(fus_img)
        MI = metrics.calculate_MI(ir_img, vis_img, fus_img)
        SF = metrics.calculate_SF(fus_img)
        AG = metrics.calculate_AG(fus_img)
        SD = metrics.calculate_SD(fus_img)
        PSNR = metrics.calculate_PSNR(ir_img, vis_img, fus_img)
        MSE = metrics.calculate_MSE(ir_img, vis_img, fus_img)
        VIF = metrics.calculate_VIF(ir_img, vis_img, fus_img)
        CC = metrics.calculate_CC(ir_img, vis_img, fus_img)
        SCD = metrics.calculate_SCD(ir_img, vis_img, fus_img)
        Qabf = metrics.calculate_Qabf(ir_img, vis_img, fus_img)
        Nabf = metrics.calculate_get_Nabf(ir_img, vis_img, fus_img)
        SSIM = metrics.calculate_SSIM(ir_img, vis_img, fus_img)
        # 返回字典， 访问指定指标即可
        return {'EN': [EN], 'MI': [MI], 'SF': [SF], 'AG': [AG], 'SD': [SD], 'PSNR': [PSNR], 'MSE': [MSE], 'VIF': [VIF], 'CC': [CC], 'SCD': [SCD], 'Qabf': [Qabf], 'Nabf': [Nabf], 'SSIM': [SSIM]}
