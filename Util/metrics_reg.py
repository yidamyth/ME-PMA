"""
Author: YidaChen
Time is: 2023/9/6
this Code: 图像配准评价指标计算方法 [MI, MS-SSIM, NCC]
"""
import os.path
import math

import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


# 避免错误提示
np.seterr(divide='ignore', invalid='ignore')


class ImageMetrics:
    def __init__(self, ir_path, ir_reg_path):
        self.ir_path = ir_path
        self.ir_reg_path = ir_reg_path

    def get_image_metrics(self):
        # 打开并转换图像
        ir_img = Image.open(self.ir_path).convert('L')
        ir_reg = Image.open(self.ir_reg_path).convert('L')

        # 转换成 numpy float64
        ir_img = np.array(ir_img).astype(np.float64)
        ir_reg = np.array(ir_reg).astype(np.float64)

        # 仅保留三个核心指标
        MI = self.mutinf(ir_reg, ir_img)
        MS_SSIM = self.calculate_MS_SSIM(ir_reg, ir_img)
        NCC = self.ncc(ir_reg, ir_img)

        return [MI, MS_SSIM, NCC]

    def _mutinf(self, a, b):
        a = a.astype(np.float64)
        b = b.astype(np.float64)

        M, N = a.shape

        # Initialize histograms
        hab = np.zeros((256, 256))
        hab_ = np.zeros((256, 256))
        ha = np.zeros(256)
        hb = np.zeros(256)

        # Normalize
        a = (a - a.min()) / (a.max() - a.min()) if a.max() != a.min() else np.zeros((M, N))
        b = (b - b.min()) / (b.max() - b.min()) if b.max() != b.min() else np.zeros((M, N))

        a = (a * 255).astype(np.uint8)
        b = (b * 255).astype(np.uint8)

        # Ensure values are in the correct range
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)

        # Build histograms
        for i in range(M):
            for j in range(N):
                index_x = a[i, j]
                index_y = b[i, j]
                hab[index_x, index_y] += 1  # Joint histogram
                ha[index_x] += 1  # Histogram for a
                hb[index_y] += 1  # Histogram for b

        # Calculate joint entropy
        hsum = np.sum(hab)
        index = hab != 0
        p = hab[index] / hsum
        Hab = -np.sum(p * np.log2(p))

        # Calculate entropy for a
        hsum = np.sum(ha)
        index = ha != 0
        p = ha[index] / hsum
        Ha = -np.sum(p * np.log2(p))

        # Calculate entropy for b
        hsum = np.sum(hb)
        index = hb != 0
        p = hb[index] / hsum
        Hb = -np.sum(p * np.log2(p))

        # Calculate mutual information
        return Ha + Hb - Hab

    def mutinf(self, ir_reg, ir_img):
        mi = self._mutinf(ir_reg, ir_img)
        return mi

    def calculate_MS_SSIM(self, ir_reg, ir_img):
        ir_reg = torch.from_numpy(ir_reg).float().unsqueeze(0).unsqueeze(0)  # 确保数据类型为 float
        ir_img = torch.from_numpy(ir_img).float().unsqueeze(0).unsqueeze(0)  # 确保数据类型为 float

        ms_ssim_v = ms_ssim(ir_reg, ir_img, data_range=255, size_average=False).detach().cpu().numpy()[0]
        return ms_ssim_v

    def ncc_calculate(self, y_true, y_pred):
        Ii = y_true
        Ji = y_pred

        Ii = torch.from_numpy(Ii).float() / 255.0
        Ii = Ii.unsqueeze(0).unsqueeze(0)

        Ji = torch.from_numpy(Ji).float() / 255.0
        Ji = Ji.unsqueeze(0).unsqueeze(0)

        # get dimension of volume
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        self.win = 9
        win = [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(Ji.device)

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return torch.mean(cc)

    def ncc(self, ir_reg, ir_img):
        ncc = self.ncc_calculate(ir_reg, ir_img)
        return ncc


if __name__ == "__main__":
    # 值得注意的是，我们论文中计算的是3个数据集的均值
    ir_file = './DataSet/IVIF/RoadScene/RoadS_test/ir'
    ir_reg_file = './DataSet/IVIF/RoadScene/RoadS_test/Results/Ours/ir_reg'
    file = os.listdir(ir_reg_file)
    total = []
    for img_name in file:
        print(img_name)
        if img_name.endswith(('.png', 'jpg', 'jpeg')):
            # 获取图像路径
            ir_path = os.path.join(ir_file, img_name)
            ir_reg_path = os.path.join(ir_reg_file, img_name)
            # 计算指标
            metric = ImageMetrics(ir_path, ir_reg_path)
            metrics = metric.get_image_metrics()
            print(
                "MI::{}\nMS_SSIM::{}\nNCC::{}\n".format(
                    *metrics))
            total.append(metrics)
    # 求平均值
    avg_metrics = [0 for i in range(len(total[0]))]
    for i in range(len(total)):
        for j in range(len(total[0])):
            if total[i][j] == None:
                total[i][j] = 0
            avg_metrics[j] += total[i][j]
    avg_metrics = np.array(avg_metrics)
    avg_metrics = avg_metrics / len(total)
    print("平均指标为：")
    print(
        "MI::{}\nMS_SSIM::{}\nNCC::{}\n".format(
            *avg_metrics))
