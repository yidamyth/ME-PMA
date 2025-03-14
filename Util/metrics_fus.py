"""
Author: YidaChen
Time: 2023/9/6
Description: 图像融合评价指标计算方法（仅保留6个指标）
计算的6个指标为：[CE, MI, Qabf, Qcb, Qcv, VIF]
修改后的计算方式完全参考下面整合了14个指标的代码中相应指标的实现。
"""

import os
import numpy as np
import torch
from PIL import Image
from torch.nn.functional import conv2d
import torch.nn.functional as F

# 避免 numpy 运算中的除零和无效运算错误
np.seterr(divide='ignore', invalid='ignore')


class ImageMetrics:
    def __init__(self, ir_path, vis_path, fus_path):
        self.ir_path = ir_path
        self.vis_path = vis_path
        self.fus_path = fus_path

    def get_image_metrics(self):
        # 读取图像并转换为灰度图
        ir_img = Image.open(self.ir_path).convert('L')
        vis_img = Image.open(self.vis_path).convert('L')
        fus_img = Image.open(self.fus_path).convert('L')

        # 转换为 numpy 数组（float64 类型）
        ir_img = np.array(ir_img).astype(np.float64)
        vis_img = np.array(vis_img).astype(np.float64)
        fus_img = np.array(fus_img).astype(np.float64)

        MI  = self.mutinf(fus_img, ir_img, vis_img)
        CE  = self.cross_entropy(fus_img, ir_img, vis_img)
        VIF  = self.vif(fus_img, ir_img, vis_img)
        Qabf = self.qabf(fus_img, ir_img, vis_img)
        Qcb  = self.qcb(fus_img, ir_img, vis_img)
        Qcv  = self.qcv(fus_img, ir_img, vis_img)

        return [MI, CE, VIF, Qabf, Qcb, Qcv]

    # ========================= 交叉熵 ========================= #
    def _cross_entropy(self, img1, fused):
        P1 = np.histogram(img1.flatten(), range(0, 257), density=True)[0]
        P2 = np.histogram(fused.flatten(), range(0, 257), density=True)[0]

        result = 0
        for k in range(256):
            if P1[k] != 0 and P2[k] != 0:
                result += P1[k] * np.log2(P1[k] / P2[k])

        return result



    def cross_entropy(self, image_F, image_A, image_B):
            cross_entropy_VI = self._cross_entropy(image_A, image_F)
            cross_entropy_IR = self._cross_entropy(image_B, image_F)
            return (cross_entropy_VI + cross_entropy_IR) / 2.0

    # ========================= 互信息 ========================= #
    def _mutinf(self, a, b):
        a = a.astype(np.float64)
        b = b.astype(np.float64)
        M, N = a.shape
        hab = np.zeros((256, 256))
        ha = np.zeros(256)
        hb = np.zeros(256)
        # 归一化至 [0,255]
        if a.max() != a.min():
            a = ((a - a.min()) / (a.max() - a.min()) * 255).astype(np.uint8)
        else:
            a = np.zeros((M, N), dtype=np.uint8)
        if b.max() != b.min():
            b = ((b - b.min()) / (b.max() - b.min()) * 255).astype(np.uint8)
        else:
            b = np.zeros((M, N), dtype=np.uint8)
        a = np.clip(a, 0, 255)
        b = np.clip(b, 0, 255)
        for i in range(M):
            for j in range(N):
                index_x = a[i, j]
                index_y = b[i, j]
                hab[index_x, index_y] += 1
                ha[index_x] += 1
                hb[index_y] += 1
        hsum = np.sum(hab)
        mask = hab != 0
        p_joint = hab[mask] / hsum
        Hab = -np.sum(p_joint * np.log2(p_joint))
        hsum_a = np.sum(ha)
        mask_a = ha != 0
        p_a = ha[mask_a] / hsum_a
        Ha = -np.sum(p_a * np.log2(p_a))
        hsum_b = np.sum(hb)
        mask_b = hb != 0
        p_b = hb[mask_b] / hsum_b
        Hb = -np.sum(p_b * np.log2(p_b))
        return Ha + Hb - Hab

    def mutinf(self, image_F, image_A, image_B):
        mi_A = self._mutinf(image_A, image_F)
        mi_B = self._mutinf(image_B, image_F)
        return mi_A + mi_B

    # ========================= Qabf ========================= #
    def _filter(self, win_size, sigma, dtype, device):
        coords = torch.arange(win_size, dtype=dtype, device=device) - (win_size - 1) / 2
        g = coords ** 2
        g = torch.exp(-(g.unsqueeze(0) + g.unsqueeze(1)) / (2.0 * sigma ** 2))
        g /= torch.sum(g)
        return g

    def compute_gradients_and_orientations(self, image, h1, h3):
        gx = conv2d(image, h3, padding=1)
        gy = conv2d(image, h1, padding=1)
        grad = torch.sqrt(gx ** 2 + gy ** 2)
        angle = torch.atan2(gy, gx)
        angle[torch.isnan(angle)] = np.pi / 2
        return grad, angle

    def compute_quality(self, g1, a1, g2, a2, gF, aF, Tg, Ta, kg, ka, Dg, Da):
        G = torch.where(g1 > gF, gF / g1, torch.where(g1 == gF, gF, g1 / gF))
        A = 1 - torch.abs(a1 - aF) / (np.pi / 2)
        Qg = Tg / (1 + torch.exp(kg * (G - Dg)))
        Qa = Ta / (1 + torch.exp(ka * (A - Da)))
        return Qg * Qa

    def qabf(self, image_F, image_A, image_B):
        pA = torch.from_numpy(image_A).float().unsqueeze(0).unsqueeze(0)
        pB = torch.from_numpy(image_B).float().unsqueeze(0).unsqueeze(0)
        pF = torch.from_numpy(image_F).float().unsqueeze(0).unsqueeze(0)
        # 定义梯度滤波器
        h1 = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        h2 = torch.tensor([[0, 1, 2],
                           [-1, 0, 1],
                           [-2, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        h3 = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        gA, aA = self.compute_gradients_and_orientations(pA, h1, h3)
        gB, aB = self.compute_gradients_and_orientations(pB, h1, h3)
        gF, aF = self.compute_gradients_and_orientations(pF, h1, h3)
        # 参数设定（与参考代码一致）
        L = 1
        Tg, Ta = 0.9994, 0.9879
        kg, ka = -15, -22
        Dg, Da = 0.5, 0.8
        QAF = self.compute_quality(gA, aA, gB, aB, gF, aF, Tg, Ta, kg, ka, Dg, Da)
        QBF = self.compute_quality(gB, aB, gA, aA, gF, aF, Tg, Ta, kg, ka, Dg, Da)
        wA = gA ** L
        wB = gB ** L
        deno = torch.sum(wA + wB)
        nume = torch.sum(QAF * wA + QBF * wB)
        return (nume / deno).item()

    # ========================= Qcb ========================= #
    def qcb(self, image_F, image_A, image_B):
        image_A = image_A.astype(np.float64)
        image_B = image_B.astype(np.float64)
        image_F = image_F.astype(np.float64)
        image_A = (
            (image_A - image_A.min()) / (image_A.max() - image_A.min())
            if image_A.max() != image_A.min()
            else image_A
        )
        image_A = np.round(image_A * 255).astype(np.uint8)
        image_B = (
            (image_B - image_B.min()) / (image_B.max() - image_B.min())
            if image_B.max() != image_B.min()
            else image_B
        )
        image_B = np.round(image_B * 255).astype(np.uint8)
        image_F = (
            (image_F - image_F.min()) / (image_F.max() - image_F.min())
            if image_F.max() != image_F.min()
            else image_F
        )
        image_F = np.round(image_F * 255).astype(np.uint8)

        f0 = 15.3870
        f1 = 1.3456
        a = 0.7622
        k = 1
        h = 1
        p = 3
        q = 2
        Z = 0.0001

        M, N = image_A.shape

        # Use the correct meshgrid for frequency space
        u, v = np.meshgrid(np.fft.fftfreq(N, 0.5), np.fft.fftfreq(M, 0.5))
        u *= N / 30
        v *= M / 30
        r = np.sqrt(u ** 2 + v ** 2)
        Sd = np.exp(-((r / f0) ** 2)) - a * np.exp(-((r / f1) ** 2))

        # Ensure Sd matches the shape of the images
        Sd = Sd[:M, :N]  # This should ensure proper matching

        # Fourier Transform
        fused1 = np.fft.ifft2(np.fft.fft2(image_A) * Sd).real
        fused2 = np.fft.ifft2(np.fft.fft2(image_B) * Sd).real
        ffused = np.fft.ifft2(np.fft.fft2(image_F) * Sd).real

        x = np.linspace(-15, 15, 31)
        y = np.linspace(-15, 15, 31)
        X, Y = np.meshgrid(x, y)
        sigma = 2
        G1 = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)
        sigma = 4
        G2 = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma ** 2)

        G1 = torch.from_numpy(G1).float().unsqueeze(0).unsqueeze(0)
        G2 = torch.from_numpy(G2).float().unsqueeze(0).unsqueeze(0)
        fused1 = torch.from_numpy(fused1).float().unsqueeze(0).unsqueeze(0)
        fused2 = torch.from_numpy(fused2).float().unsqueeze(0).unsqueeze(0)
        ffused = torch.from_numpy(ffused).float().unsqueeze(0).unsqueeze(0)

        buff = conv2d(fused1, G1, padding=15)
        buff1 = conv2d(fused1, G2, padding=15)
        contrast_value = buff / buff1 - 1
        contrast_value = torch.abs(contrast_value)
        C1P = (k * (contrast_value ** p)) / (h * (contrast_value ** q) + Z)
        buff = conv2d(fused2, G1, padding=15)
        buff1 = conv2d(fused2, G2, padding=15)
        contrast_value = buff / buff1 - 1
        contrast_value = torch.abs(contrast_value)
        C2P = (k * (contrast_value ** p)) / (h * (contrast_value ** q) + Z)
        buff = conv2d(ffused, G1, padding=15)
        buff1 = conv2d(ffused, G2, padding=15)
        contrast_value = buff / buff1 - 1
        contrast_value = torch.abs(contrast_value)
        CfP = (k * (contrast_value ** p)) / (h * (contrast_value ** q) + Z)

        mask = C1P < CfP
        Q1F = CfP / C1P
        Q1F[mask] = (C1P / CfP)[mask]
        mask = C2P < CfP
        Q2F = CfP / C2P
        Q2F[mask] = (C2P / CfP)[mask]

        ramda1 = (C1P ** 2) / (C1P ** 2 + C2P ** 2)
        ramda2 = (C2P ** 2) / (C1P ** 2 + C2P ** 2)

        Q = ramda1 * Q1F + ramda2 * Q2F
        return Q.mean().item()

    # ========================= Qcv ========================= #
    def qcv(self, image_F, image_A, image_B):
        alpha_c = 1
        alpha_s = 0.685
        f_c = 97.3227
        f_s = 12.1653
        window_size = 16
        alpha = 5

        def normalize(image):
            if image.max() != image.min():
                image = (image - image.min()) / (image.max() - image.min())
            return np.round(image * 255)

        image_A = normalize(image_A)
        image_B = normalize(image_B)
        image_F = normalize(image_F)

        h1 = torch.tensor([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        h3 = torch.tensor([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        pA = torch.from_numpy(image_A).float().unsqueeze(0).unsqueeze(0)
        pB = torch.from_numpy(image_B).float().unsqueeze(0).unsqueeze(0)

        img1X = conv2d(pA, h3, padding=1)
        img1Y = conv2d(pA, h1, padding=1)
        im1G = torch.sqrt(img1X ** 2 + img1Y ** 2)

        img2X = conv2d(pB, h3, padding=1)
        img2Y = conv2d(pB, h1, padding=1)
        im2G = torch.sqrt(img2X ** 2 + img2Y ** 2)

        M, N = image_A.shape
        kernel = torch.ones(1, 1, window_size, window_size, dtype=im1G.dtype)
        ramda1 = conv2d(im1G ** alpha, kernel, stride=window_size)
        ramda2 = conv2d(im2G ** alpha, kernel, stride=window_size)

        f1 = image_A - image_F
        f2 = image_B - image_F

        u, v = np.meshgrid(np.fft.fftfreq(N, 0.5), np.fft.fftfreq(M, 0.5))
        u *= N / 8
        v *= M / 8
        r = np.sqrt(u ** 2 + v ** 2)
        theta_m = 2.6 * (0.0192 + 0.144 * r) * np.exp(-((0.144 * r) ** 1.1))

        Df1 = np.fft.ifft2(np.fft.fft2(f1) * theta_m).real
        Df2 = np.fft.ifft2(np.fft.fft2(f2) * theta_m).real

        Df1 = torch.from_numpy(Df1).float().unsqueeze(0).unsqueeze(0)
        Df2 = torch.from_numpy(Df2).float().unsqueeze(0).unsqueeze(0)

        kernel_avg = torch.ones(1, 1, window_size, window_size, dtype=Df1.dtype) / (window_size ** 2)
        D1 = conv2d(Df1 ** 2, kernel_avg, stride=window_size)
        D2 = conv2d(Df2 ** 2, kernel_avg, stride=window_size)

        Q = torch.sum(ramda1 * D1 + ramda2 * D2) / torch.sum(ramda1 + ramda2)
        return Q.item()

    # ========================= VIF ========================= #
    def getvif(self, preds, target, sigma_n_sq=2.0):
        preds = torch.from_numpy(preds).float()
        target = torch.from_numpy(target).float()
        dtype = preds.dtype
        device = preds.device

        preds = preds.unsqueeze(0).unsqueeze(0)
        target = target.unsqueeze(0).unsqueeze(0)
        eps = torch.tensor(1e-10, dtype=dtype, device=device)
        sigma_n_sq = torch.tensor(sigma_n_sq, dtype=dtype, device=device)
        preds_vif = torch.zeros(1, dtype=dtype, device=device)
        target_vif = torch.zeros(1, dtype=dtype, device=device)
        for scale in range(4):
            n = 2.0 ** (4 - scale) + 1
            kernel = self._filter(n, n / 5, dtype=dtype, device=device)[None, None, :]
            if scale > 0:
                target = conv2d(target.float(), kernel)[:, :, ::2, ::2]
                preds = conv2d(preds.float(), kernel)[:, :, ::2, ::2]
            mu_target = conv2d(target, kernel)
            mu_preds = conv2d(preds, kernel)
            mu_target_sq = mu_target ** 2
            mu_preds_sq = mu_preds ** 2
            mu_target_preds = mu_target * mu_preds
            if scale == 0:
                target = target.byte()
                preds = preds.byte()
            sigma_target_sq = torch.clamp(conv2d((target ** 2).float(), kernel) - mu_target_sq, min=0.0)
            sigma_preds_sq = torch.clamp(conv2d((preds ** 2).float(), kernel) - mu_preds_sq, min=0.0)
            sigma_target_preds = conv2d((target * preds).float(), kernel) - mu_target_preds
            g = sigma_target_preds / (sigma_target_sq + eps)
            sigma_v_sq = sigma_preds_sq - g * sigma_target_preds
            mask = sigma_target_sq < eps
            g[mask] = 0
            sigma_v_sq[mask] = sigma_preds_sq[mask]
            sigma_target_sq[mask] = 0
            mask = sigma_preds_sq < eps
            g[mask] = 0
            sigma_v_sq[mask] = 0
            mask = g < 0
            sigma_v_sq[mask] = sigma_preds_sq[mask]
            g[mask] = 0
            sigma_v_sq = torch.clamp(sigma_v_sq, min=eps)
            preds_vif_scale = torch.log10(1.0 + (g ** 2.0) * sigma_target_sq / (sigma_v_sq + sigma_n_sq))
            preds_vif = preds_vif + torch.sum(preds_vif_scale, dim=[1, 2, 3])
            target_vif = target_vif + torch.sum(torch.log10(1.0 + sigma_target_sq / sigma_n_sq), dim=[1, 2, 3])
        vif = preds_vif / target_vif
        return 1.0 if torch.isnan(vif) else vif.item()

    def vif(self, image_F, image_A, image_B):
        vif_A = self.getvif(image_F, image_A)
        vif_B = self.getvif(image_F, image_B)
        return vif_A + vif_B


if __name__ == "__main__":
    # 数据集目录（包含 ir、vis 和 fus 图像）
    test_path = './DataSet/IVIF/RoadScene/RoadS_test'
    ir_file = os.path.join(test_path, 'ir')
    vis_file = os.path.join(test_path, 'vis')
    # 融合图像
    fus_file = os.path.join(test_path, 'Results', 'Ours', 'fus_rgb')
    file_list = os.listdir(fus_file)
    total = []
    for img_name in file_list:
        if img_name.lower().endswith(('.png', 'jpg', 'jpeg')):
            ir_path = os.path.join(ir_file, img_name)
            vis_path = os.path.join(vis_file, img_name)
            fus_path = os.path.join(fus_file, img_name)
            metric = ImageMetrics(ir_path, vis_path, fus_path)
            fusion_metrics = metric.get_image_metrics()
            print("MI: {}\nCE: {}\nVIF: {}\nQabf: {}\nQcb: {}\nQcv: {}\n".format(*fusion_metrics))
            total.append(fusion_metrics)
    # 计算平均指标
    avg_metrics = np.mean(np.array(total), axis=0)
    print("平均指标为：")
    print("MI: {}\nCE: {}\nVIF: {}\nQabf: {}\nQcb: {}\nQcv: {}\n".format(*avg_metrics))