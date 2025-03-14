"""
Author: YidaChen
Time is: 2023/11/30
this Code: 第二阶段训练，加载第一阶段训练好的权重，训练变形场模块；写成一个类的形式方便调用；既可以加载参数单独训练也可以放在一起训练，这是等价的。
"""
import copy
import os
import os.path
import time

import numpy as np
import torch
from Config.config_parser import ConfigManager
from Model.Architecture.RegImageFusModel import RegImageFusModel
from Util.dataset import IRVISDataset
from Util.image_transform import ImageTransformer
from Util.random_seed import random_seed
from Util.training_monitor import Tracker
from Util.transform import DataTransform
from setproctitle import setproctitle
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
import math
import torch.nn.functional as F
import kornia
import torch.nn.init as init
import kornia.utils as KU


class SpatialTransformer(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]  # 交换坐标轴 不调整会让图像是倒的
        # new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, torch.clamp(new_locs, min=-1, max=1), align_corners=True, mode=self.mode, padding_mode='reflection')


# 检查错误
# autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class SpatialTransformer_block(nn.Module):

    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode

    def forward(self, src, flow):
        shape = flow.shape[2:]

        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.to(flow.device)

        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]  # 交换坐标轴 不调整会让图像是倒的
        # new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class Grad(nn.Module):
    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, _, y_pred):
        # 计算y方向的梯度差异
        dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        # 计算x方向的梯度差异
        dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx

        # 计算梯度损失
        d = torch.mean(dx) + torch.mean(dy)
        grad = d / 2.0

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad


class NCC:
    def __init__(self, win=9):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [self.win] * ndims

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to(y_pred.device)

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

        return -torch.mean(cc)


def compute_gradient(tensor):
    """
    计算张量的梯度。
    :param tensor: 输入张量，形状为 (b, 2, h, w)
    :return: 在高度和宽度方向上的梯度
    """
    dy = torch.abs(tensor[:, :, 1:, :] - tensor[:, :, :-1, :])
    dx = torch.abs(tensor[:, :, :, 1:] - tensor[:, :, :, :-1])
    return dy, dx


def gradient_smoothness_loss(flow):
    """
    计算梯度平滑损失。
    :param flow: 变形场张量，形状为 (b, 2, h, w)
    :return: 梯度平滑损失
    """
    dy, dx = compute_gradient(flow)
    dy_smooth = torch.mean(dy)
    dx_smooth = torch.mean(dx)
    return dy_smooth + dx_smooth


class Train_Phase2:
    def __init__(self, phase2_ModelPath, model_id):
        self.phase2_path = phase2_ModelPath
        self.model_id = model_id

    def train(self):
        # 预训练模型路径
        phase2_ModelPath = self.phase2_path
        if torch.cuda.is_available():
            # 释放未使用的显存
            torch.cuda.empty_cache()
            # 输出显卡信息
            print(torch.cuda.get_device_name(0))
        model_id = self.model_id
        # 初始化对应保存模型参数文件夹
        model_path = "./Model/Parameters/" + model_id
        # 设置线程名称
        setproctitle(f"yida_{model_id}")
        # 创建 Tensorboard
        # writer = SummaryWriter(f'./Logs/tensorboard/{model_id}')
        # 0. 设计随机数种子
        random_seed(seed=args.seed)
        # 1. 数据加载
        transform = DataTransform(img_size=args.img_size, size_mode=args.size_mode, train_mode='train')

        # 数据集路径 获取字典，然后再传入数据集名称
        train_path = args.train_paths[args.dataset]
        # 自定义dataset
        dataset = IRVISDataset(ir_path=os.path.join(train_path, 'ir'),
                               vis_path=os.path.join(train_path, 'vis'), transform=transform, img_mode=args.img_mode,
                               train_mode='train')
        # dataloader
        dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True,
                                pin_memory=True, drop_last=True)

        # 测试数据集
        # 数据集路径 获取字典，然后再传入数据集名称
        test_path = args.train_paths[args.testset]
        # 自定义dataset
        testset = IRVISDataset(ir_path=os.path.join(test_path, 'ir'),
                               vis_path=os.path.join(test_path, 'vis'), transform=transform, img_mode=args.img_mode,
                               train_mode='train')
        # dataloader
        test_dataloader = DataLoader(testset, batch_size=1, num_workers=args.num_workers, shuffle=True,
                                     pin_memory=True, drop_last=False)

        # 第二阶段训练变形场，初始化模型方法
        # model = RegImageFusModel(registration=args.registration, use_bn=args.use_bn, distil=False)  # args.use)bn
        # model = model.to(device)
        # # 加载预训练的模型权重
        # model.load_state_dict(torch.load(phase2_ModelPath, map_location=device), strict=False)

        model = torch.load(phase2_ModelPath, map_location=device)
        model.registration = args.registration
        model.use_bn = args.use_bn

        model_test = copy.deepcopy(model)  # 应对pytorch的潜在bug，交替切换model.eval和model.train是无法真正切换eval模式 -> 导致融合图像出问题

        # 固定模型参数
        deformable_transformation = model.deformable_transformation  # 初始化的变形场模块
        # feature_fusion = model.feature_fusion  # 初始化的变形场模块
        # feature_reconstruction = model.feature_reconstruction  # 初始化的变形场模块

        # 冻结整个模型的参数
        for name, param in model.named_parameters():
            param.requires_grad = False
        # 2. 将冻结的BN层设置为评估模式
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()  # 设置为评估模式
                module.weight.requires_grad = False
                module.bias.requires_grad = False

        # 解冻 DeformableFieldPredictor 模块的参数
        for name, param in deformable_transformation.named_parameters():
            param.requires_grad = True

        # 变形场的梯度太大了，把它用全零初始化来慢慢学习，我们将遍历它的所有参数，并将它们初始化为零
        for param in deformable_transformation.parameters():
            init.constant_(param, 0)
            nn.init.normal_(param, mean=0, std=0.01)

        for name, param in model.named_parameters():
            print(f"{name} : {param.requires_grad}")

        # 现在进行训练，只有 DeformableFieldPredictor 模块的参数会被更新
        # 3. 定义损失函数和优化器
        if args.opt_type == 'Adam':
            print("选择使用 Adam 优化器...")
            # optimizer = optim.Adam(deform_field_predictor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            optimizer = optim.Adam(
                [
                    {'params': deformable_transformation.parameters()},
                    # {'params': feature_fusion.parameters()},
                    # {'params': feature_reconstruction.parameters()}
                ]
                , lr=args.learning_rate, weight_decay=args.weight_decay)

            if args.scheduler[0] == 'True':
                print('使用学习率更新策略...', args.scheduler)
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)  # T_max=500, 从0-500轮学习率越来越小；500-1000在越来越大，确实应该设置为epoch或者epoch/2
                # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(args.scheduler[1]), gamma=0.1)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
            # 学习率衰减策略
            # scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96)
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        elif args.opt_type == 'RMSprop':
            print("选择使用 RMSprop 优化器...")
            optimizer = optim.RMSprop(deformable_transformation.parameters(), lr=args.learning_rate, alpha=0.99,
                                      eps=1e-08,
                                      weight_decay=args.weight_decay, momentum=0,
                                      centered=False)
            # optimizer = optim.RMSprop(model.parameters(), lr=0.001, alpha=0.99, eps=1e-08,
            #                           weight_decay=0, momentum=0,
            #                           centered=False)
            # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)

        else:
            print("请选择正确的优化器...")

        # 4. 训练循环
        num_epochs = args.num_epochs
        # 开启训练模型
        model.train()
        print("数据准备已完成，正在准备模型训练...")
        #  封装类, 调用各种功能
        tracker = Tracker()
        # 保存最佳loss对应模型参数， 初始化字典
        loss_best = {'loss': np.inf, 'epoch': 2023}
        print_info = []
        # 初始化变换，每个step的变换都是随机的
        image_transformer = ImageTransformer()
        # TODO 待完成
        for epoch in range(num_epochs):
            with tqdm(enumerate(dataloader), total=len(dataloader),
                      desc=f'\33[31m Epoch {epoch + 1}/{num_epochs}') as t:
                loss_total = 0  # 初始化每一轮的总loss
                for step, (ir_img, vis_img) in t:  # 用tqdm来替换原始的输出 ->for step, (ir_img, vis_img) in enumerate(dataloader):
                    optimizer.zero_grad()
                    ir_img, vis_img = ir_img.to(device), vis_img.to(device)
                    # 训练步骤1 -> 得到对齐的融合图像以及特征 -> 用于监督未配准图像融合
                    # # 生成偏移图像对, 对红外图像添加仿射和弹性变换
                    ir_move = image_transformer.transform(ir_img)
                    # with torch.no_grad():
                    #     model_test.eval()
                    #     model_test.registration = True
                    #     model_dict_test = model_test(ir_img=ir_img, vis_img=vis_img)
                    #     fus_img_test, recon_ir_gradient = model_dict_test['fus_img'], model_dict_test['recon_gradient']['recon_ir_gradient']
                    #     # 未对齐直接图像融合结果
                    #     unreg_model_dict = model_test(ir_img=ir_move, vis_img=vis_img)
                    #     unreg_img_test = unreg_model_dict['fus_img']
                    # # 训练步骤2 -> 得到未对齐的融合图像以及特征
                    # model.eval()  # TODO 这儿有待讨论，感觉不用加上eval的 ; 这个必须要有，不然没办法生成融合图像
                    model.registration = False
                    model_dict_move = model(ir_img=ir_move, vis_img=vis_img)
                    flow, reg_fus_img, reg_recon_ir_gradient = model_dict_move['flow'], model_dict_move['unreg_fus_img'], model_dict_move['reg_recon_ir_gradient']

                    # 解决偏移的红外图像
                    STransformer = SpatialTransformer()
                    ir_reg = STransformer(src=ir_move, flow=flow)

                    # 损失1 变形场稀疏损失
                    Grad_Loss = Grad()
                    loss1 = 2 * Grad_Loss(_=None, y_pred=flow)

                    # 损失 2 配准一致性损失
                    NCC_Loss = NCC(win=9)
                    loss2 = NCC_Loss.loss(y_true=ir_img, y_pred=ir_reg)

                    # 损失3 应该在加一个自监督的损失, 用融合图像来进行自监督可能梯度流向不那么明确，试试用这个重建的红外图像来进行自监督
                    # loss3 = F.l1_loss(reg_recon_ir_gradient, recon_ir_gradient)
                    loss3 = F.l1_loss(reg_recon_ir_gradient, ir_img)
                    # loss3 = NCC_Loss.loss(y_true=reg_recon_ir_gradient, y_pred=recon_ir_gradient)

                    loss = loss1 + loss2 + loss3
                    loss.backward()
                    optimizer.step()

                    # 计算一个batch的loss
                    loss_total += loss.item()

                    # 为了使用tqdm，导致模型或者损失函数中间的值无法直接print，因此以info的形式返回，每个epoch结束时输出
                    # print_info = [loss_info["loss1_info"], loss_info["loss2_info"]]
                    # 更新 tqdm 后缀，显示 epoch、step 和 loss；\31m是红色 \33m是蓝色，为了在文本末尾重置格式（回到默认格式），通常会使用 \33[0m
                    t.set_postfix_str(f"MoveStep: {step + 1}/{len(dataloader)}, MoveLoss: {loss.item():.4f}")
                    # 输出训练过程
                    # 保存损失
                    tracker.add_loss(loss)
                    # 实时保存
                tracker.plot_loss(save_path=model_id + '-P2_MoveLossPlt.png', now_epoch=epoch + 1,
                                  dataset=args.dataset, best_epoch=loss_best['epoch'])
                tracker.save_fus(fus_img=ir_move, img_path=model_id + '-P2_MoveIRImg.png')
                tracker.save_fus(fus_img=ir_reg, img_path=model_id + '-P2_MoveIRReg.png')
                tracker.save_fus(fus_img=ir_img, img_path=model_id + '-P2_IRImg.png')
                tracker.save_fus(fus_img=vis_img, img_path=model_id + '-P2_VISImg.png')
                tracker.save_fus(fus_img=reg_fus_img, img_path=model_id + '-P2_RegFusImg.png')
                # tracker.save_fus(fus_img=fus_img_test, img_path=model_id + '-P2_FusImg.png')
                # tracker.save_fus(fus_img=recon_ir_gradient, img_path=model_id + '-P2_Recon_ir_Img.png')
                tracker.save_fus(fus_img=reg_recon_ir_gradient, img_path=model_id + '-P2_Reg_Recon_ir_Img.png')
                # tracker.save_fus(fus_img=unreg_img_test, img_path=model_id + '-P2_UnRegFusImg.png')

                # 实时保存模型 last和best
                # 5.保存best模型 -> 应该是计算一个batch的loss而不是每个step的loss
                if loss_total <= loss_best['loss']:
                    loss_best['loss'] = loss_total
                    loss_best['epoch'] = epoch + 1
                    tracker.save_model(model, model_path=model_path + '/MoveRegImageFusModel-best.pth')
                # 每个epoch输出关键信息以及，测试结果
                t.write("\33[0m" + '\n'.join(print_info))
                # 使用学习率衰减策略
                if args.scheduler[0] == 'True':
                    scheduler.step()
                print(f"flow_loss= {loss1.item()} reg_loss= {loss2.item()} ir_img_loss= {loss3.item()}")
                print(f"learning={optimizer.state_dict()['param_groups'][0]['lr']}, 总损失为:{loss_total:.4f}，最佳损失为第{loss_best['epoch']}/{epoch + 1}e:{loss_best['loss']:.8f}, 训练时间：{round((time.time() - start_time) / 60 / 60):.4f} / {round((time.time() - start_time) / 60 / 60 * num_epochs / (epoch + 1)):.4f}小时...{time.strftime('%y-%m%d-%H%M-%S')}\n")
                # 更新学习率
            # scheduler.step()
        # 5. 保存last模型
        tracker.save_model(model, model_path=model_path + '/MoveRegImageFusModel-last.pth')
        print(f"模型已全部训练完成...{args.id}...结束时间为：{time.strftime('%y-%m%d-%H%M')}...训练需：{round((time.time() - start_time) / 60 / 60, 2)}小时...最佳模型Epoch为：{loss_best['epoch']}.")
        # Tensorboard: 记录模型


if __name__ == "__main__":
    start_time = time.time()
    print(f"当前时间为：{time.strftime('%y-%m%d-%H%M')}")
    # 初始化命令行参数
    args = ConfigManager(config_path='./Config/config.yaml').args
    print(args)
    # 模型训练主函数
    p2 = Train_Phase2(model_id=args.phase2_model_id, phase2_ModelPath=args.phase2_ModelPath)
    p2.train()
