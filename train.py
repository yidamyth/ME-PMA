"""
Author: YidaChen
Time is: 2023/9/4
this Code: 初步实现图像融合与配准的完整代码

"""
import os
import os.path
import time

import numpy as np
import torch
import torchvision
from setproctitle import setproctitle
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from Config.config_parser import ConfigManager
from Model.Architecture.RegImageFusModel import RegImageFusModel
from Util.dataset import IRVISDataset
from Util.loss_function import FusionLoss
from Util.random_seed import random_seed
from Util.training_monitor import Tracker
from Util.transform import DataTransform

# 检查错误
# autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    if torch.cuda.is_available():
        # 释放未使用的显存
        torch.cuda.empty_cache()
        # 输出显卡信息
        print(torch.cuda.get_device_name(0))
    # 模型的唯一标识符
    model_id = args.id
    # 初始化对应保存模型参数文件夹
    model_path = "./Model/Parameters/" + model_id
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # 设置线程名称
    setproctitle(f"yida_{model_id}")
    # 创建 Tensorboard
    writer = SummaryWriter(f'./Logs/tensorboard/{model_id}')
    # 0. 设计随机数种子
    random_seed(seed=args.seed)
    # 1. 数据加载
    transform = DataTransform(img_size=args.img_size, size_mode=args.size_mode, train_mode='train')

    # 数据集路径 获取字典，然后再传入数据集名称
    train_path = args.train_paths[args.dataset]
    # 自定义dataset
    dataset = IRVISDataset(ir_path=os.path.join(train_path, 'ir'),
                           vis_path=os.path.join(train_path, 'vis'), transform=transform, img_mode=args.img_mode, train_mode='train')
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

    # 2. 初始化模型 registration为 True， use_bn为 True
    model = RegImageFusModel(registration=args.registration, use_bn=args.use_bn)
    model = model.to(device)

    # 3. 定义损失函数和优化器
    if args.opt_type == 'Adam':
        print("选择使用 Adam 优化器...")
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
        optimizer = optim.RMSprop(model.parameters(), lr=args.learning_rate, alpha=0.99, eps=1e-08,
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
    print(model)
    print(model.registration)
    print("数据准备已完成，正在准备模型训练...")
    #  封装类, 调用各种功能
    tracker = Tracker()
    # 保存最佳loss对应模型参数， 初始化字典
    loss_best = {'loss': np.inf, 'epoch': 2023}
    print_info = []
    for epoch in range(num_epochs):
        with tqdm(enumerate(dataloader), total=len(dataloader), desc=f'\33[31m Epoch {epoch + 1}/{num_epochs}') as t:
            loss_total = 0  # 初始化每一轮的总loss
            for step, (ir_img, vis_img) in t:  # 用tqdm来替换原始的输出 ->for step, (ir_img, vis_img) in enumerate(dataloader):
                optimizer.zero_grad()
                ir_img, vis_img = ir_img.to(device), vis_img.to(device)

                # 训练步骤
                model_dict = model(ir_img=ir_img, vis_img=vis_img)
                fus_img = model_dict['fus_img']

                # img_grid = torchvision.utils.make_grid(fus_img[:6, :, :, :])
                # writer.add_image('image_grid', img_grid, epoch + 1)

                loss_info = FusionLoss(model_dict, registration=True, loss_weight=args.loss_weight).get_loss()
                loss = loss_info["loss"].to(device)
                writer.add_scalar('TrainingLoss', loss.item(), epoch * len(dataloader) + step)

                loss.backward()
                optimizer.step()

                # 计算一个batch的loss
                loss_total += loss.item()
                # 为了使用tqdm，导致模型或者损失函数中间的值无法直接print，因此以info的形式返回，每个epoch结束时输出
                print_info = [loss_info["loss1_info"], loss_info["loss2_info"]]
                # 更新 tqdm 后缀，显示 epoch、step 和 loss；\31m是红色 \33m是蓝色，为了在文本末尾重置格式（回到默认格式），通常会使用 \33[0m
                t.set_postfix_str(f"Step: {step + 1}/{len(dataloader)}, Loss: {loss.item():.4f}")
                # 输出训练过程
                # if (step + 1) % 10 == 0:
                #     print(f"Epoch [{epoch + 1}/{num_epochs}], Step [{step + 1}/{len(dataloader)}], Loss: {loss.item()}\n")

                # 保存损失
                tracker.add_loss(loss)
                # # 实时保存
                # tracker.plot_loss(save_path=model_id + '-LossPlt.png', now_epoch=epoch + 1, dataset=args.dataset, best_epoch=loss_best['epoch'])
                # tracker.save_fus(fus_img=fus_img, img_path=model_id + '-FusImg.png')

            # 实时保存模型 last和best
            # 5.保存best模型 -> 应该是计算一个batch的loss而不是每个step的loss
            if loss_total <= loss_best['loss']:
                loss_best['loss'] = loss_total
                loss_best['epoch'] = epoch + 1
                tracker.save_model(model, model_path=model_path + '/RegImageFusModel-best.pth')
            # 每个epoch输出关键信息以及，测试结果
            t.write("\33[0m" + '\n'.join(print_info))
            # 使用学习率衰减策略
            if args.scheduler[0] == 'True':
                scheduler.step(),
            print(f"learning={optimizer.state_dict()['param_groups'][0]['lr']}, 总损失为:{loss_total:.4f}，最佳损失为第{loss_best['epoch']}/{epoch + 1}e:{loss_best['loss']:.4f}, 训练时间"
                  f"：{round((time.time() - start_time) / 60.0 / 60.0):.4f} / {round((time.time() - start_time) / 60.0 / 60.0 * num_epochs / (epoch + 1)):.4f}小时...{time.strftime('%y-%m%d-%H%M-%S')} \n")
            # 实时每个Epoch保存
            tracker.plot_loss(save_path=model_id + '-LossPlt.png', now_epoch=epoch + 1, dataset=args.dataset, best_epoch=loss_best['epoch'])
            tracker.save_fus(fus_img=fus_img, img_path=model_id + '-FusImg.png')
            # 新增重建的红外与可见光图像
            # tracker.save_fus(fus_img=model_dict['recon_img']['recon_ir'], img_path=model_id + '-ReconIr.png')
            # tracker.save_fus(fus_img=model_dict['recon_img']['recon_vis'], img_path=model_id + '-ReconVis.png')
            # tracker.save_fus(fus_img=model_dict['recon_img']['de_recon_ir'], img_path=model_id + '-DeReconIr.png')
            # tracker.save_fus(fus_img=model_dict['recon_img']['de_recon_vis'], img_path=model_id + '-DeReconVis.png')
            tracker.save_fus(fus_img=model_dict['recon_gradient']['recon_ir_gradient'], img_path=model_id + '-ReconIrGradient.png')
            tracker.save_fus(fus_img=model_dict['recon_gradient']['recon_vis_gradient'], img_path=model_id + '-ReconVisGradient.png')
            tracker.save_fus(fus_img=ir_img, img_path=model_id + '-ir.png')
            tracker.save_fus(fus_img=vis_img, img_path=model_id + '-vis.png')

        # 每间隔多少轮测试一次
        # if (epoch + 1) % args.num_step == 0:
        #     # TODO：最好加上全部指标，可以边融合的时候观测指标的变化
        #     print("已进入模型测试...正在计算指标...")
        #     # 进入测试模式 -> epoch循环的时候没有调回去，你这样会影响模型的啊！
        #     # model.eval()
        #     # 计算指标数共13个
        #     total_metrics = [[] for i in range(13)]
        #     metrics_dict = {}
        #     with torch.no_grad():
        #         for step_test, (ir_test, vis_test) in enumerate(test_dataloader):
        #             ir_test, vis_test = ir_test.to(device), vis_test.to(device)
        #             # 测试步骤
        #             model_dict_test = model(ir_img=ir_test, vis_img=vis_test)
        #             fus_test = model_dict_test['fus_img']  # 融合图像
        #             # 计算指标，返回的是一个字典
        #             metrics_dict = tracker.get_metrics(ir_img=ir_test, vis_img=vis_test, fus_img=fus_test)
        #             for i, value in enumerate(metrics_dict.values()):
        #                 total_metrics[i] += value
        #         for i, j in enumerate(total_metrics):
        #             total_metrics[i] = sum(j) / len(j)
        #         for i, j in enumerate(metrics_dict.keys()):
        #             print(f"{j}={total_metrics[i]:.6f}")  # 保留6位小数
        #         print("\n")
        # 更新学习率
    # 5. 保存last模型
    tracker.save_model(model, model_path=model_path + '/RegImageFusModel-last.pth')
    print(
        f"模型已全部训练完成...{args.id}...结束时间为：{time.strftime('%y-%m%d-%H%M')}...训练需：{round((time.time() - start_time) / 60 / 60, 2)}小时...最佳模型Epoch为：{loss_best['epoch']}.")
    # Tensorboard: 记录模型


if __name__ == "__main__":
    start_time = time.time()
    print(f"当前时间为：{time.strftime('%y-%m%d-%H%M')}")
    # 初始化命令行参数
    args = ConfigManager(config_path='./Config/config.yaml').args
    print(args)
    # 模型训练主函数
    main()
