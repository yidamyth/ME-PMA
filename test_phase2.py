"""
Author: YidaChen
Time is: 2023/9/5
this Code: 基于扭曲图像的变换
"""
import argparse
import os
import os.path
import shutil

import torch
import torchvision
from torch.utils.data import DataLoader

from Model.Architecture.RegImageFusModel import RegImageFusModel
from Util.dataset import IRVISDataset
from Util.gray_to_RGB import YtoRGB
from Util.image_transform import ImageTransformer
from Util.transform import DataTransform

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test():
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
    else:
        print("cpu only")
    test_path = args.test_path

    if args.make_move:
        # 开启测试模式，将返回图像路径
        print("直接生成偏移红外图像...")
        dataset_test = IRVISDataset(ir_path=os.path.join(test_path, 'ir'),
                                    vis_path=os.path.join(test_path, 'vis'), transform=transform, img_mode=args.img_mode,
                                    train_mode='test')
    else:
        # 直接加载有偏移图像
        print("直接加载偏移红外图像...")
        dataset_test = IRVISDataset(ir_path=os.path.join(test_path, 'ir_move'),
                                    vis_path=os.path.join(test_path, 'vis'), transform=transform, img_mode=args.img_mode,
                                    train_mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=4, shuffle=True,
                                 pin_memory=True, drop_last=False)

    # 2. 初始化模型方法1
    # model = RegImageFusModel(registration=True, use_bn=True)
    # model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

    # # 初始化模型方法2
    # model = RegImageFusModel(registration=False, use_bn=True, distil=False)
    # model = model.to(device)
    # # 加载预训练的模型权重
    # model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

    # 初始化模型方法2: 加载模型全部参数
    model = torch.load(args.model_path, map_location=device)
    model.registration = False
    model.use_bn = True
    model.distil = False

    # 初始化文件夹
    if not os.path.exists(fusion_path):
        os.makedirs(fusion_path)
        print(f"正在创建{fusion_path}...")
    else:
        shutil.rmtree(fusion_path)
        os.makedirs(fusion_path)
        print(f"已存在，正在创建{fusion_path}...")

    if not os.path.exists(ir_reg_path):
        os.makedirs(ir_reg_path)
        print(f"正在创建{ir_reg_path}...")
    else:
        shutil.rmtree(ir_reg_path)
        os.makedirs(ir_reg_path)
        print(f"已存在，正在创建{ir_reg_path}...")

    if args.make_move:
        if not os.path.exists(ir_move_path):
            os.makedirs(ir_move_path)
            print(f"正在创建{ir_move_path}...")
        else:
            shutil.rmtree(ir_move_path)
            os.makedirs(ir_move_path)
            print(f"已存在，正在创建{ir_move_path}...")

    # 开启测试模型
    model.eval()
    with torch.no_grad():  # 在这个块中，不计算梯度
        print("测试即将开始...")
        num = 1
        for step, (ir_img, vis_img, img_name) in enumerate(dataloader_test):
            # 分批次放入模型
            if args.make_move:
                # gpu加速
                ir_img, vis_img = ir_img.to(device), vis_img.to(device)
                image_transformer = ImageTransformer()
                # # 生成偏移图像对, 对红外图像添加仿射和弹性变换
                move_ir_img = image_transformer.transform(ir_img)
                model_dict = model(ir_img=move_ir_img, vis_img=vis_img)
                fus_img, reg_ir_img = model_dict['unreg_fus_img'], model_dict['reg_recon_ir_gradient']

            else:
                # gpu加速
                ir_img, vis_img = ir_img.to(device), vis_img.to(device)
                move_ir_img = ir_img
                model_dict = model(ir_img=move_ir_img, vis_img=vis_img)
                fus_img, reg_ir_img = model_dict['unreg_fus_img'], model_dict['reg_recon_ir_gradient']
            print(f"{str(num)} / {len(dataset_test)}...正在融合图像{img_name[0]}...")
            # print(fusion_path)
            # 保存图像
            save_path = os.path.join(fusion_path, img_name[0])
            save_path_ir_reg = os.path.join(ir_reg_path, img_name[0])

            print(save_path)
            torchvision.utils.save_image(fus_img, save_path)
            torchvision.utils.save_image(reg_ir_img, save_path_ir_reg)

            # 保存偏移图像
            if args.make_move:
                save_ir_path = os.path.join(ir_move_path, img_name[0])
                torchvision.utils.save_image(move_ir_img, save_ir_path)

            num += 1
        # TODO：加上结合YCbCr通道，直接保存彩色图
        if args.img_mode == 'rgb':
            vis_path = os.path.join(test_path, 'vis')
            fus_path = fusion_path
            rgb_fus_path = os.path.join(test_path, 'Results','UnAligned','fus_rgb')
            y_rgb = YtoRGB(vis_path=vis_path, fus_path=fus_path, rgb_fus_path=rgb_fus_path)
            y_rgb.master()

        print(f"测试完成，请到{fusion_path}查看融合图像...")


if __name__ == '__main__':
    test_path = {'RoadScene': './DataSet/IVIF/RoadScene/RoadS_test', 'M3FD': './DataSet/IVIF/M3FD/test', 'MSRS': './DataSet/IVIF/MSRS/test'}
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default=test_path['MSRS'],
                        help='测试集路径')
    parser.add_argument('--model_path', type=str,
                        default='./Model/Parameters/24-1119-1001/MoveRegImageFusModel-best.pth',
                        help='加载融合模型的路径')
    parser.add_argument('--img_mode', type=str,
                        default='rgb',
                        help='{gray, rgb}一个将可见光转换成灰度图，另外一个则是取YCbCr通道的Y通道进行')
    parser.add_argument('--make_move', action='store_false', default=False,
                        help='如果为True，那么直接对ir图像施加偏移然后保存成ir_move; 否则加载数据集中的ir_move 替换ir图像得到融合图像结果')
    args = parser.parse_args()
    print(args)
    # 保存融合图像的路径  ./fus
    fusion_path = os.path.join(args.test_path, 'Results','UnAligned','fus')
    ir_move_path = os.path.join(args.test_path, 'Results','UnAligned','ir_move')
    ir_reg_path = os.path.join(args.test_path, 'Results','UnAligned','ir_reg')

    # 数据变换
    # transform = DataTransform(img_size='original', train_mode='test')  # 'original'
    transform = DataTransform(img_size=[304, 400], train_mode='test')  # 256
    # 主函数
    test()
