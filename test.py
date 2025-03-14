"""
Author: YidaChen
Time is: 2023/9/5
this Code: 模型测试
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
from Util.transform import DataTransform
import subprocess  # 调用其他python程序

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import torch
import torchvision.models as models





def test():
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name())
    else:
        print("cpu only")
    print(args.model_path)
    test_path = args.test_path

    # 开启测试模式，将返回图像路径
    dataset_test = IRVISDataset(ir_path=os.path.join(test_path, args.ir_path),
                                vis_path=os.path.join(test_path, 'vis'), transform=transform, img_mode=args.img_mode,
                                train_mode='test')
    dataloader_test = DataLoader(dataset_test, batch_size=1, num_workers=4, shuffle=True,
                                 pin_memory=True, drop_last=False)

    # 初始化模型方法1
    # model = RegImageFusModel(registration=True, use_bn=True)
    # model = model.to(device)
    # model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)

    # 初始化模型方法2: 加载模型全部参数
    model = torch.load(args.model_path, map_location=device)
    model.registration = True
    model.use_bn = True

    # 初始化文件夹
    if not os.path.exists(fusion_path):
        os.makedirs(fusion_path)
        print(f"正在创建{fusion_path}...")
    else:
        shutil.rmtree(fusion_path)
        os.makedirs(fusion_path)
        print(f"已存在，正在创建{fusion_path}...")

    # 重建IR图像
    if not os.path.exists(recon_ir_img_path):
        os.makedirs(recon_ir_img_path)
        print(f"正在创建{recon_ir_img_path}...")
    else:
        shutil.rmtree(recon_ir_img_path)
        os.makedirs(recon_ir_img_path)
        print(f"已存在，正在创建{recon_ir_img_path}...")

    # 开启测试模型
    model.eval()
    with torch.no_grad():  # 在这个块中，不计算梯度
        print("测试即将开始...")
        num = 1
        for step, (ir_img, vis_img, img_name) in enumerate(dataloader_test):
            # 分批次放入模型
            # gpu加速
            ir_img, vis_img = ir_img.to(device), vis_img.to(device)
            model_dict = model(ir_img=ir_img, vis_img=vis_img)
            fus_img = model_dict['fus_img']
            recon_ir_img = model_dict['recon_gradient']['recon_ir_gradient']
            print(f"{str(num)} / {len(dataset_test)}...正在融合图像{img_name[0]}...")
            # print(fusion_path)
            # 保存图像
            save_path = os.path.join(fusion_path, img_name[0])
            save_path_recon_ir = os.path.join(recon_ir_img_path, img_name[0])

            print(save_path)
            torchvision.utils.save_image(fus_img, save_path)
            torchvision.utils.save_image(recon_ir_img, save_path_recon_ir)

            num += 1
        # TODO：加上结合YCbCr通道，直接保存彩色图
        if args.img_mode == 'rgb':
            vis_path = os.path.join(test_path, 'vis')
            fus_path = fusion_path
            rgb_fus_path = os.path.join(test_path, 'Results','Aligned', 'fus_rgb')
            y_rgb = YtoRGB(vis_path=vis_path, fus_path=fus_path, rgb_fus_path=rgb_fus_path)
            y_rgb.master()

        print(f"测试完成，请到{fusion_path}查看融合图像...")
        # 自动调用测试脚本
        # subprocess.run(["/home/yida/anaconda3/envs/pytorch112/bin/python3.9", "/home/yida/PyCharmProject/4090/RegImageFusModel_v3/Util/metrics_batch.py"])


if __name__ == '__main__':
    test_path = {'RoadScene': './DataSet/IVIF/RoadScene/RoadS_test', 'M3FD': './DataSet/IVIF/M3FD/test', 'MSRS': './DataSet/IVIF/MSRS/test'}
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default=test_path['RoadScene'],
                        help='测试集路径')
    parser.add_argument('--model_path', type=str,
                        default='./Model/Parameters/24-1119-1001/MoveRegImageFusModel-best.pth',
                        help='加载融合模型的路径')
    parser.add_argument('--img_mode', type=str,
                        default='rgb',
                        help='{gray, rgb}一个将可见光转换成灰度图，另外一个则是取YCbCr通道的Y通道进行')
    parser.add_argument('--ir_path', type=str,
                        default='ir_move',
                        help='{ir, ir_move}默认为对其的ir进行测试，如果修改为ir_move的话会采用数据集中保存的偏移图像进行测试ir_move,前提是有对应的ir_move文件')
    args = parser.parse_args()
    # 保存融合图像的路径  ./fus
    fusion_path = os.path.join(args.test_path, 'Results','Aligned', 'fus')
    recon_ir_img_path = os.path.join(args.test_path, 'Results','Aligned','recon_ir_img')

    # 数据变换
    transform = DataTransform(img_size='original', train_mode='test')
    # transform = DataTransform(img_size=[304, 400], train_mode='test')
    # 主函数
    test()
