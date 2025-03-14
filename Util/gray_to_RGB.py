"""
Author: YidaChen
Time is: 2023/10/5
this Code: 当 img_mode=rgb 时分离 YCbCr 通道的 Y 通道与 IR 图像进行融合，对融合结果与 VIS 图像的 CbCr通道进行融合得到彩色图像

注意：尽可能保存融合图像与 VIS 图像维度大小相同
"""
import os.path
import shutil

from PIL import Image


class YtoRGB:
    def __init__(self, vis_path, fus_path, rgb_fus_path):
        self.vis_path = vis_path
        self.fus_path = fus_path
        self.rgb_fus_path = rgb_fus_path

    def master(self):
        vis_path = self.vis_path
        fus_path = self.fus_path
        rgb_fus_path = self.rgb_fus_path
        if not os.path.exists(rgb_fus_path):
            print("正在创建文件夹...")
            os.makedirs(rgb_fus_path)
        else:
            print("文件夹已存在，正在重新创建...")
            shutil.rmtree(rgb_fus_path)
            os.makedirs(rgb_fus_path)

            # 获取图像名字
        file_name = os.listdir(fus_path)
        for img_name in file_name:
            if img_name.lower().endswith(('.jpg', 'png', 'jpeg', 'tiff')):
                # 融合图像gray
                fus_img_path = os.path.join(fus_path, img_name)
                # 灰度图 这是融合后的 Y 通道
                fus_img = Image.open(fus_img_path).convert('L')
                w, h = fus_img.size
                # 可见光图像 rgb
                vis_img_path = os.path.join(vis_path, img_name)
                vis_img = Image.open(vis_img_path).convert('RGB')
                vis_img = vis_img.convert('YCbCr')
                y, cb, cr = vis_img.split()

                # 如果大小不同就 resize 相同大小，尽可能保证输入大小相同
                if cb.size != fus_img.size:
                    cb = cb.resize((w, h))
                    cr = cr.resize((w, h))
                # 融合 Y, Cb, Cr 通道
                merged_image = Image.merge('YCbCr', (fus_img, cb, cr))

                # 转回 RGB 色彩空间（如果需要）
                merged_image_rgb = merged_image.convert('RGB')
                # 保存图像
                img_save_path = os.path.join(rgb_fus_path, img_name)
                merged_image_rgb.save(img_save_path)
                print(f"正在保存{img_save_path}")


if __name__ == '__main__':

    # ir vis fus数据的路径
    dataset_path = '/Users/yida/Desktop/dataset/RoadScene-master'
    # 融合图像路径gray
    fus_path = os.path.join(dataset_path, 'fus')
    # 可见光图像路径 rgb
    vis_path = os.path.join(dataset_path, 'vis')
    # 重建 rgb 融合图像路径
    rgb_fus_path = os.path.join(dataset_path, 'fus_rgb')
    # 新建文件夹
    y_rgb = YtoRGB(vis_path=vis_path, fus_path=fus_path, rgb_fus_path=rgb_fus_path)
    y_rgb.master()
