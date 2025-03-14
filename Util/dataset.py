"""
Author: YidaChen
Time is: 2023/9/4
this Code: 重写数据加载类
"""
import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader

from Util.transform import DataTransform


class IRVISDataset(Dataset):
    def __init__(self, ir_path, vis_path, transform=None, img_mode='gray', train_mode='train'):
        self.ir_folder = ir_path
        self.vis_folder = vis_path
        # transform 类
        self.transform = transform

        self.ir_images = self._load_images(self.ir_folder)
        self.vis_images = self._load_images(self.vis_folder)

        self.pairs = self._get_image_pairs(self.ir_images, self.vis_images)

        # 设置输入图像的模型，如果为 gray 就全部转换成灰度图，为 rgb就转换到 YCbCr 颜色空间分离 Y 通道
        self.img_mode = img_mode
        # 设置训练模式，训练模式不返回图像路径，测试模式返回图像路径; 保存融合图像与 ir 和 vis 图像相同名字
        self.train_mode = train_mode

    def _load_images(self, folder):
        valid_formats = ['jpg', 'png', 'jpeg', 'tiff']
        return {f: os.path.join(folder, f) for f in os.listdir(folder) if f.split('.')[-1].lower() in valid_formats}

    def _get_image_pairs(self, ir_images, vis_images):
        pairs = []
        for name, ir_path in ir_images.items():
            if name in vis_images:
                pairs.append((ir_path, vis_images[name]))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # 获取数据集的图像路径
        ir_path, vis_path = self.pairs[idx]
        # 获取图像名称
        img_name = os.path.basename(ir_path)
        if self.img_mode == 'gray':
            # 目前将 ir 和 vis 都转换成灰度图
            ir_image = Image.open(ir_path).convert('L')  # Convert to grayscale
            vis_image = Image.open(vis_path).convert('L')  # Convert to RGB

        elif self.img_mode == 'rgb':
            # 红外图像采用灰度图进行融合，融合的结果与Y通道进行融合，最终结合CbCr通道转到灰度图 -> 灰度图和他的Y通道不同，因此转换到Y通道在操作
            ir_image = Image.open(ir_path).convert('L')  # Convert to grayscale
            # ir_image, _, _ = ir_image.convert('RGB').convert('YCbCr').split()
            # 将 RGB 空间转换到YCbCr颜色空间，并分离 Y 通道
            vis_image = Image.open(vis_path).convert('RGB')  # Convert to RGB
            # 转换颜色空间
            vis_image = vis_image.convert('YCbCr')
            # 通道分离
            y, cb, cr = vis_image.split()
            # 基于 Y 通道进行操作
            vis_image = y

        # 变换红外光与可见光
        ir_image, vis_image = self.transform.get_transform(ir_image, vis_image)
        # save_image(ir_image, f"/Users/yida/Desktop/test/ir_vi/ir/{idx}_ir.png")
        # save_image(vis_image, f"/Users/yida/Desktop/test/ir_vi/vis/{idx}_vis.png")

        # 模式选择
        if self.train_mode == 'train':
            return ir_image, vis_image
        elif self.train_mode == 'test':
            return ir_image, vis_image, img_name


if __name__ == '__main__':

    transform = DataTransform(256)

    # 初始化数据集和数据加载器
    ir_vis_dataset = IRVISDataset(ir_path='/Users/yida/Desktop/dataset/M3FD/M3FD_Fusion/Ir',
                                  vis_path='/Users/yida/Desktop/dataset/M3FD/M3FD_Fusion/Vis', transform=transform,
                                  img_mode='rgb', train_mode='test')
    dataloader = DataLoader(ir_vis_dataset, batch_size=4, shuffle=True)
    for img_ir, img_vis, img_name in dataloader:
        print(img_ir.shape)
        print(img_vis.shape)
        print(img_name)
