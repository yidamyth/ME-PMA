"""
Author: YidaChen
Time is: 2023/11/30
this Code: 对图像应用仿射和弹性变换，封装成一个类，可以测试时调用；也可以在训练时调用，应用变换
"""
import kornia.augmentation as K
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import ToTensor


class ImageTransformer:
    def __init__(self):
        # degrees：旋转角度，在这个范围随机旋转； translate：水平和垂直方向的平移范围比例，这个范围内随机；scale：缩小比例在这个范围内随机缩小或者放大； shear：剪切变换(0,10)
        # self.affine_aug = K.RandomAffine(degrees=(-2, 2), translate=(0.02, 0.02), scale=(0.98, 1.02), shear=0, p=1.0)
        self.affine_aug = K.RandomAffine(degrees=(-2, 2), translate=(0.02, 0.02), shear=0, p=1.0)


        # 弹性变换
        # self.elastic_aug = K.RandomElasticTransform(p=1.0, kernel_size=(101, 101), sigma=(32, 32), keepdim=True)
        self.elastic_aug = K.RandomElasticTransform(p=1.0, kernel_size=(63, 63), sigma=(32, 32), keepdim=True)

    def transform(self, image_tensor):
        # 应用仿射变换
        image_tensor = self.affine_aug(image_tensor)
        # 应用弹性变换
        image_tensor = self.elastic_aug(image_tensor)
        return image_tensor


def load_image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')
    return ToTensor()(image)


def visualize(original_image, transformed_image):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(original_image.permute(1, 2, 0))
    axs[0].set_title('Original Image')
    axs[0].axis('off')

    axs[1].imshow(transformed_image.permute(1, 2, 0))
    axs[1].set_title('Transformed Image')
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig("/Users/yida/Desktop/1.png")
    plt.show()


if __name__ == '__main__':
    # 示例使用
    image_path = '/Users/yida/Desktop/ir_256.png'  # 替换为实际路径
    # 转换成tensor格式
    image_tensor = load_image_to_tensor(image_path).unsqueeze(0)  # 添加一个批次维度
    # image_tensor = torch.rand(10, 3, 256, 256)
    print(image_tensor.shape)

    # 实例化类
    transformer = ImageTransformer()
    transformed_image = transformer.transform(image_tensor)
    print(transformed_image.shape)

    # 可视化结果
    visualize(image_tensor.squeeze(), transformed_image.squeeze())
