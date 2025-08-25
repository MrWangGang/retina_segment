import cv2
import imageio
import torch
# 定义一个自定义的数据集类，继承自 torch.utils.data.Dataset
class RetinaData(torch.utils.data.Dataset):
    def __init__(self, images, masks):
        # 初始化图像文件路径列表
        self.image_paths = images
        # 初始化掩码文件路径列表
        self.masks_paths = masks

    def __getitem__(self, i):
        # 根据索引 i 读取对应的图像，以灰度模式读取
        x = cv2.imread(self.image_paths[i], cv2.IMREAD_GRAYSCALE)
        # 将图像调整为 512x512 大小
        x = cv2.resize(x, (512, 512))
        # 调整图像的维度，增加一个通道维度，使其形状为 (1, 512, 512)
        x = x.reshape(1, 512, 512)

        # 根据索引 i 读取对应的掩码图像，以灰度模式读取
        y = cv2.imread(self.masks_paths[i], cv2.IMREAD_GRAYSCALE)
        # 注释掉的代码是使用 imageio 读取多帧图像中的第一帧作为掩码图像
        #y = imageio.mimread(self.masks_paths[i])[0]
        # 将掩码图像调整为 512x512 大小，使用最近邻插值方法
        y = cv2.resize(y, (512, 512), interpolation=cv2.INTER_NEAREST)
        # 调整掩码图像的维度，使其形状为 (512, 512)
        y = y.reshape(512, 512)

        # 返回处理后的图像和掩码图像
        return (x, y)

    def __len__(self):
        # 返回数据集的长度，即图像文件路径列表的长度
        return len(self.image_paths)