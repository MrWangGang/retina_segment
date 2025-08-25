import numpy as np
import tifffile as tiff
from PIL import Image

try:
    # 读取掩膜图像
    mask = tiff.imread("./datasets/train/masks/Cirrus_TRAIN004_045.tiff")
    # 查看图像的数据类型和取值范围
    print(f"数据类型: {mask.dtype}")
    print(f"最小值: {np.min(mask)}, 最大值: {np.max(mask)}")

    # 将 NumPy 数组转换为 PIL 图像
    mask_image = Image.fromarray(mask.astype(np.uint8))

    # 调整对比度
    min_val = 0
    max_val = 3
    # 这里简单线性拉伸对比度
    scale = 255 / (max_val - min_val)
    contrast_adjusted_image = mask_image.point(lambda p: (p - min_val) * scale if min_val <= p <= max_val else 0)

    # 显示图像
    contrast_adjusted_image.show()

except FileNotFoundError:
    print("未找到指定的图像文件，请检查文件路径。")
except Exception as e:
    print(f"处理图像时出现错误: {e}")