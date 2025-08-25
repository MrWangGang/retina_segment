import os
from glob import glob
import torch
import streamlit as st
from torch.utils.data import random_split, DataLoader
from RetinaData import RetinaData
import matplotlib.pyplot as plt
import numpy as np


# 加载模型到 CPU
model = torch.load('best_model.pth', map_location=torch.device('cpu'), weights_only=False)

# 获取训练集图像文件路径，并按字母顺序排序
train_x = sorted(glob(os.path.join("./datasets/train/images", "*.tiff")))
# 获取训练集掩码文件路径，并按字母顺序排序
train_y = sorted(glob(os.path.join("./datasets/train/masks", "*.tiff")))

model.eval()
dataset = RetinaData(train_x, train_y)
train_size = 0.3
train_dataset, val_dataset = random_split(dataset, [int(train_size * len(dataset)), len(dataset) - int(train_size * len(dataset))])
test_dataset, val_dataset = random_split(val_dataset, [int(train_size * len(val_dataset)), len(val_dataset) - int(train_size * len(val_dataset))])

test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

model.to("cpu")
x = 0

# 定义颜色映射，这里假设类别数为 4
num_classes = 4
color_map = np.array([
    [0, 0, 0],  # 背景类，黑色
    [255, 0, 0],  # 类别 1，红色
    [0, 255, 0],  # 类别 2，绿色
    [0, 0, 255]  # 类别 3，蓝色
])

# 定义类别名称
class_names = ["Background", "Intraretinal Fluid", "Subretinal Fluid", "Pigment Epithelial Detachment"]

st.title("图像分割结果展示")

for data in test_loader:
    img, msk = data
    img = img.float().to("cpu")  # 确保数据在 CPU 上
    msk = msk.long().to("cpu")  # 确保数据在 CPU 上

    # 检查掩码中的类别数量是否为 4
    unique_classes = np.unique(msk.cpu().numpy())
    if len(unique_classes) == num_classes:
        with torch.no_grad():
            out = model(img)

            # 获取预测分割结果
            pred_seg_result = torch.argmax(out[0], dim=0).cpu().numpy()
            # 获取真实分割结果
            true_seg_result = msk[0].cpu().numpy()

            # 将预测分割结果转换为彩色图像
            pred_color_seg = np.zeros((pred_seg_result.shape[0], pred_seg_result.shape[1], 3), dtype=np.uint8)
            for c in range(num_classes):
                pred_color_seg[pred_seg_result == c] = color_map[c]

            # 将真实分割结果转换为彩色图像
            true_color_seg = np.zeros((true_seg_result.shape[0], true_seg_result.shape[1], 3), dtype=np.uint8)
            for c in range(num_classes):
                true_color_seg[true_seg_result == c] = color_map[c]

            # 将彩色分割结果与原图融合
            img_np = img[0][0].cpu().numpy()
            img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)  # 将单通道图像转换为三通道
            alpha = 0.5  # 透明度

            # 原图 + 真实掩膜图映射
            true_blended = (alpha * true_color_seg + (1 - alpha) * img_np).astype(np.uint8)

            # 原图 + 预测掩膜图映射
            pred_blended = (alpha * pred_color_seg + (1 - alpha) * img_np).astype(np.uint8)

            # 创建一个包含两个子图的画布
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))

            # 显示原图 + 真实掩膜图映射
            axes[0].imshow(true_blended)
            axes[0].set_title('Original Image + True Mask')

            # 显示原图 + 预测掩膜图映射
            axes[1].imshow(pred_blended)
            axes[1].set_title('Original Image + Predicted Mask')

            # 添加图例
            patches = [plt.Rectangle((0, 0), 1, 1, fc=color_map[c] / 255.0) for c in range(num_classes)]
            axes[0].legend(patches, class_names, loc='upper right')
            axes[1].legend(patches, class_names, loc='upper right')

            st.pyplot(fig)

            x += 1
            if x >= 20:
                break