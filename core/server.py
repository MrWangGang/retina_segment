import os
from glob import glob
import torch
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io


# 加载模型到 CPU
model = torch.load('best_model.pth', map_location=torch.device('cpu'), weights_only=False)
model.eval()

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


def preprocess_image(uploaded_image):
    image = Image.open(uploaded_image).convert('L')  # 转换为单通道灰度图像
    image = np.array(image)
    image = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 添加批次维度和通道维度
    return image


def predict(image):
    with torch.no_grad():
        image = image.float().to("cpu")
        out = model(image)
        pred_seg_result = torch.argmax(out[0], dim=0).cpu().numpy()
        return pred_seg_result


def convert_to_color_seg(seg_result):
    color_seg = np.zeros((seg_result.shape[0], seg_result.shape[1], 3), dtype=np.uint8)
    for c in range(num_classes):
        color_seg[seg_result == c] = color_map[c]
    return color_seg


def blend_images(img_np, color_seg):
    img_np = np.repeat(img_np[:, :, np.newaxis], 3, axis=2)
    alpha = 0.5
    blended = (alpha * color_seg + (1 - alpha) * img_np).astype(np.uint8)
    return blended


st.title("图像分割预测")
uploaded_file = st.file_uploader("上传一张图片", type=["tiff", "jpg", "png"])

if uploaded_file is not None:
    # 预处理图像
    input_image = preprocess_image(uploaded_file)
    img_np = input_image[0][0].cpu().numpy()

    # 进行预测
    pred_seg_result = predict(input_image)

    # 将预测分割结果转换为彩色图像
    pred_color_seg = convert_to_color_seg(pred_seg_result)

    # 将彩色分割结果与原图融合
    pred_blended = blend_images(img_np, pred_color_seg)

    # 显示原图和预测结果
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[1].imshow(pred_blended)
    axes[1].set_title('Original Image + Predicted Mask')

    # 添加图例
    patches = [plt.Rectangle((0, 0), 1, 1, fc=color_map[c] / 255.0) for c in range(num_classes)]
    axes[1].legend(patches, class_names, loc='upper right')

    st.pyplot(fig)