import json
import numpy as np
import matplotlib.pyplot as plt

# 定义 JSON 文件列表
json_files = [
    'dice_coefficients1.json',
    'dice_coefficients2.json',
    'dice_coefficients3.json',
    'dice_coefficients4.json',
    'dice_coefficients5.json'
]

# 定义对应的学习率列表
learning_rates = [0.005, 0.0021, 0.0058, 0.0005, 0.0001]

# 颜色列表，用于区分不同类别
colors = ['r', 'g', 'b', 'y', 'm']

# 创建一个包含 5 个子图的图形
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
# 因为只需要 5 个子图，隐藏多余的子图
axes[1, 2].axis('off')
axes = axes.flatten()[:5]

# 循环处理每个 JSON 文件
for i, file in enumerate(json_files):
    try:
        # 从 JSON 文件中读取数据
        with open(file, 'r') as f:
            loaded_data = json.load(f)

        # 提取每个类别的 Dice 系数
        num_classes = len(loaded_data[0]) - 1  # 减去 'epoch' 键
        classes = [f"Class {j}" for j in range(num_classes)]
        dice_coeffs = {cls: [] for cls in classes}
        for epoch_data in loaded_data:
            for cls in classes:
                dice_coeffs[cls].append(epoch_data[cls])

        # 绘制 Dice 系数增长曲线到对应的子图
        for j, cls in enumerate(classes):
            axes[i].plot(range(1, len(loaded_data) + 1), dice_coeffs[cls], label=cls, color=colors[j])

        # 添加标签和标题
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Dice Coefficient')
        axes[i].set_title(f'train with learning rate {learning_rates[i]}')
        axes[i].legend()

    except FileNotFoundError:
        print(f"文件 {file} 未找到，请检查文件路径。")
    except Exception as e:
        print(f"读取文件 {file} 时出现错误: {e}")

# 调整子图布局
plt.tight_layout()
# 显示图形
plt.show()
