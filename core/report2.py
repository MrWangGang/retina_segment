import json
import matplotlib.pyplot as plt

# 学习率列表
learning_rates = [0.0001, 0.0005, 0.0021, 0.005, 0.0058]

# 创建一个2x3的子图布局，共5个有效子图
fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()  # 将二维数组展平为一维数组以便于索引

# 循环读取5个JSON文件并在对应的子图中绘制曲线
for i in range(5):
    file_path = f"loss_data_{i + 1}.json"
    with open(file_path, 'r') as file:
        data = json.load(file)
        losses = data["epoch_losses"]

        ax = axes[i]
        ax.plot(losses, label=f"Learning Rate: {learning_rates[i]}")
        ax.set_title(f"Learning Rate: {learning_rates[i]}")
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

# 隐藏多余的子图
if len(learning_rates) < 6:
    axes[-1].axis('off')

# 调整子图之间的间距
plt.tight_layout()

# 显示图形
plt.show()