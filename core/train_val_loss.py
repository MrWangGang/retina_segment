import matplotlib.pyplot as plt
import re

# 从文本文件中读取内容
with open('未命名2.rtf', 'r', encoding='utf-8') as file:  # 将 'your_file.txt' 替换为实际的文件名
    log_text = file.read()

# 提取训练损失
train_loss_pattern = r'Loss for Epoch:(\d+) Loss:([\d.]+)'
train_epochs = []
train_losses = []
for match in re.finditer(train_loss_pattern, log_text):
    epoch = int(match.group(1))
    loss = float(match.group(2))
    train_epochs.append(epoch)
    train_losses.append(loss)

# 提取验证损失
val_loss_pattern = r'Validation Loss for Epoch:(\d+) Loss:([\d.]+)'
val_epochs = []
val_losses = []
for match in re.finditer(val_loss_pattern, log_text):
    epoch = int(match.group(1))
    loss = float(match.group(2))
    val_epochs.append(epoch)
    val_losses.append(loss)

# 绘制训练损失曲线
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_epochs, train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# 绘制验证损失曲线
plt.subplot(1, 2, 2)
plt.plot(val_epochs, val_losses, marker='o', color='orange')
plt.title('Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()