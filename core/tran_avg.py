import matplotlib.pyplot as plt
import re

# 从文件读取内容
file_path = '未命名2.rtf'  # 替换为实际文件路径
try:
    with open(file_path, 'r', encoding='utf-8') as file:
        log_text = file.read()
except FileNotFoundError:
    print(f"未找到文件: {file_path}")
except Exception as e:
    print(f"读取文件时出错: {e}")
else:
    # 定义正则表达式模式来匹配包含 grad_fn=<DivBackward0> 的 Dice 系数
    pattern = r'Dice coeff for class  (\d+)  =  tensor\(([\d.e-]+), device=\'cuda:0\', grad_fn=<DivBackward0>\)'

    # 初始化存储每个 class 的 Dice 系数的列表
    class_dice_coeffs = {i: [] for i in range(4)}

    # 遍历匹配结果
    for match in re.finditer(pattern, log_text):
        class_num = int(match.group(1))
        dice_coeff = float(match.group(2))
        class_dice_coeffs[class_num].append(dice_coeff)

    # 计算每个 epoch 的平均 Dice 系数
    num_epochs = len(next(iter(class_dice_coeffs.values())))
    avg_dice_coeffs = []
    for epoch in range(num_epochs):
        epoch_coeffs = [class_dice_coeffs[class_num][epoch] for class_num in range(4)]
        avg_dice_coeff = sum(epoch_coeffs) / 4
        avg_dice_coeffs.append(avg_dice_coeff)

    # 绘制平均 Dice 系数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(avg_dice_coeffs, marker='o', color='red', label='Average Dice Coefficient')

    plt.title('Average Train Dice Coefficient')
    plt.xlabel('Epoch')
    plt.ylabel('Average Dice Coefficient')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
