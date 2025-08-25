import os
from glob import glob
import segmentation_models_pytorch as smp
import tqdm
import torch
from RetinaData import RetinaData
from Loss import loss_funct


class Config:
    # 数据集路径
    TRAIN_IMAGE_DIR = "./datasets/train/images"
    TRAIN_MASK_DIR = "./datasets/train/masks"
    # 训练集比例
    TRAIN_SIZE = 0.8
    # 批量大小
    BATCH_SIZE = 32
    # 编码器名称
    ENCODER_NAME = "resnet34"
    # 编码器预训练权重
    ENCODER_WEIGHTS = "imagenet"
    # 输入通道数
    IN_CHANNELS = 1
    # 输出通道数
    CLASSES = 4
    # 输出激活函数
    ACTIVATION = 'softmax'
    # 优化器学习率
    LEARNING_RATE = 0.0001
    # 训练周期数
    EPOCHS = 100
    # 连续多少个epoch没有提升则停止训练
    EARLY_STOPPING_PATIENCE = 10


# 获取训练图像和掩膜图像的路径
train_x = sorted(glob(os.path.join(Config.TRAIN_IMAGE_DIR, "*.tiff")))
train_y = sorted(glob(os.path.join(Config.TRAIN_MASK_DIR, "*.tiff")))

# 创建数据集并划分为训练集和验证集
dataset = RetinaData(train_x, train_y)
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(Config.TRAIN_SIZE * len(dataset)),
                                                                     len(dataset) - int(Config.TRAIN_SIZE * len(dataset))])
print(len(train_dataset), len(val_dataset))

# 数据加载器，用于批量加载训练和验证数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
print(train_loader, val_loader)

# 使用预训练的ResNet34作为编码器，搭建UNet模型
model = smp.Unet(
    encoder_name=Config.ENCODER_NAME,
    encoder_weights=Config.ENCODER_WEIGHTS,
    in_channels=Config.IN_CHANNELS,
    classes=Config.CLASSES,
    activation=Config.ACTIVATION
)

# 设置计算设备（如果有GPU，则使用GPU）
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# 使用Adam优化器，学习率为0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

# 初始化最小损失和训练周期计数器
loss_lowest = 100
num = 0
divs = len(train_loader)

# 训练指定周期数
for epoch in range(Config.EPOCHS):
    loss_per_epoch = 0
    bat_no = 0
    dicel1 = [0, 0, 0, 0]  # Dice系数初始化，用于存储每个类别的Dice系数

    # 训练模式下进行每个批次的训练
    for data in tqdm.tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{Config.EPOCHS}', unit='batch'):
        bat_no += 1
        model.train()  # 设置模型为训练模式
        inputs, labels = data  # 获取输入和标签
        labels = labels.long()  # 确保标签是long类型
        inputs = inputs.to(device)  # 将输入数据转移到计算设备
        labels = labels.to(device)  # 将标签数据转移到计算设备

        # 模型前向传播
        outputs = model(inputs.float())

        # 计算损失
        loss = loss_funct(outputs.float(), labels, Config.CLASSES, dicel1)

        # 反向传播，优化器更新
        optimizer.zero_grad()  # 清除之前的梯度
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

        # 累加每个epoch的损失
        loss_per_epoch += loss.item()

    # 输出每个epoch的平均损失和Dice系数
    print("Loss for Epoch:{} Loss:{}".format(epoch + 1, loss_per_epoch / divs))
    for i in range(Config.CLASSES):
        print("Dice coeff for class ", i, " = ", dicel1[i] / divs)

    # 进行验证
    model.eval()  # 设置模型为验证模式
    dicel1 = [0, 0, 0, 0]
    loss_per_epoch = 0
    with torch.no_grad():  # 不计算梯度
        valo = 0
        for val_data in tqdm.tqdm(val_loader, desc=f'Validation Epoch {epoch + 1}/{Config.EPOCHS}', unit='batch'):
            val_input, val_mask = val_data  # 获取验证集的输入和掩膜
            valo = valo + 1
            val_mask = val_mask.long()  # 确保标签是long类型
            val_input = val_input.to(device)  # 将输入数据转移到计算设备
            val_mask = val_mask.to(device)  # 将标签数据转移到计算设备

            # 模型前向传播
            out = model(val_input.float())

            # 计算损失
            loss = loss_funct(outputs.float(), labels, Config.CLASSES, dicel1)
            loss_per_epoch += loss.item()

    # 输出验证集的损失和Dice系数
    print("Validation Loss for Epoch:{} Loss:{}".format(epoch + 1, loss_per_epoch / len(val_loader)))
    for i in range(Config.CLASSES):
        print("Dice coeff for class ", i, " = ", dicel1[i] / len(val_loader))

    # 如果当前验证损失更低，则保存当前最佳模型
    if loss_per_epoch / len(val_loader) < loss_lowest:
        loss_lowest = loss_per_epoch / len(val_loader)
        torch.save(model, 'best_model.pth')  # 保存模型
        num = 0
        print('saved')
    else:
        num = num + 1
        if num > Config.EARLY_STOPPING_PATIENCE:  # 如果连续指定个epoch没有提升，则停止训练
            print("Breaking")
            break