import torch
import torch.nn.functional as F

# 定义多类Dice系数函数
def multiclass_dice(pred, target):
    smooth = 1e-8  # 防止除零错误，添加平滑项
    target = F.one_hot(target, num_classes=4)  # 将目标转换为独热编码形式，num_classes=4表示4个类别

    # 计算每个类别的预测值总和
    a_sum = torch.sum(pred, dim=3)
    a_sum = torch.sum(a_sum, dim=2)
    a_sum = torch.sum(a_sum, dim=0).view(-1)  # 展平为1D张量

    # 计算每个类别的真实标签总和
    b_sum = torch.sum(target, dim=1)
    b_sum = torch.sum(b_sum, dim=1)
    b_sum = torch.sum(b_sum, dim=0).view(-1)  # 展平为1D张量

    # 将目标张量从[batch, channels, height, width]的形状变为[batch, height, width, channels]
    reshaped_target = torch.permute(target, (0, 3, 1, 2))

    # 计算交集部分
    intersect = pred * reshaped_target

    # 计算交集的总和
    intersect_sum = torch.sum(intersect, dim=3)
    intersect_sum = torch.sum(intersect_sum, dim=2)
    intersect_sum = torch.sum(intersect_sum, dim=0).view(-1)  # 展平为1D张量

    # 计算每个类别的Dice系数，注意使用平滑项防止分母为0
    class_wise_dice = torch.div(intersect_sum * 2 + smooth, a_sum + b_sum + smooth)

    # 清除中间变量以节省内存
    del(a_sum)
    del(b_sum)
    del(reshaped_target)
    del(intersect)
    del(intersect_sum)

    # 返回每个类别的Dice系数
    return class_wise_dice

# 定义损失函数
def loss_funct(pred, mask, num_classes, dice):
    # 计算交叉熵损失
    x = torch.nn.CrossEntropyLoss()(pred, mask)

    # 计算Dice系数
    y = multiclass_dice(pred, mask)

    # 累积每个类别的Dice系数
    l = 0
    for i in range(num_classes):
        dice[i] += y[i]

    # 计算总损失
    l = torch.sum(y, dim=0)
    l = torch.sub(4, l)  # 目标是最大化Dice系数，因此总损失是4减去每个类别的Dice系数

    return l
