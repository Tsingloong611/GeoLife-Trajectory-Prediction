"""
轨迹预测的损失函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def haversine_loss(y_pred, y_true, reduction='mean'):
    """
    Haversine距离损失函数，适用于地理坐标

    参数:
    y_pred (torch.Tensor): 预测的坐标，形状为 (batch_size, seq_len, 2)
    y_true (torch.Tensor): 真实的坐标，形状为 (batch_size, seq_len, 2)
    reduction (str): 损失计算方式，可选['mean', 'sum', 'none']

    返回:
    torch.Tensor: Haversine距离损失
    """
    batch_size, seq_len, _ = y_pred.size()

    # 提取经纬度
    pred_lat, pred_lon = y_pred[:, :, 0], y_pred[:, :, 1]
    true_lat, true_lon = y_true[:, :, 0], y_true[:, :, 1]

    # 转换为弧度
    pred_lat = torch.deg2rad(pred_lat)
    pred_lon = torch.deg2rad(pred_lon)
    true_lat = torch.deg2rad(true_lat)
    true_lon = torch.deg2rad(true_lon)

    # 计算Haversine距离
    dlat = true_lat - pred_lat
    dlon = true_lon - pred_lon

    # Haversine公式
    R = 6371000.0  # 地球半径(米)
    a = torch.sin(dlat / 2) ** 2 + torch.cos(pred_lat) * torch.cos(true_lat) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distance = R * c  # 单位：米

    # 应用损失计算方式
    if reduction == 'mean':
        return torch.mean(distance)
    elif reduction == 'sum':
        return torch.sum(distance)
    else:  # 'none'
        return distance


def combined_loss(y_pred, y_true, mse_weight=1.0, haversine_weight=1.0, reduction='mean'):
    """
    组合损失函数：MSE + Haversine

    参数:
    y_pred (torch.Tensor): 预测的坐标，形状为 (batch_size, seq_len, 2)
    y_true (torch.Tensor): 真实的坐标，形状为 (batch_size, seq_len, 2)
    mse_weight (float): MSE损失的权重
    haversine_weight (float): Haversine损失的权重
    reduction (str): 损失计算方式，可选['mean', 'sum', 'none']

    返回:
    tuple: (总损失, MSE损失, Haversine损失)
    """
    # MSE损失
    mse_loss = F.mse_loss(y_pred, y_true, reduction=reduction)

    # Haversine损失
    hav_loss = haversine_loss(y_pred, y_true, reduction=reduction)

    # 归一化Haversine损失（单位为米，数值可能较大）
    # 这里除以1000将距离转换为公里，以使两种损失的量级更接近
    hav_loss = hav_loss / 1000.0

    # 组合损失
    if reduction == 'none':
        total_loss = mse_weight * mse_loss + haversine_weight * hav_loss
    else:
        total_loss = mse_weight * mse_loss + haversine_weight * hav_loss

    return total_loss, mse_loss, hav_loss


class CustomLoss(nn.Module):
    """可配置的自定义损失函数类"""

    def __init__(self, mse_weight=1.0, haversine_weight=1.0, reduction='mean'):
        """
        初始化自定义损失函数

        参数:
        mse_weight (float): MSE损失的权重
        haversine_weight (float): Haversine损失的权重
        reduction (str): 损失计算方式，可选['mean', 'sum', 'none']
        """
        super(CustomLoss, self).__init__()
        self.mse_weight = mse_weight
        self.haversine_weight = haversine_weight
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        """
        计算损失

        参数:
        y_pred (torch.Tensor): 预测的坐标，形状为 (batch_size, seq_len, 2)
        y_true (torch.Tensor): 真实的坐标，形状为 (batch_size, seq_len, 2)

        返回:
        torch.Tensor: 总损失
        """
        return combined_loss(
            y_pred, y_true,
            self.mse_weight, self.haversine_weight,
            self.reduction
        )[0]  # 只返回总损失