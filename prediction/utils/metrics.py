"""
轨迹预测的评估指标
"""
import torch
import numpy as np
from scipy import stats
import math


def ade_metric(pred, true):
    """
    平均位移误差 (Average Displacement Error)

    参数:
    pred (torch.Tensor): 预测的轨迹，形状为 (batch_size, seq_len, 2)
    true (torch.Tensor): 真实的轨迹，形状为 (batch_size, seq_len, 2)

    返回:
    float: 平均位移误差（米）
    """
    # 计算欧氏距离
    diff = pred - true
    euclidean = torch.sqrt(torch.sum(diff ** 2, dim=2))

    # 计算平均误差
    return torch.mean(euclidean).item()


def fde_metric(pred, true):
    """
    最终位移误差 (Final Displacement Error)

    参数:
    pred (torch.Tensor): 预测的轨迹，形状为 (batch_size, seq_len, 2)
    true (torch.Tensor): 真实的轨迹，形状为 (batch_size, seq_len, 2)

    返回:
    float: 最终位移误差（米）
    """
    # 取最后一个时间步
    pred_final = pred[:, -1, :]
    true_final = true[:, -1, :]

    # 计算欧氏距离
    diff = pred_final - true_final
    euclidean = torch.sqrt(torch.sum(diff ** 2, dim=1))

    # 计算平均最终误差
    return torch.mean(euclidean).item()


def haversine_metric(pred, true):
    """
    Haversine距离误差（地理空间距离）

    参数:
    pred (torch.Tensor): 预测的轨迹，形状为 (batch_size, seq_len, 2)
    true (torch.Tensor): 真实的轨迹，形状为 (batch_size, seq_len, 2)

    返回:
    float: Haversine距离误差（米）
    """
    batch_size, seq_len, _ = pred.size()

    # 提取经纬度
    pred_lat, pred_lon = pred[:, :, 0], pred[:, :, 1]
    true_lat, true_lon = true[:, :, 0], true[:, :, 1]

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

    # 计算平均距离
    return torch.mean(distance).item()


def direction_metric(pred, true, threshold_degrees=20):
    """
    方向预测准确度

    参数:
    pred (torch.Tensor): 预测的轨迹，形状为 (batch_size, seq_len, 2)
    true (torch.Tensor): 真实的轨迹，形状为 (batch_size, seq_len, 2)
    threshold_degrees (float): 角度阈值，小于此阈值认为方向预测准确

    返回:
    float: 方向预测准确率 (0-1)
    """
    batch_size, seq_len, _ = pred.size()

    if seq_len < 2:
        return 0.0

    # 计算相邻点之间的方向角
    def calculate_direction_angles(traj):
        angles = []

        for i in range(seq_len - 1):
            # 提取相邻两个点
            p1_lat, p1_lon = traj[:, i, 0], traj[:, i, 1]
            p2_lat, p2_lon = traj[:, i + 1, 0], traj[:, i + 1, 1]

            # 转换为弧度
            p1_lat = torch.deg2rad(p1_lat)
            p1_lon = torch.deg2rad(p1_lon)
            p2_lat = torch.deg2rad(p2_lat)
            p2_lon = torch.deg2rad(p2_lon)

            # 计算方位角（与正北方向的夹角）
            y = torch.sin(p2_lon - p1_lon) * torch.cos(p2_lat)
            x = torch.cos(p1_lat) * torch.sin(p2_lat) - \
                torch.sin(p1_lat) * torch.cos(p2_lat) * torch.cos(p2_lon - p1_lon)

            bearing = torch.atan2(y, x)
            bearing = torch.rad2deg(bearing)
            bearing = (bearing + 360) % 360  # 转换为0-360范围

            angles.append(bearing)

        return torch.stack(angles, dim=1)

    # 计算预测和真实轨迹的方向角
    pred_angles = calculate_direction_angles(pred)
    true_angles = calculate_direction_angles(true)

    # 计算角度差异
    angle_diff = torch.abs(pred_angles - true_angles)
    angle_diff = torch.min(angle_diff, 360 - angle_diff)  # 取最小角度差

    # 计算准确率：角度差小于阈值的比例
    accuracy = torch.mean((angle_diff < threshold_degrees).float()).item()

    return accuracy


def compute_metrics(pred, true, with_confidence=True):
    """
    计算所有指标

    参数:
    pred (torch.Tensor): 预测的轨迹，形状为 (batch_size, seq_len, 2)
    true (torch.Tensor): 真实的轨迹，形状为 (batch_size, seq_len, 2)
    with_confidence (bool): 是否计算置信区间

    返回:
    dict: 包含各项指标的字典
    """
    batch_size = pred.size(0)

    # 计算每个样本的指标
    ade_values = []
    fde_values = []
    haversine_values = []

    for i in range(batch_size):
        # 提取单个样本
        pred_sample = pred[i:i + 1]
        true_sample = true[i:i + 1]

        # 计算指标
        ade_values.append(ade_metric(pred_sample, true_sample))
        fde_values.append(fde_metric(pred_sample, true_sample))
        haversine_values.append(haversine_metric(pred_sample, true_sample))

    # 方向准确度
    direction_acc = direction_metric(pred, true)

    # 计算平均值
    metrics = {
        'ade': np.mean(ade_values),
        'fde': np.mean(fde_values),
        'haversine': np.mean(haversine_values),
        'direction_accuracy': direction_acc
    }

    # 计算置信区间
    if with_confidence and batch_size > 1:
        # 95%置信区间
        confidence = 0.95

        # ADE置信区间
        ade_ci = stats.t.interval(
            confidence,
            len(ade_values) - 1,
            loc=np.mean(ade_values),
            scale=stats.sem(ade_values)
        )

        # FDE置信区间
        fde_ci = stats.t.interval(
            confidence,
            len(fde_values) - 1,
            loc=np.mean(fde_values),
            scale=stats.sem(fde_values)
        )

        # Haversine置信区间
        haversine_ci = stats.t.interval(
            confidence,
            len(haversine_values) - 1,
            loc=np.mean(haversine_values),
            scale=stats.sem(haversine_values)
        )

        # 添加置信区间到指标字典
        metrics.update({
            'ade_ci_lower': ade_ci[0],
            'ade_ci_upper': ade_ci[1],
            'fde_ci_lower': fde_ci[0],
            'fde_ci_upper': fde_ci[1],
            'haversine_ci_lower': haversine_ci[0],
            'haversine_ci_upper': haversine_ci[1]
        })

    return metrics