"""
轨迹数据加载工具
"""
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import GroupShuffleSplit

from .dataset import TrajectoryDataset
from ..config import (
    DATA_DIR, PREDICTION_SAMPLES_DIR, FEATURES_DIR,
    BATCH_SIZE, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
)


def load_prediction_samples(input_length=None, output_length=None, file_path=None):
    """
    加载预测样本数据

    参数:
    input_length (int): 输入长度，如果file_path为None则用于构建文件名
    output_length (int): 输出长度，如果file_path为None则用于构建文件名
    file_path (str): 可选，直接指定样本文件路径

    返回:
    pd.DataFrame: 包含预测样本的DataFrame
    """
    if file_path is None:
        if input_length is None or output_length is None:
            raise ValueError("必须指定input_length和output_length，或者提供file_path")

        file_name = f"prediction_samples_in{input_length}_out{output_length}.csv"
        file_path = os.path.join(PREDICTION_SAMPLES_DIR, file_name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"无法找到样本文件: {file_path}")

    # 加载样本数据
    samples_df = pd.read_csv(file_path)

    # 检查格式是否正确
    required_prefixes = ['in_', 'out_lat_', 'out_lon_']
    for prefix in required_prefixes:
        if not any(col.startswith(prefix) for col in samples_df.columns):
            raise ValueError(f"样本数据格式不正确，缺少{prefix}前缀的列")

    return samples_df


def load_selected_features(file_path=None):
    """
    加载选定的特征列表

    参数:
    file_path (str): 特征列表文件路径，默认为features/selected_features.csv

    返回:
    list: 特征名称列表
    """
    if file_path is None:
        file_path = os.path.join(FEATURES_DIR, "selected_features.csv")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"无法找到特征文件: {file_path}")

    features_df = pd.read_csv(file_path)

    # 检查格式是否正确
    if 'feature' not in features_df.columns:
        raise ValueError("特征文件格式不正确，缺少feature列")

    return features_df['feature'].tolist()


def create_dataloaders(samples_df, selected_features=None, batch_size=BATCH_SIZE,
                       train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, test_ratio=TEST_RATIO,
                       input_length=None, output_length=None, normalize=True, random_seed=42):
    """
    创建训练、验证和测试数据加载器

    参数:
    samples_df (pd.DataFrame): 预测样本数据
    selected_features (list): 选定的特征列表
    batch_size (int): 批大小
    train_ratio (float): 训练集比例
    val_ratio (float): 验证集比例
    test_ratio (float): 测试集比例
    input_length (int): 输入序列长度
    output_length (int): 输出序列长度
    normalize (bool): 是否标准化数据
    random_seed (int): 随机种子

    返回:
    tuple: (train_loader, val_loader, test_loader, dataset)
    """
    # 检查比例总和是否为1
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-5:
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1")

    # 创建数据集
    dataset = TrajectoryDataset(
        samples_df,
        selected_features=selected_features,
        input_length=input_length,
        output_length=output_length,
        normalize=normalize
    )

    # 按用户和轨迹ID分组划分数据集，确保同一轨迹的点不会分散到不同数据集
    groups = samples_df['user_id'].astype(str) + '_' + samples_df['traj_id'].astype(str)

    # 训练集和剩余部分的划分
    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_ratio, random_state=random_seed)
    train_idx, temp_idx = next(gss1.split(X=samples_df, groups=groups))

    # 验证集和测试集的划分
    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)  # 调整验证集比例
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_ratio_adjusted, random_state=random_seed)
    val_idx, test_idx = next(gss2.split(X=samples_df.iloc[temp_idx], groups=groups.iloc[temp_idx]))

    # 将相对索引转换为绝对索引
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    # 创建采样器
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    # 创建数据加载器
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,  # 可以根据系统性能调整
        pin_memory=True
    )

    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler,
        num_workers=0,
        pin_memory=True
    )

    print(f"数据集划分完成: 训练集={len(train_idx)}样本, 验证集={len(val_idx)}样本, 测试集={len(test_idx)}样本")

    return train_loader, val_loader, test_loader, dataset