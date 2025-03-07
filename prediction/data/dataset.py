"""
轨迹预测数据集类
"""
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """轨迹预测数据集类"""

    def __init__(self, samples_df, feature_columns=None, selected_features=None,
                 input_length=None, output_length=None, normalize=True):
        """
        初始化轨迹预测数据集

        参数:
        samples_df (pd.DataFrame): 包含预测样本的DataFrame
        feature_columns (list): 输入特征列名列表，如果为None则自动推断
        selected_features (list): 要使用的特征子集，如果为None则使用全部feature_columns
        input_length (int): 输入序列长度，如果为None则自动推断
        output_length (int): 输出序列长度，如果为None则自动推断
        normalize (bool): 是否对特征进行标准化
        """
        self.samples_df = samples_df
        self.normalize = normalize

        # 自动推断输入和输出长度
        if input_length is None or output_length is None:
            # 找出所有以 'in_' 开头的列，提取数字部分
            in_cols = [col for col in samples_df.columns if col.startswith('in_')]
            in_indices = set()
            for col in in_cols:
                parts = col.split('_')
                if len(parts) >= 3 and parts[-1].isdigit():
                    in_indices.add(int(parts[-1]))

            # 找出所有以 'out_lat_' 或 'out_lon_' 开头的列，提取数字部分
            out_cols = [col for col in samples_df.columns if col.startswith('out_lat_') or col.startswith('out_lon_')]
            out_indices = set()
            for col in out_cols:
                parts = col.split('_')
                if len(parts) >= 3 and parts[-1].isdigit():
                    out_indices.add(int(parts[-1]))

            # 设置输入和输出长度
            self.input_length = input_length or (max(in_indices) + 1 if in_indices else 0)
            self.output_length = output_length or (max(out_indices) + 1 if out_indices else 0)
        else:
            self.input_length = input_length
            self.output_length = output_length

        # 确定特征列名
        if feature_columns is None:
            # 自动推断特征列名（除了user_id, traj_id, start_time, end_time）
            base_feature_cols = set()
            for col in samples_df.columns:
                if col.startswith('in_'):
                    parts = col.split('_')
                    if len(parts) >= 3 and parts[-1].isdigit():
                        # 去掉索引部分，保留特征名
                        base_feature = '_'.join(parts[1:-1])
                        base_feature_cols.add(base_feature)
            self.base_feature_cols = list(base_feature_cols)
        else:
            self.base_feature_cols = feature_columns

        # 如果指定了特征子集，则使用特征子集
        if selected_features is not None:
            self.base_feature_cols = [f for f in self.base_feature_cols if f in selected_features]

        # 创建输入特征列名
        self.input_columns = []
        for t in range(self.input_length):
            for feat in self.base_feature_cols:
                self.input_columns.append(f'in_{feat}_{t}')

        # 创建输出列名
        self.output_columns = []
        for t in range(self.output_length):
            self.output_columns.append(f'out_lat_{t}')
            self.output_columns.append(f'out_lon_{t}')

        # 数据预处理
        self._preprocess_data()

    def _preprocess_data(self):
        """预处理数据，包括标准化和缺失值处理"""
        # 提取输入特征和输出目标
        X = self.samples_df[self.input_columns].values
        y = self.samples_df[self.output_columns].values

        # 填充缺失值
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        # 标准化处理
        if self.normalize:
            # 对输入特征进行标准化
            self.X_mean = np.mean(X, axis=0)
            self.X_std = np.std(X, axis=0)
            self.X_std[self.X_std == 0] = 1.0  # 避免除以0
            X = (X - self.X_mean) / self.X_std

            # 对输出目标不进行标准化，保留原始经纬度
            # (因为我们希望预测真实地理坐标，而标准化会改变数值含义)

        # 转换为张量
        self.X_tensor = torch.FloatTensor(X)
        self.y_tensor = torch.FloatTensor(y)

        # 将输入特征重塑为 (样本数, 时间步, 特征数)
        self.X_tensor = self.X_tensor.view(-1, self.input_length, len(self.base_feature_cols))

        # 将输出目标重塑为 (样本数, 输出时间步, 2) - [lat, lon]
        self.y_tensor = self.y_tensor.view(-1, self.output_length, 2)

    def __len__(self):
        """返回数据集大小"""
        return len(self.samples_df)

    def __getitem__(self, idx):
        """返回单个样本"""
        return {
            'input': self.X_tensor[idx],
            'target': self.y_tensor[idx],
            'user_id': self.samples_df.iloc[idx]['user_id'],
            'traj_id': self.samples_df.iloc[idx]['traj_id'],
            'start_time': pd.to_datetime(self.samples_df.iloc[idx]['start_time']).timestamp(),
            'end_time': pd.to_datetime(self.samples_df.iloc[idx]['end_time']).timestamp()
        }

    def inverse_normalize(self, normalized_data, is_input=True):
        """反标准化数据"""
        if not self.normalize:
            return normalized_data

        if is_input:
            # 输入特征的反标准化
            return normalized_data * self.X_std + self.X_mean
        else:
            # 输出目标不需要反标准化
            return normalized_data

    def get_feature_info(self):
        """返回特征信息"""
        return {
            'base_features': self.base_feature_cols,
            'input_length': self.input_length,
            'output_length': self.output_length,
            'normalized': self.normalize
        }