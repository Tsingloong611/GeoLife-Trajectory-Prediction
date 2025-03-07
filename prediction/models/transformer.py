"""
基于Transformer的轨迹预测模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random

from .base_model import BaseTrajectoryModel
from ..config import DEVICE


class PositionalEncoding(nn.Module):
    """位置编码模块"""

    def __init__(self, d_model, max_len=5000, dropout=0.1):
        """
        初始化位置编码

        参数:
        d_model (int): 模型维度
        max_len (int): 最大序列长度
        dropout (float): Dropout比率
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        # 注册为非训练参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        应用位置编码

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)

        返回:
        torch.Tensor: 添加位置编码后的张量
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码模块（适用于地理坐标）"""

    def __init__(self, d_model, dropout=0.1, base_freq=1 / 10000):
        """
        初始化正弦位置编码

        参数:
        d_model (int): 模型维度
        dropout (float): Dropout比率
        base_freq (float): 基础频率
        """
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.base_freq = base_freq

    def forward(self, x, positions=None):
        """
        应用位置编码，可以使用特定的位置值（如地理坐标）

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_model)
        positions (torch.Tensor, optional): 位置值，形状为 (batch_size, seq_len, 2)
                                          对于轨迹数据，这是经纬度坐标

        返回:
        torch.Tensor: 添加位置编码后的张量
        """
        batch_size, seq_len, _ = x.size()

        # 如果没有提供位置值，则使用序列位置
        if positions is None:
            positions = torch.arange(0, seq_len, device=x.device).float().unsqueeze(0).repeat(batch_size, 1)
            # 扩展为 (batch_size, seq_len, 2)，复制同一个值用于两个坐标维度
            positions = positions.unsqueeze(-1).repeat(1, 1, 2)

        # 为每个位置创建编码
        pe = torch.zeros(batch_size, seq_len, self.d_model, device=x.device)

        # 对每个batch单独处理
        for b in range(batch_size):
            for pos in range(seq_len):
                # 获取当前位置的坐标
                lat, lon = positions[b, pos, 0], positions[b, pos, 1]

                # 计算不同频率的正弦和余弦
                for i in range(0, self.d_model, 2):
                    freq = self.base_freq ** (i / self.d_model)
                    pe[b, pos, i] = math.sin(lat * freq)
                    pe[b, pos, i + 1] = math.cos(lon * freq) if i + 1 < self.d_model else math.sin(lon * freq)

        # 将位置编码添加到输入
        x = x + pe
        return self.dropout(x)


class TransformerTrajectoryPredictor(BaseTrajectoryModel):
    """基于Transformer的轨迹预测模型"""

    def __init__(self, input_size, output_size=2, hidden_size=128, nhead=8,
                 num_encoder_layers=3, num_decoder_layers=3, dim_feedforward=512,
                 dropout=0.1, target_len=None, pos_encoding_type='standard',
                 device=DEVICE):
        """
        初始化Transformer轨迹预测模型

        参数:
        input_size (int): 输入特征维度
        output_size (int): 输出特征维度，默认为2 (lat, lon)
        hidden_size (int): 隐藏层大小
        nhead (int): 多头注意力的头数
        num_encoder_layers (int): 编码器层数
        num_decoder_layers (int): 解码器层数
        dim_feedforward (int): 前馈网络的维度
        dropout (float): Dropout比率
        target_len (int): 目标序列长度，如果为None则在forward中从target推断
        pos_encoding_type (str): 位置编码类型，可选['standard', 'sinusoidal']
        device (torch.device): 计算设备
        """
        super(TransformerTrajectoryPredictor, self).__init__(input_size, output_size, device)

        self.hidden_size = hidden_size
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.target_len = target_len
        self.pos_encoding_type = pos_encoding_type

        # 输入投影层
        self.input_projection = nn.Linear(input_size, hidden_size)

        # 添加目标投影层
        self.target_projection = nn.Linear(output_size, hidden_size)

        # 输出投影层
        self.output_projection = nn.Linear(hidden_size, output_size)

        # 位置编码
        if pos_encoding_type == 'standard':
            self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)
        elif pos_encoding_type == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(hidden_size, dropout=dropout)
        else:
            raise ValueError(f"未知的位置编码类型: {pos_encoding_type}")

        # Transformer模型
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # 将模型移到指定设备
        self.to(device)

    def forward(self, x, target=None, teacher_forcing_ratio=0.5):
        """
        前向传播

        参数:
        x (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, input_size)
        target (torch.Tensor, optional): 目标序列，用于教师强制，形状为 (batch_size, target_len, output_size)
        teacher_forcing_ratio (float): 教师强制比率

        返回:
        torch.Tensor: 预测序列，形状为 (batch_size, target_len, output_size)
        """
        batch_size = x.size(0)
        seq_len = x.size(1)

        # 确定目标序列长度
        target_len = self.target_len
        if target_len is None:
            target_len = target.size(1) if target is not None else 1

        # 将输入移动到设备
        x = x.to(self.device)
        if target is not None:
            target = target.to(self.device)

        # 输入投影到隐藏维度
        x = self.input_projection(x)

        # 应用位置编码
        x = self.pos_encoder(x)

        # 为Transformer解码器创建目标掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(target_len).to(self.device)

        # 创建初始输出序列（全零）
        if target is not None and random.random() < teacher_forcing_ratio:
            # 使用目标投影层而不是输入投影层
            tgt = self.target_projection(target)
        else:
            # 创建一个全零的初始目标序列
            tgt = torch.zeros(batch_size, target_len, self.hidden_size, device=self.device)

        # 应用目标位置编码
        tgt = self.pos_encoder(tgt)

        # 应用Transformer
        output = self.transformer(
            src=x,
            tgt=tgt,
            tgt_mask=tgt_mask
        )

        # 输出投影到输出维度
        output = self.output_projection(output)

        return output

    def predict(self, x):
        """
        预测函数

        参数:
        x (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, input_size)

        返回:
        torch.Tensor: 预测序列，形状为 (batch_size, target_len, output_size)
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.forward(x, teacher_forcing_ratio=0.0)
        return outputs

    def get_config(self):
        """获取模型配置"""
        return {
            'hidden_size': self.hidden_size,
            'nhead': self.nhead,
            'num_encoder_layers': self.num_encoder_layers,
            'num_decoder_layers': self.num_decoder_layers,
            'dim_feedforward': self.dim_feedforward,
            'dropout': self.dropout,
            'target_len': self.target_len,
            'pos_encoding_type': self.pos_encoding_type
        }