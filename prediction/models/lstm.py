"""
LSTM轨迹预测模型
"""
import torch
import torch.nn as nn
import random
import numpy as np

from .base_model import BaseTrajectoryModel
from ..config import DEVICE


class LSTMEncoder(nn.Module):
    """LSTM编码器"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, bidirectional=False):
        """
        初始化LSTM编码器

        参数:
        input_size (int): 输入特征维度
        hidden_size (int): 隐藏层大小
        num_layers (int): LSTM层数
        dropout (float): Dropout比率
        bidirectional (bool): 是否使用双向LSTM
        """
        super(LSTMEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

    def forward(self, x):
        """
        前向传播

        参数:
        x (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, input_size)

        返回:
        tuple: (outputs, hidden)
            outputs: 所有时间步的输出，形状为 (batch_size, seq_len, hidden_size * D)
            hidden: 最后一个时间步的隐藏状态，形状为 (D * num_layers, batch_size, hidden_size)
            其中 D = 2 如果是双向LSTM，否则 D = 1
        """
        # 输出形状: (batch_size, seq_len, hidden_size * D)
        # 隐藏状态形状: (D * num_layers, batch_size, hidden_size)
        outputs, hidden = self.lstm(x)
        return outputs, hidden


class LSTMDecoder(nn.Module):
    """LSTM解码器"""

    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0):
        """
        初始化LSTM解码器

        参数:
        input_size (int): 输入特征维度
        hidden_size (int): 隐藏层大小
        output_size (int): 输出特征维度
        num_layers (int): LSTM层数
        dropout (float): Dropout比率
        """
        super(LSTMDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 输出层
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        """
        前向传播

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, 1, input_size)
        hidden (tuple): 隐藏状态和单元状态，来自编码器或上一步解码

        返回:
        tuple: (output, hidden)
            output: 当前时间步的输出，形状为 (batch_size, 1, output_size)
            hidden: 更新后的隐藏状态
        """
        # 输出形状: (batch_size, 1, hidden_size)
        output, hidden = self.lstm(x, hidden)

        # 应用全连接层得到预测
        # 输出形状: (batch_size, 1, output_size)
        prediction = self.out(output)

        return prediction, hidden


class LSTMTrajectoryPredictor(BaseTrajectoryModel):
    """基于LSTM的轨迹预测模型"""

    def __init__(self, input_size, output_size=2, hidden_size=128, num_layers=2,
                dropout=0.2, bidirectional=True, target_len=None, device=DEVICE):
        """
        初始化LSTM轨迹预测模型

        参数:
        input_size (int): 输入特征维度
        output_size (int): 输出特征维度，默认为2 (lat, lon)
        hidden_size (int): 隐藏层大小
        num_layers (int): LSTM层数
        dropout (float): Dropout比率
        bidirectional (bool): 编码器是否使用双向LSTM
        target_len (int): 目标序列长度，如果为None则在forward中从target推断
        device (torch.device): 计算设备
        """
        super(LSTMTrajectoryPredictor, self).__init__(input_size, output_size, device)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.target_len = target_len

        # 编码器
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # 解码器输入大小
        # 如果编码器是双向的，则解码器的输入大小需要匹配
        decoder_hidden_size = hidden_size * 2 if bidirectional else hidden_size

        # 解码器
        self.decoder = LSTMDecoder(
            input_size=output_size,  # 解码器输入为上一步的输出
            hidden_size=decoder_hidden_size,
            output_size=output_size,
            num_layers=num_layers,
            dropout=dropout
        )

        # 如果编码器是双向的，需要将隐藏状态转换为单向的
        if bidirectional:
            self.hidden_transform = nn.Linear(hidden_size * 2, hidden_size)

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

        # 确定目标序列长度
        target_len = self.target_len
        if target_len is None:
            target_len = target.size(1) if target is not None else 1

        # 将输入移动到设备
        x = x.to(self.device)
        if target is not None:
            target = target.to(self.device)

        # 编码输入序列
        _, hidden = self.encoder(x)

        # 如果编码器是双向的，需要处理隐藏状态
        if self.bidirectional:
            # 调整隐藏状态，将前向和后向的状态连接
            h_n, c_n = hidden

            # h_n形状：(D * num_layers, batch_size, hidden_size)
            # 我们需要将D(方向)维度的状态合并
            h_n = h_n.view(self.num_layers, 2, batch_size, self.hidden_size)
            c_n = c_n.view(self.num_layers, 2, batch_size, self.hidden_size)

            # 合并双向状态
            h_n = torch.cat([h_n[:, 0, :, :], h_n[:, 1, :, :]], dim=2)
            c_n = torch.cat([c_n[:, 0, :, :], c_n[:, 1, :, :]], dim=2)

            hidden = (h_n, c_n)

        # 准备解码器的第一个输入 (全零)
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=self.device)

        # 存储所有解码输出
        decoder_outputs = []

        # 逐步解码
        for t in range(target_len):
            # 解码一个时间步
            decoder_output, hidden = self.decoder(decoder_input, hidden)
            decoder_outputs.append(decoder_output)

            # 教师强制：使用真实的目标作为下一步的输入
            teacher_force = random.random() < teacher_forcing_ratio

            if teacher_force and target is not None:
                decoder_input = target[:, t:t+1, :]
            else:
                # 否则使用预测作为下一步的输入
                decoder_input = decoder_output

        # 拼接所有时间步的输出
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        return decoder_outputs

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
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'target_len': self.target_len
        }