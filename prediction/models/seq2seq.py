"""
带注意力机制的Seq2Seq轨迹预测模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from .base_model import BaseTrajectoryModel
from ..config import DEVICE


class Encoder(nn.Module):
    """Seq2Seq编码器"""

    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0, bidirectional=False):
        """
        初始化编码器

        参数:
        input_size (int): 输入特征维度
        hidden_size (int): 隐藏层大小
        num_layers (int): RNN层数
        dropout (float): Dropout比率
        bidirectional (bool): 是否使用双向RNN
        """
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # GRU层
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        前向传播

        参数:
        x (torch.Tensor): 输入序列，形状为 (batch_size, seq_len, input_size)

        返回:
        tuple: (outputs, hidden)
            outputs: 所有时间步的输出，形状为 (batch_size, seq_len, hidden_size * D)
            hidden: 最后一个时间步的隐藏状态，形状为 (D * num_layers, batch_size, hidden_size)
            其中 D = 2 如果是双向RNN，否则 D = 1
        """
        # 应用dropout到输入
        x = self.dropout(x)

        # GRU前向传播
        outputs, hidden = self.gru(x)

        return outputs, hidden


class BahdanauAttention(nn.Module):
    """Bahdanau注意力机制"""

    def __init__(self, hidden_size):
        """
        初始化Bahdanau注意力

        参数:
        hidden_size (int): 隐藏层大小
        """
        super(BahdanauAttention, self).__init__()

        self.hidden_size = hidden_size

        # 注意力层
        self.W1 = nn.Linear(hidden_size, hidden_size)
        self.W2 = nn.Linear(hidden_size, hidden_size)
        self.V = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        """
        计算注意力权重

        参数:
        hidden (torch.Tensor): 解码器的当前隐藏状态，形状为 (batch_size, hidden_size)
        encoder_outputs (torch.Tensor): 编码器的所有输出，形状为 (batch_size, seq_len, hidden_size)

        返回:
        tuple: (attention_weights, context_vector)
            attention_weights: 注意力权重，形状为 (batch_size, seq_len, 1)
            context_vector: 上下文向量，形状为 (batch_size, hidden_size)
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # 将hidden扩展为与encoder_outputs相同的序列长度
        # hidden形状: (batch_size, 1, hidden_size)
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # 计算注意力能量
        # energy形状: (batch_size, seq_len, hidden_size)
        energy = torch.tanh(self.W1(hidden) + self.W2(encoder_outputs))

        # 计算注意力权重
        # attention_weights形状: (batch_size, seq_len, 1)
        attention_weights = F.softmax(self.V(energy), dim=1)

        # 计算上下文向量
        # context_vector形状: (batch_size, hidden_size)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs).squeeze(1)

        return attention_weights, context_vector


class LuongAttention(nn.Module):
    """Luong注意力机制"""

    def __init__(self, hidden_size, method='general'):
        """
        初始化Luong注意力

        参数:
        hidden_size (int): 隐藏层大小
        method (str): 注意力计算方法，可选['dot', 'general', 'concat']
        """
        super(LuongAttention, self).__init__()

        self.hidden_size = hidden_size
        self.method = method

        if method == 'general':
            self.Wa = nn.Linear(hidden_size, hidden_size)
        elif method == 'concat':
            self.Wa = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs):
        """
        计算注意力权重

        参数:
        hidden (torch.Tensor): 解码器的当前隐藏状态，形状为 (batch_size, hidden_size)
        encoder_outputs (torch.Tensor): 编码器的所有输出，形状为 (batch_size, seq_len, hidden_size)

        返回:
        tuple: (attention_weights, context_vector)
            attention_weights: 注意力权重，形状为 (batch_size, seq_len, 1)
            context_vector: 上下文向量，形状为 (batch_size, hidden_size)
        """
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # 计算注意力分数
        if self.method == 'dot':
            # hidden形状: (batch_size, hidden_size, 1)
            hidden = hidden.unsqueeze(2)
            # 注意力分数形状: (batch_size, seq_len, 1)
            attention_scores = torch.bmm(encoder_outputs, hidden)

        elif self.method == 'general':
            # hidden形状: (batch_size, hidden_size, 1)
            hidden = self.Wa(hidden).unsqueeze(2)
            # 注意力分数形状: (batch_size, seq_len, 1)
            attention_scores = torch.bmm(encoder_outputs, hidden)

        elif self.method == 'concat':
            # hidden形状: (batch_size, 1, hidden_size)
            hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            # energy形状: (batch_size, seq_len, hidden_size)
            energy = torch.tanh(self.Wa(torch.cat((hidden, encoder_outputs), dim=2)))
            # 注意力分数形状: (batch_size, seq_len, 1)
            attention_scores = self.v(energy)

        # 应用softmax获取注意力权重
        attention_weights = F.softmax(attention_scores, dim=1)

        # 计算上下文向量
        # context_vector形状: (batch_size, hidden_size)
        context_vector = torch.bmm(attention_weights.transpose(1, 2), encoder_outputs).squeeze(1)

        return attention_weights, context_vector


class AttentionDecoder(nn.Module):
    """带注意力机制的解码器"""

    def __init__(self, input_size, hidden_size, output_size, attention_type='bahdanau',
                num_layers=1, dropout=0):
        """
        初始化带注意力的解码器

        参数:
        input_size (int): 输入特征维度
        hidden_size (int): 隐藏层大小
        output_size (int): 输出特征维度
        attention_type (str): 注意力机制类型，可选['bahdanau', 'luong']
        num_layers (int): RNN层数
        dropout (float): Dropout比率
        """
        super(AttentionDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.attention_type = attention_type
        self.num_layers = num_layers
        self.dropout = dropout

        # 注意力机制
        if attention_type == 'bahdanau':
            self.attention = BahdanauAttention(hidden_size)
        elif attention_type == 'luong':
            self.attention = LuongAttention(hidden_size, method='general')
        else:
            raise ValueError(f"未知的注意力类型: {attention_type}")

        # 输入投影层
        self.embedding = nn.Linear(input_size, hidden_size)

        # GRU层
        self.gru = nn.GRU(
            input_size=hidden_size * 2,  # 输入 + 上下文向量
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # 输出层
        self.fc_out = nn.Linear(hidden_size * 2, output_size)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden, encoder_outputs):
        """
        前向传播

        参数:
        x (torch.Tensor): 输入张量，形状为 (batch_size, 1, input_size)
        hidden (torch.Tensor): 隐藏状态，形状为 (num_layers, batch_size, hidden_size)
        encoder_outputs (torch.Tensor): 编码器输出，形状为 (batch_size, seq_len, hidden_size)

        返回:
        tuple: (output, hidden, attention_weights)
            output: 当前时间步的输出，形状为 (batch_size, 1, output_size)
            hidden: 更新后的隐藏状态
            attention_weights: 注意力权重
        """
        # 投影输入到隐藏空间
        x = self.dropout(self.embedding(x))

        # 获取当前隐藏状态的最上层
        # 如果多层，取最后一层的隐藏状态
        top_hidden = hidden[-1]

        # 计算注意力和上下文向量
        attention_weights, context_vector = self.attention(top_hidden, encoder_outputs)

        # 将输入和上下文向量连接
        # context_vector形状: (batch_size, hidden_size)
        # x形状: (batch_size, 1, hidden_size)
        # rnn_input形状: (batch_size, 1, hidden_size * 2)
        context_vector = context_vector.unsqueeze(1)
        rnn_input = torch.cat((x, context_vector), dim=2)

        # GRU前向传播
        # output形状: (batch_size, 1, hidden_size)
        # hidden形状: (num_layers, batch_size, hidden_size)
        output, hidden = self.gru(rnn_input, hidden)

        # 将输出和上下文向量连接
        # output形状: (batch_size, 1, hidden_size)
        # combined形状: (batch_size, 1, hidden_size * 2)
        combined = torch.cat((output, context_vector), dim=2)

        # 最终输出
        # prediction形状: (batch_size, 1, output_size)
        prediction = self.fc_out(combined)

        return prediction, hidden, attention_weights


class Seq2SeqAttention(BaseTrajectoryModel):
    """带注意力机制的Seq2Seq轨迹预测模型"""

    def __init__(self, input_size, output_size=2, hidden_size=128, num_layers=2,
                dropout=0.2, bidirectional=True, attention_type='bahdanau',
                target_len=None, device=DEVICE):
        """
        初始化带注意力的Seq2Seq轨迹预测模型

        参数:
        input_size (int): 输入特征维度
        output_size (int): 输出特征维度，默认为2 (lat, lon)
        hidden_size (int): 隐藏层大小
        num_layers (int): RNN层数
        dropout (float): Dropout比率
        bidirectional (bool): 编码器是否使用双向RNN
        attention_type (str): 注意力机制类型，可选['bahdanau', 'luong']
        target_len (int): 目标序列长度，如果为None则在forward中从target推断
        device (torch.device): 计算设备
        """
        super(Seq2SeqAttention, self).__init__(input_size, output_size, device)

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.attention_type = attention_type
        self.target_len = target_len

        # 编码器
        self.encoder = Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # 如果编码器是双向的，则解码器的隐藏状态大小需要匹配
        decoder_hidden_size = hidden_size * 2 if bidirectional else hidden_size

        # 解码器
        self.decoder = AttentionDecoder(
            input_size=output_size,  # 解码器输入为上一步的输出
            hidden_size=decoder_hidden_size,
            output_size=output_size,
            attention_type=attention_type,
            num_layers=num_layers,
            dropout=dropout
        )

        # 处理双向编码器的隐藏状态
        if bidirectional:
            self.hidden_transform = nn.Linear(hidden_size * 2, hidden_size * 2)

        # 将模型移到指定设备
        self.to(device)

        # 存储注意力权重，用于可视化
        self.attention_weights = None

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
        encoder_outputs, hidden = self.encoder(x)

        # 如果编码器是双向的，需要处理隐藏状态
        if self.bidirectional:
            # hidden形状: (num_directions * num_layers, batch_size, hidden_size)
            # 我们需要连接正向和反向的隐藏状态
            hidden = hidden.view(self.num_layers, 2, batch_size, self.hidden_size)

            # 合并双向隐藏状态
            # 处理每一层的隐藏状态
            new_hidden = []
            for layer in range(self.num_layers):
                # 连接正向和反向的隐藏状态
                # 形状: (batch_size, hidden_size * 2)
                cat_hidden = torch.cat([hidden[layer, 0], hidden[layer, 1]], dim=1)
                new_hidden.append(cat_hidden)

            # 重新组织隐藏状态
            # 形状: (num_layers, batch_size, hidden_size * 2)
            hidden = torch.stack(new_hidden)

        # 准备解码器的第一个输入 (全零)
        decoder_input = torch.zeros(batch_size, 1, self.output_size, device=self.device)

        # 存储所有解码输出
        decoder_outputs = []
        all_attention_weights = []

        # 逐步解码
        for t in range(target_len):
            # 解码一个时间步
            decoder_output, hidden, attention_weights = self.decoder(decoder_input, hidden, encoder_outputs)
            decoder_outputs.append(decoder_output)
            all_attention_weights.append(attention_weights)

            # 教师强制：使用真实的目标作为下一步的输入
            teacher_force = random.random() < teacher_forcing_ratio

            if teacher_force and target is not None:
                decoder_input = target[:, t:t+1, :]
            else:
                # 否则使用预测作为下一步的输入
                decoder_input = decoder_output

        # 拼接所有时间步的输出
        decoder_outputs = torch.cat(decoder_outputs, dim=1)

        # 存储注意力权重，用于可视化
        self.attention_weights = torch.cat(all_attention_weights, dim=1)

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

    def get_attention_weights(self):
        """
        获取注意力权重

        返回:
        torch.Tensor: 注意力权重，形状为 (batch_size, target_len, seq_len)
        """
        return self.attention_weights

    def get_config(self):
        """获取模型配置"""
        return {
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'attention_type': self.attention_type,
            'target_len': self.target_len
        }