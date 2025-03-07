"""
基于图卷积网络的时空轨迹预测模型 (修复版)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from .base_model import BaseTrajectoryModel
from ..config import DEVICE


class GraphConvolution(nn.Module):
    """图卷积层"""

    def __init__(self, in_features, out_features, bias=True):
        """
        初始化图卷积层

        参数:
        in_features (int): 输入特征维度
        out_features (int): 输出特征维度
        bias (bool): 是否使用偏置
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # 权重和偏置
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """重置参数"""
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        """
        前向传播

        参数:
        x (torch.Tensor): 节点特征矩阵，形状为 (batch_size, num_nodes, in_features)
        adj (torch.Tensor): 邻接矩阵，形状为 (batch_size, num_nodes, num_nodes)

        返回:
        torch.Tensor: 更新后的节点特征，形状为 (batch_size, num_nodes, out_features)
        """
        batch_size, num_nodes, _ = x.size()

        # 特征变换
        x = torch.matmul(x, self.weight)  # (batch_size, num_nodes, out_features)

        # 应用图卷积
        output = torch.matmul(adj, x)  # (batch_size, num_nodes, out_features)

        # 添加偏置
        if self.bias is not None:
            output = output + self.bias

        return output


class TemporalConvolution(nn.Module):
    """时间卷积层"""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        """
        初始化时间卷积层

        参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核大小
        padding (int): 填充大小
        dilation (int): 扩张率
        """
        super(TemporalConvolution, self).__init__()

        # 一维卷积层
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        )

    def forward(self, x):
        """
        前向传播

        参数:
        x (torch.Tensor): 输入序列，形状为 (batch_size, num_nodes, in_channels, seq_len)

        返回:
        torch.Tensor: 输出序列，形状为 (batch_size, num_nodes, out_channels, seq_len)
        """
        batch_size, num_nodes, in_channels, seq_len = x.size()

        # 重塑为 (batch_size * num_nodes, in_channels, seq_len)
        x = x.reshape(batch_size * num_nodes, in_channels, seq_len)

        # 应用卷积
        x = self.conv(x)  # (batch_size * num_nodes, out_channels, seq_len)

        # 重塑回原来的形状
        out_channels = x.size(1)
        x = x.reshape(batch_size, num_nodes, out_channels, seq_len)

        return x


class STGCNBlock(nn.Module):
    """时空图卷积块"""

    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size=3, dropout=0.2):
        """
        初始化时空图卷积块

        参数:
        in_channels (int): 输入通道数
        hidden_channels (int): 隐藏层通道数
        out_channels (int): 输出通道数
        kernel_size (int): 时间卷积核大小
        dropout (float): Dropout比率
        """
        super(STGCNBlock, self).__init__()

        # 第一个时间卷积层
        self.temp_conv1 = TemporalConvolution(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )

        # 图卷积层
        self.graph_conv = GraphConvolution(
            in_features=hidden_channels,
            out_features=hidden_channels
        )

        # 第二个时间卷积层
        self.temp_conv2 = TemporalConvolution(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2
        )

        # 残差连接
        self.residual = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1
        )

        # 批归一化
        self.bn = nn.BatchNorm2d(out_channels)

        # Dropout层
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        """
        前向传播

        参数:
        x (torch.Tensor): 输入特征，形状为 (batch_size, num_nodes, in_channels, seq_len)
        adj (torch.Tensor): 邻接矩阵，形状为 (batch_size, num_nodes, num_nodes)

        返回:
        torch.Tensor: 输出特征，形状为 (batch_size, num_nodes, out_channels, seq_len)
        """
        batch_size, num_nodes, in_channels, seq_len = x.size()

        # 残差连接 - 使用reshape而不是view避免内存不连续问题
        residual_x = x.permute(0, 1, 3, 2).contiguous()  # (batch_size, num_nodes, seq_len, in_channels)
        residual_x = residual_x.reshape(-1, seq_len, in_channels)  # (batch_size*num_nodes, seq_len, in_channels)
        residual_x = residual_x.permute(0, 2, 1).contiguous()  # (batch_size*num_nodes, in_channels, seq_len)
        residual_x = self.residual(residual_x)  # (batch_size*num_nodes, out_channels, seq_len)
        residual_x = residual_x.permute(0, 2, 1).contiguous()  # (batch_size*num_nodes, seq_len, out_channels)
        residual_x = residual_x.reshape(batch_size, num_nodes, seq_len, -1)  # (batch_size, num_nodes, seq_len, out_channels)
        residual = residual_x.permute(0, 1, 3, 2).contiguous()  # (batch_size, num_nodes, out_channels, seq_len)

        # 第一个时间卷积
        out = self.temp_conv1(x)  # (batch_size, num_nodes, hidden_channels, seq_len)

        # 将图卷积应用到每个时间步，但创建新的张量而不是就地修改
        out_t_list = []
        for t in range(seq_len):
            # 提取当前时间点特征
            out_t = out[:, :, :, t]  # (batch_size, num_nodes, hidden_channels)

            # 应用图卷积
            out_t = self.graph_conv(out_t, adj)  # (batch_size, num_nodes, hidden_channels)
            out_t_list.append(out_t.unsqueeze(-1))  # 添加时间维度

        # 沿时间维度连接所有结果
        out = torch.cat(out_t_list, dim=-1)  # (batch_size, num_nodes, hidden_channels, seq_len)

        # 第二个时间卷积
        out = self.temp_conv2(out)  # (batch_size, num_nodes, out_channels, seq_len)

        # 残差连接 - 注意不使用就地操作
        out = out + residual  # 不使用+=进行就地修改

        # 批归一化 (通道维度)
        out_bn = out.permute(0, 2, 1, 3).contiguous()  # (batch_size, out_channels, num_nodes, seq_len)
        out_bn = self.bn(out_bn)
        out = out_bn.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_nodes, out_channels, seq_len)

        # 激活函数
        out = F.relu(out)  # 不使用就地操作的F.relu_()
        out = self.dropout(out)

        return out


class STGCN(BaseTrajectoryModel):
    """时空图卷积网络"""

    def __init__(self, input_size, output_size=2, hidden_size=64, kernel_size=3,
                 num_nodes=None, dropout=0.2, num_layers=2, target_len=None,
                 adaptive_graph=True, device=DEVICE):
        """
        初始化时空图卷积网络

        参数:
        input_size (int): 输入特征维度
        output_size (int): 输出特征维度，默认为2 (lat, lon)
        hidden_size (int): 隐藏层大小
        kernel_size (int): 时间卷积核大小
        num_nodes (int): 图节点数（默认与输入序列长度相同）
        dropout (float): Dropout比率
        num_layers (int): STGCN块的数量
        target_len (int): 输出序列长度
        adaptive_graph (bool): 是否自适应构建图
        device (torch.device): 计算设备
        """
        super(STGCN, self).__init__(input_size, output_size, device)

        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.num_nodes = num_nodes
        self.dropout = dropout
        self.num_layers = num_layers
        self.target_len = target_len
        self.adaptive_graph = adaptive_graph
        self.input_size = input_size
        self.output_size = output_size

        # 输入投影层
        self.input_proj = nn.Linear(input_size, hidden_size)

        # 目标投影层 - 用于处理输出特征到隐藏维度的映射
        self.target_proj = nn.Linear(output_size, hidden_size)

        # 创建STGCN块
        self.st_blocks = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.st_blocks.append(
                    STGCNBlock(
                        in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=kernel_size,
                        dropout=dropout
                    )
                )
            else:
                self.st_blocks.append(
                    STGCNBlock(
                        in_channels=hidden_size,
                        hidden_channels=hidden_size,
                        out_channels=hidden_size,
                        kernel_size=kernel_size,
                        dropout=dropout
                    )
                )

        # 输出层
        self.out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

        # 如果使用自适应图，则需要学习图的构建
        if adaptive_graph:
            # 学习节点之间的相似度
            self.node_embeddings = nn.Parameter(torch.randn(1, hidden_size))
            self.edge_weights = nn.Linear(hidden_size * 2, 1)

        # 将模型移动到指定设备
        self.to(device)

    def build_graph(self, x):
        """
        构建图的邻接矩阵

        参数:
        x (torch.Tensor): 节点特征，形状为 (batch_size, num_nodes, feature_dim)

        返回:
        torch.Tensor: 归一化的邻接矩阵，形状为 (batch_size, num_nodes, num_nodes)
        """
        batch_size = x.size(0)
        num_nodes = x.size(1)

        if self.adaptive_graph:
            # 更高效的图构建方法，避免就地操作和循环

            # 准备节点特征 - 确保创建新张量而非修改原始张量
            node_feat = x  # (batch_size, num_nodes, hidden_size)

            # 创建所有节点对的特征表示
            # 将每个节点扩展为与所有其他节点的配对
            node_i = node_feat.unsqueeze(2)  # (batch_size, num_nodes, 1, hidden_size)
            node_j = node_feat.unsqueeze(1)  # (batch_size, 1, num_nodes, hidden_size)

            # 连接每对节点的特征
            node_i_expanded = node_i.expand(-1, -1, num_nodes, -1)  # (batch_size, num_nodes, num_nodes, hidden_size)
            node_j_expanded = node_j.expand(-1, num_nodes, -1, -1)  # (batch_size, num_nodes, num_nodes, hidden_size)

            # 合并特征
            paired_features = torch.cat([node_i_expanded, node_j_expanded], dim=3)  # (batch_size, num_nodes, num_nodes, 2*hidden_size)

            # 重塑以便处理
            paired_features_flat = paired_features.reshape(batch_size * num_nodes * num_nodes, -1)

            # 计算相似度/边权重
            edge_weights_flat = self.edge_weights(paired_features_flat)  # (batch_size*num_nodes*num_nodes, 1)
            edge_weights = edge_weights_flat.reshape(batch_size, num_nodes, num_nodes)  # (batch_size, num_nodes, num_nodes)

            # 使用softmax归一化邻接矩阵
            adj = F.softmax(edge_weights, dim=2)

        else:
            # 使用简单的全连接图
            adj = torch.ones(batch_size, num_nodes, num_nodes, device=self.device) / num_nodes

        return adj

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
        batch_size, seq_len, _ = x.size()

        # 确定目标序列长度
        target_len = self.target_len
        if target_len is None:
            target_len = target.size(1) if target is not None else 1

        # 确定节点数
        num_nodes = self.num_nodes if self.num_nodes is not None else seq_len

        # 将输入移动到设备
        x = x.to(self.device)
        if target is not None:
            target = target.to(self.device)

        # 输入投影
        x_proj = self.input_proj(x)  # (batch_size, seq_len, hidden_size)

        # 构建图
        adj = self.build_graph(x_proj)  # (batch_size, num_nodes, num_nodes)

        # 将序列视为时间步，重塑为 (batch_size, num_nodes, hidden_size, time_steps=1)
        x_reshaped = x_proj.unsqueeze(-1)  # (batch_size, seq_len, hidden_size, 1)

        # 应用时空图卷积块
        features = x_reshaped
        for st_block in self.st_blocks:
            features = st_block(features, adj)  # (batch_size, num_nodes, hidden_size, time_steps)

        # 提取最后一个时间步的特征用于预测
        last_features = features[:, :, :, -1]  # (batch_size, num_nodes, hidden_size)

        # 对于每个目标时间步，基于图特征进行预测
        outputs = []
        current_features = last_features  # 使用副本避免修改原始张量

        for t in range(target_len):
            # 预测下一个位置
            next_pos = self.out(current_features[:, -1, :])  # 使用最后一个节点的特征，形状为 (batch_size, output_size)
            outputs.append(next_pos.unsqueeze(1))  # (batch_size, 1, output_size)

            # 如果需要预测更多时间步，更新特征
            if t < target_len - 1:
                # 根据教师强制选择下一步输入
                if target is not None and torch.rand(1).item() < teacher_forcing_ratio:
                    # 使用真实目标作为下一个输入，投影到hidden_size
                    new_feat = self.target_proj(target[:, t, :])  # (batch_size, hidden_size)
                else:
                    # 使用预测作为下一个输入，投影到hidden_size
                    new_feat = self.target_proj(next_pos)  # (batch_size, hidden_size)

                # 准备新特征
                new_feat = new_feat.unsqueeze(1)  # (batch_size, 1, hidden_size)

                # 创建更新后的特征矩阵
                current_features = torch.cat([current_features[:, 1:, :], new_feat], dim=1)

                # 更新图结构 - 使用新的特征计算
                adj = self.build_graph(current_features)

        # 连接所有预测输出
        outputs = torch.cat(outputs, dim=1)  # (batch_size, target_len, output_size)

        return outputs

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
            outputs = self.forward(x, teacher_forcing_ratio=0.0)
        return outputs

    def get_config(self):
        """获取模型配置"""
        return {
            'hidden_size': self.hidden_size,
            'kernel_size': self.kernel_size,
            'num_nodes': self.num_nodes,
            'dropout': self.dropout,
            'num_layers': self.num_layers,
            'target_len': self.target_len,
            'adaptive_graph': self.adaptive_graph
        }