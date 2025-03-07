"""
轨迹预测模型基类
"""
import os
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod

from ..config import DEVICE, MODELS_DIR


class BaseTrajectoryModel(nn.Module, ABC):
    """轨迹预测模型基类"""

    def __init__(self, input_size, output_size=2, device=DEVICE):
        """
        初始化轨迹预测模型基类

        参数:
        input_size (int): 输入特征维度
        output_size (int): 输出特征维度，默认为2 (lat, lon)
        device (torch.device): 计算设备
        """
        super(BaseTrajectoryModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.device = device
        self.model_name = self.__class__.__name__

    @abstractmethod
    def forward(self, x, target=None, teacher_forcing_ratio=0.0):
        """
        前向传播

        参数:
        x (torch.Tensor): 输入张量 (batch_size, seq_len, input_size)
        target (torch.Tensor, optional): 目标张量，用于教师强制 (batch_size, seq_len, output_size)
        teacher_forcing_ratio (float): 教师强制比率

        返回:
        torch.Tensor: 预测结果 (batch_size, target_len, output_size)
        """
        pass

    def predict(self, x):
        """
        预测函数

        参数:
        x (torch.Tensor): 输入张量 (batch_size, seq_len, input_size)

        返回:
        torch.Tensor: 预测结果 (batch_size, target_len, output_size)
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            outputs = self.forward(x)
        return outputs

    def save(self, path=None, filename=None):
        """
        保存模型

        参数:
        path (str): 保存路径
        filename (str): 文件名

        返回:
        str: 保存的文件路径
        """
        if path is None:
            path = os.path.join(MODELS_DIR, self.model_name.lower())

        os.makedirs(path, exist_ok=True)

        if filename is None:
            filename = f"{self.model_name}_{self.get_parameter_count()}.pt"

        filepath = os.path.join(path, filename)

        # 保存模型状态和配置
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'model_config': self.get_config(),
            'model_name': self.model_name
        }, filepath)

        print(f"模型已保存到: {filepath}")
        return filepath

    @classmethod
    def load(cls, filepath, device=DEVICE):
        """
        加载模型

        参数:
        filepath (str): 模型文件路径
        device (torch.device): 计算设备

        返回:
        BaseTrajectoryModel: 加载的模型实例
        """
        checkpoint = torch.load(filepath, map_location=device)

        # 检查模型类名是否匹配
        if checkpoint['model_name'] != cls.__name__:
            print(f"警告: 加载的模型类型({checkpoint['model_name']})与当前类型({cls.__name__})不匹配")

        # 创建模型实例
        model = cls(
            input_size=checkpoint['input_size'],
            output_size=checkpoint['output_size'],
            **checkpoint.get('model_config', {}),
            device=device
        )

        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        print(f"模型已从 {filepath} 加载")
        return model

    def get_parameter_count(self):
        """获取模型参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_config(self):
        """获取模型配置，用于保存和加载"""
        return {}  # 基类返回空配置，子类应覆盖此方法

    def __str__(self):
        """返回模型描述"""
        return f"{self.model_name}(input_size={self.input_size}, output_size={self.output_size}, params={self.get_parameter_count():,})"