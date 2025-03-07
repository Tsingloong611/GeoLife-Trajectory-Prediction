"""
轨迹预测模型模块
"""

from .base_model import BaseTrajectoryModel
from .lstm import LSTMTrajectoryPredictor
from .seq2seq import Seq2SeqAttention
from .transformer import TransformerTrajectoryPredictor
from .stgcn import STGCN

__all__ = [
    'BaseTrajectoryModel',
    'LSTMTrajectoryPredictor',
    'Seq2SeqAttention',
    'TransformerTrajectoryPredictor',
    'STGCN'
]

# 注册可用模型
MODEL_REGISTRY = {
    'lstm': LSTMTrajectoryPredictor,
    'seq2seq': Seq2SeqAttention,
    'transformer': TransformerTrajectoryPredictor,
    'stgcn': STGCN
}


def get_model(model_name):
    """
    根据名称获取模型类

    参数:
    model_name (str): 模型名称

    返回:
    class: 模型类
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"未知的模型名称: {model_name}，可用模型: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[model_name]
