"""
轨迹预测模块配置
"""
import os
import json
import torch

# 数据路径
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"data", "processed"))
PREDICTION_SAMPLES_DIR = DATA_DIR
FEATURES_DIR = DATA_DIR

# 模型输出路径
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"))
MODELS_DIR = os.path.join(OUTPUT_DIR, 'models')
PREDICTIONS_DIR = os.path.join(OUTPUT_DIR, 'predictions')

# 报告路径
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "reports"))
FIGURES_DIR = os.path.join(REPORTS_DIR, 'figures/prediction_results')

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 训练配置
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 10
GRADIENT_CLIP = 5.0

# 数据集划分
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 输入输出窗口配置
DEFAULT_INPUT_LENGTH = 20
DEFAULT_OUTPUT_LENGTH = 5

# 模型配置
MODEL_CONFIGS = {
    'lstm': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': True
    },
    'seq2seq': {
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'bidirectional': True,
        'attention_type': 'bahdanau'  # 'bahdanau' or 'luong'
    },
    'transformer': {
        'hidden_size': 128,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 512,
        'dropout': 0.1
    },
    'stgcn': {
        'hidden_size': 64,
        'kernel_size': 3,
        'num_nodes': 20,  # 默认使用输入序列长度作为节点数
        'dropout': 0.2,
        'num_layers': 2,
        'adaptive_graph': True
    }
}

# 损失函数配置
LOSS_CONFIG = {
    'mse_weight': 1.0,
    'haversine_weight': 1.0
}


def load_experiment_config(config_file):
    """加载实验配置文件"""
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    return config


def save_experiment_config(config, output_file):
    """保存实验配置"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4)