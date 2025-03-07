"""
轨迹预测模型训练脚本
"""
import os
import time
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datetime import datetime

from .models import get_model
from .data import load_prediction_samples, load_selected_features, create_dataloaders
from .utils import combined_loss, compute_metrics
from .config import (
    DEVICE, BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    EARLY_STOPPING_PATIENCE, GRADIENT_CLIP, MODEL_CONFIGS,
    MODELS_DIR, DEFAULT_INPUT_LENGTH, DEFAULT_OUTPUT_LENGTH,
    load_experiment_config, save_experiment_config,
)


def train_epoch(model, dataloader, optimizer, device, clip_grad=None):
    """
    训练一个epoch

    参数:
    model: 模型
    dataloader: 数据加载器
    optimizer: 优化器
    device: 计算设备
    clip_grad: 梯度裁剪值

    返回:
    float: 平均训练损失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(dataloader)

    for batch in dataloader:
        # 获取数据和目标
        inputs = batch['input'].to(device)
        targets = batch['target'].to(device)

        # 前向传播
        outputs = model(inputs, targets, teacher_forcing_ratio=0.5)

        # 计算损失
        loss, _, _ = combined_loss(outputs, targets)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        # 更新参数
        optimizer.step()

        # 累计损失
        total_loss += loss.item()

    # 计算平均损失
    avg_loss = total_loss / num_batches

    return avg_loss


def validate(model, dataloader, device):
    """
    验证模型

    参数:
    model: 模型
    dataloader: 数据加载器
    device: 计算设备

    返回:
    tuple: (平均验证损失, 指标字典)
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(dataloader)

    # 存储所有预测和真实值
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            # 获取数据和目标
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)

            # 前向传播
            outputs = model(inputs, targets, teacher_forcing_ratio=0.0)

            # 计算损失
            loss, _, _ = combined_loss(outputs, targets)

            # 累计损失
            total_loss += loss.item()

            # 收集预测和真实值
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # 计算平均损失
    avg_loss = total_loss / num_batches

    # 连接所有批次的预测和真实值
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # 计算指标
    metrics = compute_metrics(all_preds, all_targets)

    return avg_loss, metrics


def train(
        model_name,
        input_length=DEFAULT_INPUT_LENGTH,
        output_length=DEFAULT_OUTPUT_LENGTH,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        gradient_clip=GRADIENT_CLIP,
        model_config=None,
        experiment_name=None,
        save_best=True,
        verbose=True
):
    """
    训练轨迹预测模型

    参数:
    model_name (str): 模型名称
    input_length (int): 输入序列长度
    output_length (int): 输出序列长度
    batch_size (int): 批大小
    learning_rate (float): 学习率
    num_epochs (int): 训练轮数
    early_stopping_patience (int): 早停耐心值
    gradient_clip (float): 梯度裁剪值
    model_config (dict): 模型配置
    experiment_name (str): 实验名称
    save_best (bool): 是否保存最佳模型
    verbose (bool): 是否打印详细信息

    返回:
    tuple: (训练历史, 最佳模型路径)
    """
    # 设置实验名称
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{model_name}_{input_length}_{output_length}_{timestamp}"

    # 创建保存目录
    model_save_dir = os.path.join(MODELS_DIR, model_name)
    os.makedirs(model_save_dir, exist_ok=True)

    # 加载数据
    print(f"加载预测样本 (输入长度: {input_length}, 输出长度: {output_length})...")
    samples_df = load_prediction_samples(input_length, output_length)

    # 加载特征列表
    print("加载特征列表...")
    selected_features = load_selected_features()

    # 创建数据加载器
    print("创建数据加载器...")
    train_loader, val_loader, test_loader, dataset = create_dataloaders(
        samples_df, selected_features, batch_size=batch_size
    )

    # 获取特征信息
    feature_info = dataset.get_feature_info()
    input_size = len(feature_info['base_features'])
    print(f"输入特征数量: {input_size}")

    # 使用默认模型配置
    if model_config is None:
        if model_name in MODEL_CONFIGS:
            model_config = MODEL_CONFIGS[model_name]
        else:
            model_config = {}

    # 实例化模型
    print(f"创建{model_name}模型...")
    model_class = get_model(model_name)
    model = model_class(
        input_size=input_size,
        output_size=2,  # 输出维度为2 (lat, lon)
        target_len=output_length,
        device=DEVICE,
        **model_config
    )

    # 打印模型信息
    print(f"模型参数数量: {model.get_parameter_count():,}")

    # 定义优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'metrics': [],
        'best_epoch': 0,
        'best_val_loss': float('inf')
    }

    # 早停计数器
    early_stopping_counter = 0
    best_model_path = None

    # 训练循环
    print(f"开始训练 {num_epochs} 轮...")
    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, DEVICE, gradient_clip)

        # 验证
        val_loss, metrics = validate(model, val_loader, DEVICE)

        # 更新学习率
        scheduler.step(val_loss)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['metrics'].append(metrics)

        # 计算耗时
        epoch_time = time.time() - start_time

        # 打印进度
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Haversine: {metrics['haversine']:.2f}m, "
                  f"Time: {epoch_time:.2f}s")

        # 检查是否是最佳模型
        if val_loss < history['best_val_loss']:
            history['best_val_loss'] = val_loss
            history['best_epoch'] = epoch
            early_stopping_counter = 0

            # 保存最佳模型
            if save_best:
                best_model_path = os.path.join(model_save_dir, f"{experiment_name}_best.pt")
                model.save(best_model_path)
                if verbose:
                    print(f"保存最佳模型到 {best_model_path}")
        else:
            early_stopping_counter += 1
            if verbose and early_stopping_counter > 0:
                print(f"EarlyStopping counter: {early_stopping_counter}/{early_stopping_patience}")

        # 早停
        if early_stopping_counter >= early_stopping_patience:
            if verbose:
                print(f"Early stopping at epoch {epoch + 1}")
            break

    # 保存训练历史
    history_path = os.path.join(model_save_dir, f"{experiment_name}_history.json")
    with open(history_path, 'w') as f:
        # 转换不可序列化的值
        serializable_history = {
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss'],
            'metrics': [{k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                         for k, v in m.items()} for m in history['metrics']],
            'best_epoch': history['best_epoch'],
            'best_val_loss': float(history['best_val_loss'])
        }
        json.dump(serializable_history, f, indent=4)

    # 保存实验配置
    config = {
        'model_name': model_name,
        'input_length': input_length,
        'output_length': output_length,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'gradient_clip': gradient_clip,
        'model_config': model_config,
        'feature_info': feature_info,
        'experiment_name': experiment_name,
        'best_metrics': {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                         for k, v in history['metrics'][history['best_epoch']].items()},
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    config_path = os.path.join(model_save_dir, f"{experiment_name}_config.json")
    save_experiment_config(config, config_path)

    print(f"训练完成! 最佳模型在第 {history['best_epoch'] + 1} 轮，验证损失: {history['best_val_loss']:.4f}")

    # 返回训练历史和最佳模型路径
    return history, best_model_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="轨迹预测模型训练")

    parser.add_argument("--model", type=str, required=True,
                        choices=["lstm", "seq2seq", "transformer", "stgcn"],
                        help="要训练的模型类型")

    parser.add_argument("--config", type=str, default=None,
                        help="实验配置文件路径")

    parser.add_argument("--input_length", type=int, default=DEFAULT_INPUT_LENGTH,
                        help=f"输入序列长度 (默认: {DEFAULT_INPUT_LENGTH})")

    parser.add_argument("--output_length", type=int, default=DEFAULT_OUTPUT_LENGTH,
                        help=f"输出序列长度 (默认: {DEFAULT_OUTPUT_LENGTH})")

    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help=f"批大小 (默认: {BATCH_SIZE})")

    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help=f"学习率 (默认: {LEARNING_RATE})")

    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help=f"训练轮数 (默认: {NUM_EPOCHS})")

    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE,
                        help=f"早停耐心值 (默认: {EARLY_STOPPING_PATIENCE})")

    parser.add_argument("--clip", type=float, default=GRADIENT_CLIP,
                        help=f"梯度裁剪值 (默认: {GRADIENT_CLIP})")

    parser.add_argument("--name", type=str, default=None,
                        help="实验名称，默认为'模型名_输入长度_输出长度_时间戳'")

    parser.add_argument("--no_save", action="store_true",
                        help="不保存最佳模型")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 打印GPU信息
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU进行训练")

    # 加载实验配置
    if args.config is not None:
        config = load_experiment_config(args.config)
        print(f"从 {args.config} 加载实验配置")

        # 使用配置文件中的参数
        model_name = config.get('model_name', args.model)
        input_length = config.get('input_length', args.input_length)
        output_length = config.get('output_length', args.output_length)
        batch_size = config.get('batch_size', args.batch_size)
        learning_rate = config.get('learning_rate', args.lr)
        num_epochs = config.get('num_epochs', args.epochs)
        early_stopping_patience = config.get('early_stopping_patience', args.patience)
        gradient_clip = config.get('gradient_clip', args.clip)
        model_config = config.get('model_config', None)
        experiment_name = config.get('experiment_name', args.name)
    else:
        # 使用命令行参数
        model_name = args.model
        input_length = args.input_length
        output_length = args.output_length
        batch_size = args.batch_size
        learning_rate = args.lr
        num_epochs = args.epochs
        early_stopping_patience = args.patience
        gradient_clip = args.clip
        model_config = MODEL_CONFIGS.get(model_name, {})
        experiment_name = args.name

    # 训练模型
    train(
        model_name=model_name,
        input_length=input_length,
        output_length=output_length,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        early_stopping_patience=early_stopping_patience,
        gradient_clip=gradient_clip,
        model_config=model_config,
        experiment_name=experiment_name,
        save_best=not args.no_save,
        verbose=True
    )