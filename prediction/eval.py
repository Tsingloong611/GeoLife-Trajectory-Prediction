"""
轨迹预测模型评估脚本
"""
import os
import argparse
import torch
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from .models import get_model
from .data import load_prediction_samples, load_selected_features, create_dataloaders
from .utils import (
    compute_metrics, plot_trajectory_comparison, plot_model_performance,
    plot_attention_weights, plot_ablation_study
)
from .config import (
    DEVICE, DEFAULT_INPUT_LENGTH, DEFAULT_OUTPUT_LENGTH,
    MODELS_DIR, PREDICTIONS_DIR, REPORTS_DIR, FIGURES_DIR
)


def load_model(model_path, model_name=None):
    """
    加载训练好的模型

    参数:
    model_path (str): 模型文件路径
    model_name (str, optional): 模型名称，如果为None则从模型文件中提取

    返回:
    模型实例
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 加载模型信息以获取模型类型
    checkpoint = torch.load(model_path, map_location=DEVICE)

    if model_name is None:
        model_name = checkpoint['model_name'].lower()

        # 处理模型名称
        if 'lstm' in model_name.lower():
            model_name = 'lstm'
        elif 'seq2seq' in model_name.lower():
            model_name = 'seq2seq'
        elif 'transformer' in model_name.lower():
            model_name = 'transformer'
        elif 'stgcn' in model_name.lower():
            model_name = 'stgcn'
        else:
            raise ValueError(f"无法识别的模型类型: {model_name}")

    # 获取模型类
    model_class = get_model(model_name)

    # 创建模型实例
    model = model_class.load(model_path, device=DEVICE)

    return model


def evaluate_model(model, test_loader, device=DEVICE, n_samples=5, save_dir=None):
    """
    评估模型性能

    参数:
    model: 模型实例
    test_loader: 测试数据加载器
    device: 计算设备
    n_samples: 可视化的样本数量
    save_dir: 保存结果的目录

    返回:
    tuple: (评估指标, 可视化结果)
    """
    model.eval()

    # 存储所有预测和真实值
    all_preds = []
    all_targets = []
    all_inputs = []
    all_user_ids = []
    all_traj_ids = []

    with torch.no_grad():
        for batch in test_loader:
            # 获取数据和目标
            inputs = batch['input'].to(device)
            targets = batch['target'].to(device)
            user_ids = batch['user_id']
            traj_ids = batch['traj_id']

            # 前向传播
            outputs = model(inputs, teacher_forcing_ratio=0.0)

            # 收集预测和真实值
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())
            all_inputs.append(inputs.cpu())
            all_user_ids.extend(user_ids)
            all_traj_ids.extend(traj_ids)

    # 连接所有批次的预测和真实值
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_inputs = torch.cat(all_inputs, dim=0)

    # 计算指标
    metrics = compute_metrics(all_preds, all_targets)

    # 保存结果
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

        # 保存指标
        metrics_path = os.path.join(save_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            # 转换numpy类型
            serializable_metrics = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                                   for k, v in metrics.items()}
            json.dump(serializable_metrics, f, indent=4)

        # 保存预测结果
        predictions_df = []
        for i in range(len(all_preds)):
            for j in range(all_preds.size(1)):
                predictions_df.append({
                    'user_id': all_user_ids[i],
                    'trajectory_id': all_traj_ids[i],
                    'time_step': j,
                    'pred_lat': all_preds[i, j, 0].item(),
                    'pred_lon': all_preds[i, j, 1].item(),
                    'true_lat': all_targets[i, j, 0].item(),
                    'true_lon': all_targets[i, j, 1].item()
                })

        predictions_df = pd.DataFrame(predictions_df)
        predictions_path = os.path.join(save_dir, "predictions.csv")
        predictions_df.to_csv(predictions_path, index=False)

        # 可视化一些样本
        figures_dir = os.path.join(save_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        visualizations = []

        # 随机选择样本进行可视化
        indices = np.random.choice(len(all_preds), min(n_samples, len(all_preds)), replace=False)

        for i, idx in enumerate(indices):
            pred = all_preds[idx].numpy()
            target = all_targets[idx].numpy()
            input_data = all_inputs[idx, :, :2].numpy()  # 只取经纬度坐标

            user_id = all_user_ids[idx]
            traj_id = all_traj_ids[idx]

            # 绘制轨迹对比图
            fig_title = f"用户 {user_id} 轨迹 {traj_id} 预测对比"
            html_path = os.path.join(figures_dir, f"trajectory_{i}.html")

            fig = plot_trajectory_comparison(
                pred_traj=pred,
                true_traj=target,
                input_traj=input_data,
                title=fig_title,
                save_path=html_path
            )

            visualizations.append({
                'index': i,
                'user_id': str(user_id),  # 确保是字符串
                'trajectory_id': str(traj_id),  # 确保是字符串
                'html_path': html_path
            })

        # 保存可视化信息
        vis_path = os.path.join(save_dir, "visualizations.json")
        with open(vis_path, 'w') as f:
            json.dump(visualizations, f, indent=4)

    return metrics, (all_preds, all_targets, all_inputs, all_user_ids, all_traj_ids)


def eval_single_model(
    model_path,
    model_name=None,
    input_length=DEFAULT_INPUT_LENGTH,
    output_length=DEFAULT_OUTPUT_LENGTH,
    batch_size=64,
    n_vis_samples=5,
    save_results=True
):
    """
    评估单个模型

    参数:
    model_path (str): 模型文件路径
    model_name (str, optional): 模型名称
    input_length (int): 输入序列长度
    output_length (int): 输出序列长度
    batch_size (int): 批大小
    n_vis_samples (int): 可视化的样本数量
    save_results (bool): 是否保存结果

    返回:
    dict: 评估指标
    """
    # 加载模型
    print(f"加载模型 {model_path}...")
    model = load_model(model_path, model_name)

    # 获取模型名称
    if model_name is None:
        model_name = model.__class__.__name__.lower()
        if 'lstm' in model_name:
            model_name = 'lstm'
        elif 'seq2seq' in model_name:
            model_name = 'seq2seq'
        elif 'transformer' in model_name:
            model_name = 'transformer'
        elif 'stgcn' in model_name:
            model_name = 'stgcn'

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

    # 保存目录
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(PREDICTIONS_DIR, f"{model_name}_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
    else:
        save_dir = None

    # 评估模型
    print("评估模型...")
    metrics, _ = evaluate_model(
        model, test_loader, DEVICE, n_samples=n_vis_samples, save_dir=save_dir
    )

    # 打印指标
    print("\n评估指标:")
    for metric_name, value in metrics.items():
        if not metric_name.endswith('_ci_lower') and not metric_name.endswith('_ci_upper'):
            print(f"{metric_name}: {value:.4f}")

    return metrics


def compare_models(
    model_paths,
    model_names=None,
    input_length=DEFAULT_INPUT_LENGTH,
    output_length=DEFAULT_OUTPUT_LENGTH,
    batch_size=64,
    save_results=True
):
    """
    比较多个模型的性能

    参数:
    model_paths (list): 模型文件路径列表
    model_names (list, optional): 模型名称列表
    input_length (int): 输入序列长度
    output_length (int): 输出序列长度
    batch_size (int): 批大小
    save_results (bool): 是否保存结果

    返回:
    dict: 包含所有模型评估指标的字典
    """
    if model_names is None:
        model_names = [None] * len(model_paths)

    # 保存目录
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(REPORTS_DIR, "model_comparison")
        figures_dir = os.path.join(FIGURES_DIR, "model_comparison")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

    # 评估结果
    all_metrics = {}

    # 逐个评估模型
    for i, (model_path, model_name) in enumerate(zip(model_paths, model_names)):
        print(f"\n评估模型 {i+1}/{len(model_paths)}: {model_path}")
        metrics = eval_single_model(
            model_path, model_name, input_length, output_length, batch_size, save_results=False
        )

        # 如果model_name为None，从模型中提取
        if model_name is None:
            model = load_model(model_path)
            model_name = model.__class__.__name__.lower()
            if 'lstm' in model_name:
                model_name = 'lstm'
            elif 'seq2seq' in model_name:
                model_name = 'seq2seq'
            elif 'transformer' in model_name:
                model_name = 'transformer'
            elif 'stgcn' in model_name:
                model_name = 'stgcn'

        all_metrics[model_name] = metrics

    # 保存比较结果
    if save_results:
        # 保存指标
        metrics_path = os.path.join(save_dir, f"model_comparison_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            # 转换不可序列化的值
            serializable_metrics = {}
            for model, metrics in all_metrics.items():
                serializable_metrics[model] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in metrics.items()
                }
            json.dump(serializable_metrics, f, indent=4)

        # 绘制性能比较图
        fig_path = os.path.join(figures_dir, f"model_comparison_{timestamp}.png")
        plot_model_performance(all_metrics, model_names=list(all_metrics.keys()),
                              title="轨迹预测模型性能比较", save_path=fig_path)

    return all_metrics


def run_ablation_study(
    base_model_path,
    ablation_configs,
    input_length=DEFAULT_INPUT_LENGTH,
    output_length=DEFAULT_OUTPUT_LENGTH,
    batch_size=64,
    save_results=True
):
    """
    运行消融实验

    参数:
    base_model_path (str): 基准模型文件路径
    ablation_configs (dict): 消融实验配置，格式为 {name: config}
    input_length (int): 输入序列长度
    output_length (int): 输出序列长度
    batch_size (int): 批大小
    save_results (bool): 是否保存结果

    返回:
    dict: 消融实验结果
    """
    # 加载基准模型
    print(f"加载基准模型 {base_model_path}...")
    base_model = load_model(base_model_path)
    model_name = base_model.__class__.__name__.lower()

    # 提取模型类型
    if 'lstm' in model_name:
        model_type = 'lstm'
    elif 'seq2seq' in model_name:
        model_type = 'seq2seq'
    elif 'transformer' in model_name:
        model_type = 'transformer'
    elif 'stgcn' in model_name:
        model_type = 'stgcn'
    else:
        raise ValueError(f"无法识别的模型类型: {model_name}")

    # 保存目录
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(REPORTS_DIR, f"ablation_{model_type}")
        figures_dir = os.path.join(FIGURES_DIR, f"ablation_{model_type}")
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)

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

    # 评估基准模型
    print("\n评估基准模型...")
    base_metrics, _ = evaluate_model(base_model, test_loader, DEVICE, n_samples=0)

    # 保存消融实验结果
    ablation_results = {'base_model': base_metrics}

    # 运行消融实验
    for config_name, config in ablation_configs.items():
        print(f"\n运行消融实验: {config_name}")

        # 获取模型类
        model_class = get_model(model_type)

        # 创建模型实例
        ablation_model = model_class(
            input_size=base_model.input_size,
            output_size=base_model.output_size,
            target_len=output_length,
            device=DEVICE,
            **config
        )

        # 评估模型
        metrics, _ = evaluate_model(ablation_model, test_loader, DEVICE, n_samples=0)

        # 保存结果
        ablation_results[config_name] = metrics

    # 保存结果
    if save_results:
        # 保存指标
        metrics_path = os.path.join(save_dir, f"ablation_results_{timestamp}.json")
        with open(metrics_path, 'w') as f:
            # 转换不可序列化的值
            serializable_results = {}
            for config, metrics in ablation_results.items():
                serializable_results[config] = {
                    k: float(v) if isinstance(v, (np.float32, np.float64)) else v
                    for k, v in metrics.items()
                }
            json.dump(serializable_results, f, indent=4)

        # 绘制消融实验结果
        fig_path = os.path.join(figures_dir, f"ablation_results_{timestamp}.png")
        plot_ablation_study(ablation_results, metric_name='haversine',
                           title=f"{model_type.upper()} 模型消融实验结果", save_path=fig_path)

    return ablation_results


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="轨迹预测模型评估")

    parser.add_argument("--model", type=str, required=True,
                        help="模型文件路径或目录")

    parser.add_argument("--mode", type=str, default="single",
                        choices=["single", "compare", "ablation"],
                        help="评估模式: single (单个模型), compare (比较多个模型), ablation (消融实验)")

    parser.add_argument("--input_length", type=int, default=DEFAULT_INPUT_LENGTH,
                        help=f"输入序列长度 (默认: {DEFAULT_INPUT_LENGTH})")

    parser.add_argument("--output_length", type=int, default=DEFAULT_OUTPUT_LENGTH,
                        help=f"输出序列长度 (默认: {DEFAULT_OUTPUT_LENGTH})")

    parser.add_argument("--batch_size", type=int, default=64,
                        help="批大小 (默认: 64)")

    parser.add_argument("--n_samples", type=int, default=5,
                        help="可视化的样本数量 (默认: 5)")

    parser.add_argument("--ablation_config", type=str, default=None,
                        help="消融实验配置文件路径")

    parser.add_argument("--no_save", action="store_true",
                        help="不保存评估结果")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # 检查设备
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("使用CPU进行评估")

    # 执行评估任务
    if args.mode == "single":
        # 评估单个模型
        eval_single_model(
            model_path=args.model,
            input_length=args.input_length,
            output_length=args.output_length,
            batch_size=args.batch_size,
            n_vis_samples=args.n_samples,
            save_results=not args.no_save
        )

    elif args.mode == "compare":
        # 比较多个模型
        if os.path.isdir(args.model):
            # 如果提供的是目录，查找所有模型文件
            model_paths = []
            for root, dirs, files in os.walk(args.model):
                for file in files:
                    if file.endswith(".pt"):
                        model_paths.append(os.path.join(root, file))

            if not model_paths:
                raise FileNotFoundError(f"在目录 {args.model} 中未找到模型文件")
        else:
            # 如果提供的是单个文件，查看是否包含多个模型路径
            try:
                with open(args.model, 'r') as f:
                    model_paths = json.load(f)

                if not isinstance(model_paths, list):
                    model_paths = [args.model]
            except:
                model_paths = [args.model]

        # 比较模型
        compare_models(
            model_paths=model_paths,
            input_length=args.input_length,
            output_length=args.output_length,
            batch_size=args.batch_size,
            save_results=not args.no_save
        )

    elif args.mode == "ablation":
        # 运行消融实验
        if args.ablation_config is None:
            raise ValueError("消融实验模式需要提供配置文件路径 (--ablation_config)")

        # 加载消融实验配置
        with open(args.ablation_config, 'r') as f:
            ablation_configs = json.load(f)

        # 运行消融实验
        run_ablation_study(
            base_model_path=args.model,
            ablation_configs=ablation_configs,
            input_length=args.input_length,
            output_length=args.output_length,
            batch_size=args.batch_size,
            save_results=not args.no_save
        )