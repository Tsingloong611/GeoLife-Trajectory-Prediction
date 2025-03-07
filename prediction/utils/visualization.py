"""
轨迹预测的可视化工具
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MarkerCluster
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix
import torch


def plot_trajectory_comparison(pred_traj, true_traj, input_traj=None,
                               title=None, zoom_start=13, save_path=None):
    """
    在地图上绘制预测轨迹和真实轨迹的对比

    参数:
    pred_traj (numpy.ndarray): 预测的轨迹坐标，形状为 (seq_len, 2)
    true_traj (numpy.ndarray): 真实的轨迹坐标，形状为 (seq_len, 2)
    input_traj (numpy.ndarray, optional): 输入的历史轨迹，形状为 (seq_len, 2)
    title (str, optional): 图表标题
    zoom_start (int): 初始缩放等级
    save_path (str, optional): 保存路径

    返回:
    folium.Map: 地图对象
    """
    # 检查输入
    if isinstance(pred_traj, torch.Tensor):
        pred_traj = pred_traj.cpu().numpy()
    if isinstance(true_traj, torch.Tensor):
        true_traj = true_traj.cpu().numpy()
    if isinstance(input_traj, torch.Tensor) and input_traj is not None:
        input_traj = input_traj.cpu().numpy()

    # 确保是numpy数组
    pred_traj = np.asarray(pred_traj)
    true_traj = np.asarray(true_traj)

    # 计算地图中心点
    if input_traj is not None:
        center_lat = np.mean(np.concatenate([input_traj[:, 0], true_traj[:, 0], pred_traj[:, 0]]))
        center_lon = np.mean(np.concatenate([input_traj[:, 1], true_traj[:, 1], pred_traj[:, 1]]))
    else:
        center_lat = np.mean(np.concatenate([true_traj[:, 0], pred_traj[:, 0]]))
        center_lon = np.mean(np.concatenate([true_traj[:, 1], pred_traj[:, 1]]))

    # 创建地图
    mymap = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start)

    # 添加真实轨迹
    true_points = [[lat, lon] for lat, lon in zip(true_traj[:, 0], true_traj[:, 1])]
    folium.PolyLine(
        true_points,
        color='blue',
        weight=3,
        opacity=0.8,
        popup='真实轨迹'
    ).add_to(mymap)

    # 添加预测轨迹
    pred_points = [[lat, lon] for lat, lon in zip(pred_traj[:, 0], pred_traj[:, 1])]
    folium.PolyLine(
        pred_points,
        color='red',
        weight=3,
        opacity=0.8,
        popup='预测轨迹'
    ).add_to(mymap)

    # 如果有输入轨迹，也添加到地图
    if input_traj is not None:
        input_points = [[lat, lon] for lat, lon in zip(input_traj[:, 0], input_traj[:, 1])]
        folium.PolyLine(
            input_points,
            color='green',
            weight=3,
            opacity=0.8,
            popup='历史轨迹'
        ).add_to(mymap)

    # 添加起点和终点标记
    # 真实轨迹
    folium.Marker(
        true_points[0],
        popup='真实起点',
        icon=folium.Icon(color='blue', icon='play')
    ).add_to(mymap)

    folium.Marker(
        true_points[-1],
        popup='真实终点',
        icon=folium.Icon(color='blue', icon='stop')
    ).add_to(mymap)

    # 预测轨迹
    folium.Marker(
        pred_points[0],
        popup='预测起点',
        icon=folium.Icon(color='red', icon='play')
    ).add_to(mymap)

    folium.Marker(
        pred_points[-1],
        popup='预测终点',
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(mymap)

    # 添加图例
    legend_html = '''
    <div style="position: fixed; 
        bottom: 50px; left: 50px; width: 150px; height: 100px; 
        border:2px solid grey; z-index:9999; font-size:14px;
        background-color:white;
        padding: 10px;
        border-radius: 5px;
        ">
    <p><span style="color:green;font-weight:bold;">&#x2501;</span> 历史轨迹</p>
    <p><span style="color:blue;font-weight:bold;">&#x2501;</span> 真实轨迹</p>
    <p><span style="color:red;font-weight:bold;">&#x2501;</span> 预测轨迹</p>
    </div>
    '''
    mymap.get_root().html.add_child(folium.Element(legend_html))

    # 如果提供了标题，添加到地图
    if title is not None:
        title_html = f'''
        <div style="position: fixed; 
            top: 10px; left: 50%; transform: translateX(-50%); 
            padding: 5px; border-radius: 5px; 
            background-color: white; 
            z-index:9999; font-size:16px; font-weight:bold;">
            {title}
        </div>
        '''
        mymap.get_root().html.add_child(folium.Element(title_html))

    # 保存地图
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mymap.save(save_path)

    return mymap


def plot_model_performance(metrics_dict, model_names=None,
                           title="模型性能对比", save_path=None):
    """
    绘制不同模型的性能指标对比图

    参数:
    metrics_dict (dict): 包含不同模型指标的字典，格式为 {model_name: {metric_name: value}}
    model_names (list, optional): 要包含的模型名称列表
    title (str): 图表标题
    save_path (str, optional): 保存路径

    返回:
    matplotlib.figure.Figure: 图表对象
    """
    if model_names is None:
        model_names = list(metrics_dict.keys())

    # 提取指标名称
    metrics = ['ade', 'fde', 'haversine', 'direction_accuracy']
    metric_labels = {
        'ade': 'ADE (米)',
        'fde': 'FDE (米)',
        'haversine': 'Haversine (米)',
        'direction_accuracy': '方向准确率'
    }

    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    # 设置颜色
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    # 绘制每个指标的条形图
    for i, metric in enumerate(metrics):
        ax = axes[i]

        # 收集数据
        values = []
        error_bars = []

        for model in model_names:
            if model in metrics_dict and metric in metrics_dict[model]:
                values.append(metrics_dict[model][metric])

                # 添加误差条（如果有置信区间）
                if f"{metric}_ci_lower" in metrics_dict[model] and f"{metric}_ci_upper" in metrics_dict[model]:
                    lower = metrics_dict[model][f"{metric}_ci_lower"]
                    upper = metrics_dict[model][f"{metric}_ci_upper"]
                    error_bars.append((values[-1] - lower, upper - values[-1]))
                else:
                    error_bars.append((0, 0))
            else:
                values.append(0)
                error_bars.append((0, 0))

        # 转换为numpy数组
        values = np.array(values)
        error_bars = np.array(error_bars).T

        # 绘制条形图
        bars = ax.bar(model_names, values, color=colors, alpha=0.7)

        # 添加误差条
        ax.errorbar(model_names, values, yerr=error_bars, fmt='none', color='black', capsize=5)

        # 设置标题和标签
        ax.set_title(metric_labels[metric])
        ax.set_ylabel(metric_labels[metric])

        # 为每个条形添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                    f'{value:.2f}', ha='center', va='bottom', rotation=0)

        # 设置刻度标签
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')

        # 添加网格线
        ax.grid(True, linestyle='--', alpha=0.7)

        # 方向准确率特殊处理（0-1范围）
        if metric == 'direction_accuracy':
            ax.set_ylim(0, 1.1)

    # 添加总标题
    fig.suptitle(title, fontsize=16)

    # 调整布局
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # 保存图表
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_attention_weights(attention_weights, input_seq=None, output_seq=None,
                           title="注意力权重可视化", save_path=None):
    """
    可视化注意力权重

    参数:
    attention_weights (torch.Tensor): 注意力权重，形状为 (batch_size, target_len, source_len)
    input_seq (list, optional): 输入序列标签
    output_seq (list, optional): 输出序列标签
    title (str): 图表标题
    save_path (str, optional): 保存路径

    返回:
    matplotlib.figure.Figure: 图表对象
    """
    # 检查输入
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.cpu().numpy()

    # 确保是numpy数组
    attention_weights = np.asarray(attention_weights)

    # 如果是批量数据，取第一个样本
    if len(attention_weights.shape) == 3:
        attention_weights = attention_weights[0]

    # 获取形状
    target_len, source_len = attention_weights.shape

    # 准备标签
    if input_seq is None:
        input_seq = [f'I{i}' for i in range(source_len)]
    if output_seq is None:
        output_seq = [f'O{i}' for i in range(target_len)]

    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制热图
    im = ax.imshow(attention_weights, cmap='viridis')

    # 添加颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("注意力权重", rotation=-90, va="bottom")

    # 设置刻度和标签
    ax.set_xticks(np.arange(source_len))
    ax.set_yticks(np.arange(target_len))
    ax.set_xticklabels(input_seq)
    ax.set_yticklabels(output_seq)

    # 旋转横轴标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个单元格上添加文本标注
    for i in range(target_len):
        for j in range(source_len):
            ax.text(j, i, f"{attention_weights[i, j]:.2f}",
                    ha="center", va="center", color="w" if attention_weights[i, j] > 0.5 else "black")

    # 设置标题
    ax.set_title(title)

    # 设置轴标签
    ax.set_xlabel("输入序列")
    ax.set_ylabel("输出序列")

    # 调整布局
    fig.tight_layout()

    # 保存图表
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_feature_importance(feature_importance, feature_names=None,
                            title="特征重要性", top_n=20, save_path=None):
    """
    绘制特征重要性条形图

    参数:
    feature_importance (dict or list): 特征重要性值
    feature_names (list, optional): 特征名称
    title (str): 图表标题
    top_n (int): 显示前N个重要特征
    save_path (str, optional): 保存路径

    返回:
    matplotlib.figure.Figure: 图表对象
    """
    # 格式化特征重要性数据
    if isinstance(feature_importance, dict):
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
    else:
        importances = feature_importance
        if feature_names is None:
            features = [f"Feature {i}" for i in range(len(importances))]
        else:
            features = feature_names

    # 创建数据框
    data = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })

    # 按重要性排序
    data = data.sort_values('Importance', ascending=False)

    # 限制特征数量
    if top_n is not None and len(data) > top_n:
        data = data.head(top_n)

    # 创建图表
    plt.figure(figsize=(12, 8))

    # 绘制条形图
    ax = sns.barplot(x='Importance', y='Feature', data=data, palette='viridis')

    # 添加数值标签
    for i, v in enumerate(data['Importance']):
        ax.text(v + 0.01, i, f"{v:.4f}", va='center')

    # 设置标题和标签
    plt.title(title, fontsize=14)
    plt.xlabel('重要性')
    plt.ylabel('特征')

    # 添加网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return plt.gcf()


def plot_ablation_study(ablation_results, metric_name='haversine',
                        title="消融实验结果", save_path=None):
    """
    绘制消融实验结果

    参数:
    ablation_results (dict): 消融实验结果，格式为 {config_name: {metric_name: value}}
    metric_name (str): 指标名称
    title (str): 图表标题
    save_path (str, optional): 保存路径

    返回:
    matplotlib.figure.Figure: 图表对象
    """
    # 提取配置名称和指标值
    configs = list(ablation_results.keys())
    values = [ablation_results[config][metric_name] for config in configs]

    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制条形图
    bars = ax.bar(configs, values, color=plt.cm.viridis(np.linspace(0, 1, len(configs))), alpha=0.7)

    # 为每个条形添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{value:.2f}', ha='center', va='bottom', rotation=0)

    # 设置标题和标签
    metric_labels = {
        'ade': 'ADE (米)',
        'fde': 'FDE (米)',
        'haversine': 'Haversine距离 (米)',
        'direction_accuracy': '方向准确率'
    }

    metric_label = metric_labels.get(metric_name, metric_name)
    ax.set_title(title)
    ax.set_ylabel(metric_label)
    ax.set_xlabel('配置')

    # 旋转横轴标签
    plt.xticks(rotation=45, ha='right')

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 如果是准确率，则设置y轴范围为0-1
    if 'accuracy' in metric_name.lower():
        ax.set_ylim(0, 1.1)

    # 调整布局
    plt.tight_layout()

    # 保存图表
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig