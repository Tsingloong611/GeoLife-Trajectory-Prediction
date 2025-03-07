import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap, MarkerCluster
import os
import random
from datetime import datetime, timedelta
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection

# 设置中文
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置黑体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题


def plot_trajectory_duration_distribution(df, output_file=None):
    """
    绘制轨迹持续时间分布图

    参数:
    df: 轨迹数据
    output_file: 输出图像文件名 (可选)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 确定使用哪个轨迹ID列
    traj_id_col = None
    for col in ['segment_trajectory_id', 'trajectory_id']:
        if col in df.columns:
            traj_id_col = col
            break

    if traj_id_col is None:
        print("错误: 数据中没有轨迹ID列 (segment_trajectory_id 或 trajectory_id)")
        return

    print("计算轨迹持续时间...")

    # 按轨迹ID分组，计算每条轨迹的持续时间（分钟）
    traj_durations = []
    traj_points = []

    for traj_id, group in df.groupby(traj_id_col):
        if len(group) < 2:
            continue

        # 按时间排序
        group = group.sort_values('datetime')

        # 计算持续时间（分钟）
        duration_minutes = (group['datetime'].max() - group['datetime'].min()).total_seconds() / 60

        # 记录持续时间和轨迹点数
        traj_durations.append(duration_minutes)
        traj_points.append(len(group))

    # 创建DataFrame以便绘图
    duration_df = pd.DataFrame({
        'duration_minutes': traj_durations,
        'point_count': traj_points
    })

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 1. 绘制持续时间分布图 - 使用对数刻度
    sns.histplot(duration_df['duration_minutes'].clip(upper=duration_df['duration_minutes'].quantile(0.99)),
                 bins=50, kde=True, ax=ax1, color='skyblue')
    ax1.set_title('轨迹持续时间分布')
    ax1.set_xlabel('持续时间（分钟）')
    ax1.set_ylabel('轨迹数量')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 添加垂直线标记典型的时间阈值
    time_thresholds = [15, 30, 60, 120]  # 15分钟，30分钟，1小时，2小时
    colors = ['green', 'blue', 'red', 'purple']

    for threshold, color in zip(time_thresholds, colors):
        ax1.axvline(threshold, color=color, linestyle='--',
                    label=f'{threshold}分钟')

    # 添加统计信息
    mean_duration = np.mean(traj_durations)
    median_duration = np.median(traj_durations)

    textstr = '\n'.join((
        f'平均持续时间: {mean_duration:.2f}分钟',
        f'中位持续时间: {median_duration:.2f}分钟',
        f'最短轨迹: {min(traj_durations):.2f}分钟',
        f'最长轨迹: {max(traj_durations):.2f}分钟',
        f'总轨迹数: {len(traj_durations)}'
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    ax1.legend()

    # 2. 绘制轨迹点数量分布图 - 使用对数刻度
    sns.histplot(duration_df['point_count'].clip(upper=duration_df['point_count'].quantile(0.99)),
                 bins=50, kde=True, ax=ax2, color='lightgreen')
    ax2.set_title('轨迹点数量分布')
    ax2.set_xlabel('轨迹点数量')
    ax2.set_ylabel('轨迹数量')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 添加统计信息
    mean_points = np.mean(traj_points)
    median_points = np.median(traj_points)

    textstr = '\n'.join((
        f'平均点数: {mean_points:.1f}',
        f'中位点数: {median_points:.1f}',
        f'最少点数: {min(traj_points)}',
        f'最多点数: {max(traj_points)}',
        f'总轨迹数: {len(traj_points)}'
    ))

    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # 保存图片
    if output_file is None:
        output_file = "trajectory_duration_distribution.png"

    plt.savefig(output_file, dpi=300)
    print(f"保存轨迹持续时间分布图到: {output_file}")

    plt.close()

    return fig


def plot_sampling_interval_distribution(df, output_file=None):
    """
    绘制采样间隔分布图

    参数:
    df: 轨迹数据
    output_file: 输出图像文件名 (可选)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 确定使用哪个轨迹ID列
    traj_id_col = None
    for col in ['segment_trajectory_id', 'trajectory_id']:
        if col in df.columns:
            traj_id_col = col
            break

    if traj_id_col is None:
        print("错误: 数据中没有轨迹ID列 (segment_trajectory_id 或 trajectory_id)")
        return

    print("计算轨迹采样间隔...")

    # 计算所有轨迹的采样间隔
    sampling_intervals = []

    for traj_id, group in df.groupby(traj_id_col):
        if len(group) < 2:
            continue

        # 按时间排序
        group = group.sort_values('datetime')

        # 计算采样间隔（秒）
        intervals = group['datetime'].diff().dt.total_seconds().dropna().tolist()
        sampling_intervals.extend(intervals)

    # 移除极端值（大于1小时的间隔）
    sampling_intervals = [i for i in sampling_intervals if i <= 3600]

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    # 1. 绘制采样间隔分布图 - 线性刻度，关注小间隔
    bins = 50
    max_interval_linear = 60  # 最大显示60秒
    filtered_intervals_linear = [i for i in sampling_intervals if i <= max_interval_linear]

    sns.histplot(filtered_intervals_linear, bins=bins, kde=True, ax=ax1, color='skyblue')
    ax1.set_title('采样间隔分布 (≤ 60秒)')
    ax1.set_xlabel('采样间隔（秒）')
    ax1.set_ylabel('频率')
    ax1.grid(True, linestyle='--', alpha=0.7)

    # 添加垂直线标记典型的采样间隔
    interval_thresholds = [1, 5, 10, 30]
    colors = ['green', 'blue', 'red', 'purple']

    for threshold, color in zip(interval_thresholds, colors):
        ax1.axvline(threshold, color=color, linestyle='--',
                    label=f'{threshold}秒')

    # 计算各阈值的百分比
    interval_percentages = {}
    for threshold in [1, 5, 10, 30, 60, 300]:
        percentage = 100 * len([i for i in sampling_intervals if i <= threshold]) / len(sampling_intervals)
        interval_percentages[threshold] = percentage

    textstr = '\n'.join((
        f'≤ 1秒: {interval_percentages[1]:.1f}%',
        f'≤ 5秒: {interval_percentages[5]:.1f}%',
        f'≤ 10秒: {interval_percentages[10]:.1f}%',
        f'≤ 30秒: {interval_percentages[30]:.1f}%',
        f'≤ 60秒: {interval_percentages[60]:.1f}%',
        f'≤ 5分钟: {interval_percentages[300]:.1f}%'
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.95, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    ax1.legend()

    # 2. 绘制采样间隔分布图 - 对数刻度，显示全范围
    sns.histplot(sampling_intervals, bins=bins, kde=True, ax=ax2, color='lightgreen')
    ax2.set_title('采样间隔分布（对数刻度）')
    ax2.set_xlabel('采样间隔（秒）')
    ax2.set_ylabel('频率')
    ax2.set_xscale('log')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 添加垂直线标记典型的采样间隔（对数刻度）
    log_thresholds = [1, 5, 10, 30, 60, 300, 600, 1800, 3600]  # 1秒到1小时
    log_colors = ['green', 'blue', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'black']

    for threshold, color in zip(log_thresholds, log_colors):
        ax2.axvline(threshold, color=color, linestyle='--',
                    label=f'{threshold}秒' if threshold < 60 else f'{threshold // 60}分钟')

    # 添加统计信息
    mean_interval = np.mean(sampling_intervals)
    median_interval = np.median(sampling_intervals)

    textstr = '\n'.join((
        f'平均间隔: {mean_interval:.2f}秒',
        f'中位间隔: {median_interval:.2f}秒',
        f'采样点总数: {len(sampling_intervals)}',
        f'> 5分钟: {100 - interval_percentages[300]:.1f}%'
    ))

    ax2.text(0.05, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='left', bbox=props)

    ax2.legend(loc='lower right')

    plt.tight_layout()

    # 保存图片
    if output_file is None:
        output_file = "sampling_interval_distribution.png"

    plt.savefig(output_file, dpi=300)
    print(f"保存采样间隔分布图到: {output_file}")

    plt.close()

    return fig


def plot_user_activity_distribution(df, output_file=None):
    """
    绘制用户活跃度分布图

    参数:
    df: 轨迹数据
    output_file: 输出图像文件名 (可选)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 确定使用哪个轨迹ID列
    traj_id_col = None
    for col in ['segment_trajectory_id', 'trajectory_id']:
        if col in df.columns:
            traj_id_col = col
            break

    if traj_id_col is None:
        print("错误: 数据中没有轨迹ID列 (segment_trajectory_id 或 trajectory_id)")
        return

    print("计算用户活跃度...")

    # 统计每个用户的轨迹数和轨迹点数
    user_stats = df.groupby('user_id').agg({
        traj_id_col: 'nunique',
        'datetime': 'count'
    }).reset_index()

    # 重命名列
    user_stats.columns = ['user_id', 'trajectory_count', 'point_count']

    # 按轨迹数排序
    user_stats = user_stats.sort_values('trajectory_count', ascending=False).reset_index(drop=True)

    # 添加累计百分比列
    user_stats['trajectory_percentage'] = user_stats['trajectory_count'].cumsum() / user_stats[
        'trajectory_count'].sum() * 100
    user_stats['point_percentage'] = user_stats['point_count'].cumsum() / user_stats['point_count'].sum() * 100

    # 创建图形
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    # 1. 用户轨迹数分布 - 前50名用户
    top_n = min(50, len(user_stats))

    ax1.bar(range(top_n), user_stats['trajectory_count'].head(top_n), color='skyblue')
    ax1.set_title(f'前{top_n}名用户的轨迹数量')
    ax1.set_xlabel('用户排名')
    ax1.set_ylabel('轨迹数量')
    ax1.grid(True, linestyle='--', alpha=0.7, axis='y')

    # 添加累计百分比线
    ax1_twin = ax1.twinx()
    ax1_twin.plot(range(top_n), user_stats['trajectory_percentage'].head(top_n), 'r-')
    ax1_twin.set_ylabel('累计百分比 (%)', color='r')
    ax1_twin.tick_params(axis='y', colors='r')

    # 2. 用户轨迹数的直方图
    sns.histplot(user_stats['trajectory_count'], bins=30, kde=True, ax=ax2, color='lightgreen')
    ax2.set_title('用户轨迹数量分布')
    ax2.set_xlabel('每用户轨迹数量')
    ax2.set_ylabel('用户数量')
    ax2.grid(True, linestyle='--', alpha=0.7)

    # 添加统计信息
    mean_traj = user_stats['trajectory_count'].mean()
    median_traj = user_stats['trajectory_count'].median()
    max_traj = user_stats['trajectory_count'].max()

    textstr = '\n'.join((
        f'用户总数: {len(user_stats)}',
        f'平均轨迹数: {mean_traj:.1f}',
        f'中位轨迹数: {median_traj:.1f}',
        f'最大轨迹数: {max_traj}'
    ))

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.95, 0.95, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    # 3. 帕累托图：用户贡献与用户百分比
    # 计算用户百分比
    user_percentages = np.arange(1, len(user_stats) + 1) / len(user_stats) * 100

    ax3.plot(user_percentages, user_stats['trajectory_percentage'], 'b-', label='轨迹数')
    ax3.plot(user_percentages, user_stats['point_percentage'], 'g-', label='轨迹点数')

    # 添加参考线
    ax3.axhline(y=80, color='r', linestyle='--', alpha=0.7, label='80%贡献')

    # 找到贡献了80%数据的用户百分比
    trajectory_80pct = user_percentages[user_stats['trajectory_percentage'] >= 80].min() if any(
        user_stats['trajectory_percentage'] >= 80) else 100
    point_80pct = user_percentages[user_stats['point_percentage'] >= 80].min() if any(
        user_stats['point_percentage'] >= 80) else 100

    ax3.axvline(x=trajectory_80pct, color='b', linestyle='--', alpha=0.7)
    ax3.axvline(x=point_80pct, color='g', linestyle='--', alpha=0.7)

    ax3.set_title('用户贡献的帕累托图')
    ax3.set_xlabel('用户百分比 (%)')
    ax3.set_ylabel('累计贡献 (%)')
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend()

    # 添加统计信息
    top10_traj_pct = user_stats['trajectory_percentage'].iloc[9] if len(user_stats) > 9 else 100
    top20_traj_pct = user_stats['trajectory_percentage'].iloc[19] if len(user_stats) > 19 else 100

    textstr = '\n'.join((
        f'前10%用户贡献: {user_stats["trajectory_percentage"].iloc[int(len(user_stats) * 0.1) - 1]:.1f}% 轨迹',
        f'前10名用户贡献: {top10_traj_pct:.1f}% 轨迹',
        f'前20名用户贡献: {top20_traj_pct:.1f}% 轨迹',
        f'贡献80%轨迹的用户比例: {trajectory_80pct:.1f}%',
        f'贡献80%轨迹点的用户比例: {point_80pct:.1f}%'
    ))

    ax3.text(0.95, 0.35, textstr, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=props)

    plt.tight_layout()

    # 保存图片
    if output_file is None:
        output_file = "user_activity_distribution.png"

    plt.savefig(output_file, dpi=300)
    print(f"保存用户活跃度分布图到: {output_file}")

    plt.close()

    return fig


def plot_trajectory_on_map(df, user_id=None, trajectory_id=None, output_file=None,
                           max_trajectories=5, random_selection=True):
    """
    在地图上绘制轨迹

    参数:
    df: 轨迹数据
    user_id: 指定用户ID (可选)
    trajectory_id: 指定轨迹ID (可选)
    output_file: 输出HTML文件名 (可选)
    max_trajectories: 最大轨迹数
    random_selection: 是否随机选择轨迹
    """
    if user_id is not None:
        df = df[df['user_id'] == user_id]

    # 确定使用哪个轨迹ID列
    traj_id_col = None
    for col in ['segment_trajectory_id', 'trajectory_id']:
        if col in df.columns:
            traj_id_col = col
            break

    if traj_id_col is None:
        print("错误: 数据中没有轨迹ID列 (segment_trajectory_id 或 trajectory_id)")
        return

    if trajectory_id is not None:
        df = df[df[traj_id_col] == trajectory_id]

    # 获取不同的轨迹
    trajectories = df[traj_id_col].unique()

    if len(trajectories) == 0:
        print("没有找到符合条件的轨迹")
        return

    # 限制轨迹数量
    if len(trajectories) > max_trajectories:
        if random_selection:
            trajectories = np.random.choice(trajectories, max_trajectories, replace=False)
        else:
            trajectories = trajectories[:max_trajectories]

    # 创建地图，以第一个轨迹的中心点为中心
    center_trajectory = df[df[traj_id_col] == trajectories[0]]
    center_lat = center_trajectory['latitude'].mean()
    center_lon = center_trajectory['longitude'].mean()

    mymap = folium.Map(location=[center_lat, center_lon], zoom_start=12)

    # 为每个轨迹生成随机颜色
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue',
              'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'lightblue', 'lightgreen']

    # 为每条轨迹绘制线条
    for i, traj_id in enumerate(trajectories):
        traj_df = df[df[traj_id_col] == traj_id].sort_values('datetime')

        # 获取用户ID和轨迹ID
        user = traj_df['user_id'].iloc[0]

        # 选择颜色
        color = colors[i % len(colors)]

        # 创建轨迹的点列表
        points = [[row['latitude'], row['longitude']] for _, row in traj_df.iterrows()]

        # 添加起点和终点标记
        start_point = points[0]
        end_point = points[-1]

        # 起点标记
        folium.Marker(
            start_point,
            popup=f'起点: 用户{user}, 轨迹{traj_id}',
            icon=folium.Icon(color='green', icon='play')
        ).add_to(mymap)

        # 终点标记
        folium.Marker(
            end_point,
            popup=f'终点: 用户{user}, 轨迹{traj_id}',
            icon=folium.Icon(color='red', icon='stop')
        ).add_to(mymap)

        # 添加轨迹线
        folium.PolyLine(
            points,
            popup=f'用户{user}, 轨迹{traj_id}',
            color=color,
            weight=3,
            opacity=0.8
        ).add_to(mymap)

    # 保存地图到HTML文件
    if output_file is None:
        output_file = "trajectory_map.html"

    print(f"保存地图到: {output_file}")
    mymap.save(output_file)

    return mymap


def plot_heatmap(df, output_file="heatmap.html"):
    """
    创建轨迹点热力图
    """
    # 创建地图，以数据中心为中心
    center_lat = df['latitude'].mean()
    center_lon = df['longitude'].mean()

    mymap = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    # 准备热力图数据
    heat_data = [[row['latitude'], row['longitude']] for _, row in df.iterrows()]

    # 添加热力图层
    HeatMap(heat_data, radius=10).add_to(mymap)

    # 保存到文件
    mymap.save(output_file)
    print(f"保存热力图到: {output_file}")

    return mymap


def plot_trajectory_with_speed(df, user_id=None, trajectory_id=None, output_file=None):
    """
    绘制带速度信息的轨迹图

    颜色表示速度: 蓝色-慢, 黄色-中, 红色-快
    """
    # 确定使用哪个轨迹ID列
    traj_id_col = None
    for col in ['segment_trajectory_id', 'trajectory_id']:
        if col in df.columns:
            traj_id_col = col
            break

    if traj_id_col is None:
        print("错误: 数据中没有轨迹ID列 (segment_trajectory_id 或 trajectory_id)")
        return

    if user_id is not None:
        df = df[df['user_id'] == user_id]

    if trajectory_id is not None:
        df = df[df[traj_id_col] == trajectory_id]
    else:
        # 如果未指定轨迹ID，随机选择一个轨迹
        trajectories = df[traj_id_col].unique()
        if len(trajectories) == 0:
            print("没有找到符合条件的轨迹")
            return
        trajectory_id = np.random.choice(trajectories)
        df = df[df[traj_id_col] == trajectory_id]

    # 确保按时间排序
    df = df.sort_values('datetime')

    # 提取用户ID和轨迹ID用于标题
    user = df['user_id'].iloc[0]
    traj = df[traj_id_col].iloc[0]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))

    # 提取经纬度和速度
    lats = df['latitude'].values
    lons = df['longitude'].values

    # 确保速度列存在，如果不存在就用0填充
    if 'speed_kmh' in df.columns:
        speeds = df['speed_kmh'].values
    else:
        print("警告: 未找到speed_kmh列，使用0代替")
        speeds = np.zeros(len(df))

    # 创建点对
    points = np.array([lons, lats]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # 创建颜色映射
    norm = plt.Normalize(0, min(max(speeds) if len(speeds) > 0 else 0, 100))  # 限制最大速度为100

    # 创建颜色映射表：蓝-绿-黄-红
    cmap = plt.cm.jet

    # 创建带颜色的线段集合
    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=2, alpha=0.7)
    lc.set_array(speeds[:-1])  # 设置每个线段的速度
    line = ax.add_collection(lc)

    # 添加颜色条
    cbar = plt.colorbar(line, ax=ax)
    cbar.set_label('速度 (km/h)')

    # 设置图形标题和轴标签
    plt.title(f'用户 {user} 的轨迹 {traj} (颜色表示速度)')
    plt.xlabel('经度')
    plt.ylabel('纬度')

    # 设置轴范围
    margin = 0.001  # 添加一点边距
    ax.set_xlim(min(lons) - margin, max(lons) + margin)
    ax.set_ylim(min(lats) - margin, max(lats) + margin)

    # 添加起点和终点标记
    ax.plot(lons[0], lats[0], 'go', markersize=10, label='起点')
    ax.plot(lons[-1], lats[-1], 'ro', markersize=10, label='终点')

    # 添加图例
    plt.legend()

    # 紧凑布局
    plt.tight_layout()

    # 保存图片
    if output_file is None:
        output_file = f"trajectory_speed_{user}_{traj}.png"

    plt.savefig(output_file, dpi=300)
    print(f"保存速度轨迹图到: {output_file}")

    plt.close()

    return fig


def plot_speed_time_series(df, user_id=None, trajectory_id=None, output_file=None):
    """
    绘制速度随时间变化的时间序列图
    """
    # 确定使用哪个轨迹ID列
    traj_id_col = None
    for col in ['segment_trajectory_id', 'trajectory_id']:
        if col in df.columns:
            traj_id_col = col
            break

    if traj_id_col is None:
        print("错误: 数据中没有轨迹ID列 (segment_trajectory_id 或 trajectory_id)")
        return

    if user_id is not None:
        df = df[df['user_id'] == user_id]

    if trajectory_id is not None:
        df = df[df[traj_id_col] == trajectory_id]
    else:
        # 如果未指定轨迹ID，随机选择一个轨迹
        trajectories = df[traj_id_col].unique()
        if len(trajectories) == 0:
            print("没有找到符合条件的轨迹")
            return
        trajectory_id = np.random.choice(trajectories)
        df = df[df[traj_id_col] == trajectory_id]

    # 确保按时间排序
    df = df.sort_values('datetime')

    # 提取用户ID和轨迹ID用于标题
    user = df['user_id'].iloc[0]
    traj = df[traj_id_col].iloc[0]

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 确保速度列存在
    if 'speed_kmh' in df.columns:
        # 绘制速度时间序列
        ax.plot(df['datetime'], df['speed_kmh'], '-o', alpha=0.7)

        # 添加平滑曲线（移动平均）
        window_size = min(15, len(df) // 5 + 1)  # 动态窗口大小
        if window_size > 1:
            df['smooth_speed'] = df['speed_kmh'].rolling(window=window_size, center=True).mean()
            ax.plot(df['datetime'], df['smooth_speed'], 'r-', linewidth=2, label=f'移动平均 (窗口={window_size})')

        # 设置Y轴范围，避免异常值影响图形比例
        upper_limit = min(df['speed_kmh'].quantile(0.95) * 1.5, df['speed_kmh'].max())
        ax.set_ylim(0, upper_limit)
    else:
        print("警告: 未找到speed_kmh列，无法绘制速度时间序列")
        ax.text(0.5, 0.5, '数据中没有速度信息', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=14)

    # 设置图形标题和轴标签
    plt.title(f'用户 {user} 的轨迹 {traj} 速度变化')
    plt.xlabel('时间')
    plt.ylabel('速度 (km/h)')

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 格式化x轴日期
    fig.autofmt_xdate()

    # 添加图例
    if 'speed_kmh' in df.columns and window_size > 1:
        plt.legend()

    # 紧凑布局
    plt.tight_layout()

    # 保存图片
    if output_file is None:
        output_file = f"speed_time_series_{user}_{traj}.png"

    plt.savefig(output_file, dpi=300)
    print(f"保存速度时间序列图到: {output_file}")

    plt.close()

    return fig


def plot_user_activity_patterns(df, user_id=None, output_file=None):
    """
    绘制用户活动模式热图（按小时和星期几）
    """
    if user_id is not None:
        df = df[df['user_id'] == user_id]
        title_prefix = f"用户 {user_id} 的"
    else:
        title_prefix = "所有用户的"

    # 确保时间特征存在
    if 'hour' not in df.columns or 'day_of_week' not in df.columns:
        print("计算小时和星期几...")
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek

    # 创建星期几-小时计数矩阵
    activity_matrix = np.zeros((7, 24))

    # 按星期几和小时计数
    for day in range(7):
        for hour in range(24):
            activity_matrix[day, hour] = len(df[(df['day_of_week'] == day) & (df['hour'] == hour)])

    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 8))

    # 绘制热图
    sns.heatmap(activity_matrix, cmap='YlGnBu', ax=ax,
                xticklabels=range(24),
                yticklabels=['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'],
                cbar_kws={'label': '轨迹点数量'})

    # 设置标题和轴标签
    plt.title(f"{title_prefix}活动模式热图")
    plt.xlabel('小时')
    plt.ylabel('星期')

    # 紧凑布局
    plt.tight_layout()

    # 保存图片
    if output_file is None:
        if user_id is not None:
            output_file = f"activity_patterns_user_{user_id}.png"
        else:
            output_file = "activity_patterns_all_users.png"

    plt.savefig(output_file, dpi=300)
    print(f"保存活动模式热图到: {output_file}")

    plt.close()

    return fig


def plot_speed_distribution(df, user_id=None, output_file=None):
    """
    绘制速度分布直方图和核密度估计
    """
    if user_id is not None:
        df = df[df['user_id'] == user_id]
        title_suffix = f" (用户 {user_id})"
    else:
        title_suffix = " (所有用户)"

    # 检查速度列是否存在
    if 'speed_kmh' not in df.columns:
        print("警告: 未找到speed_kmh列，无法绘制速度分布")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, '数据中没有速度信息', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes, fontsize=14)

        # 保存图片
        if output_file is None:
            if user_id is not None:
                output_file = f"speed_distribution_user_{user_id}.png"
            else:
                output_file = "speed_distribution_all_users.png"

        plt.savefig(output_file, dpi=300)
        print(f"保存速度分布图到: {output_file}")

        plt.close()
        return fig

    # 过滤掉异常值和前2%的数据（通常是静止不动的点）
    speed_threshold = df['speed_kmh'].quantile(0.02)
    speed_data = df[(df['speed_kmh'] > speed_threshold) & (df['speed_kmh'] < 150)]

    if len(speed_data) == 0:
        print("没有足够的速度数据进行绘图")
        return

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制速度分布直方图和核密度估计
    sns.histplot(speed_data['speed_kmh'], bins=50, kde=True, color='skyblue', ax=ax)

    # 设置标题和轴标签
    plt.title(f"速度分布{title_suffix}")
    plt.xlabel('速度 (km/h)')
    plt.ylabel('频率')

    # 设置X轴范围
    plt.xlim(0, min(150, speed_data['speed_kmh'].max() * 1.1))

    # 添加垂直线标记不同速度阈值
    percentiles = [25, 50, 75]
    colors = ['green', 'blue', 'red']

    for p, c in zip(percentiles, colors):
        percentile_value = speed_data['speed_kmh'].quantile(p / 100)
        plt.axvline(percentile_value, color=c, linestyle='--',
                    label=f'{p}% 分位数: {percentile_value:.2f} km/h')

    # 添加平均速度
    mean_speed = speed_data['speed_kmh'].mean()
    plt.axvline(mean_speed, color='purple', linestyle='-',
                label=f'平均速度: {mean_speed:.2f} km/h')

    # 添加图例
    plt.legend()

    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)

    # 紧凑布局
    plt.tight_layout()

    # 保存图片
    if output_file is None:
        if user_id is not None:
            output_file = f"speed_distribution_user_{user_id}.png"
        else:
            output_file = "speed_distribution_all_users.png"

    plt.savefig(output_file, dpi=300)
    print(f"保存速度分布图到: {output_file}")

    plt.close()

    return fig


def add_missing_columns(df):
    """
    添加可能缺失的必要列，以便可视化脚本可以正常运行
    """
    if 'segment_trajectory_id' not in df.columns and 'trajectory_id' in df.columns:
        print("注意: 将trajectory_id列用作segment_trajectory_id")
        df['segment_trajectory_id'] = df['trajectory_id']

    if 'speed_kmh' not in df.columns and 'speed' in df.columns:
        print("注意: 根据speed列计算speed_kmh")
        df['speed_kmh'] = df['speed'] * 3.6  # 将m/s转换为km/h

    return df


def main():
    # 文件路径
    input_dir = "../data/processed"
    output_dir = "../reports/figures"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 读取清理后的数据
    print("读取清理后的数据...")

    try:
        df = pd.read_csv(os.path.join(input_dir, "geolife_beijing_cleaned_data.csv"))
        # 确保datetime列为datetime类型
        df['datetime'] = pd.to_datetime(df['datetime'])
        output_dir = os.path.join(output_dir, "geolife_cleaned_data")
        os.makedirs(output_dir, exist_ok=True)
    except FileNotFoundError:
        print("清理后的数据文件不存在，尝试读取插值后的数据...")
        try:
            df = pd.read_csv(os.path.join(input_dir, "geolife_beijing_interpolated_data.csv"))
            # 确保datetime列为datetime类型
            df['datetime'] = pd.to_datetime(df['datetime'])
            output_dir = os.path.join(output_dir, "geolife_interpolated_data")
            os.makedirs(output_dir, exist_ok=True)
        except FileNotFoundError:
            print("插值后的数据文件不存在，尝试读取原始特征数据...")
            try:
                df = pd.read_csv(os.path.join(input_dir, "geolife_feature_data.csv"))
                # 确保datetime列为datetime类型
                df['datetime'] = pd.to_datetime(df['datetime'])
                output_dir = os.path.join(output_dir, "geolife_feature_data")
                os.makedirs(output_dir, exist_ok=True)
            except FileNotFoundError:
                print("找不到任何数据文件，请先运行数据处理脚本")
                return

    print(f"成功读取数据，共 {len(df)} 条记录")

    # 添加可能缺失的列
    df = add_missing_columns(df)

    # 1. 绘制随机轨迹地图
    print("绘制随机轨迹地图...")
    map_file = os.path.join(output_dir, "random_trajectories_map.html")
    plot_trajectory_on_map(df, output_file=map_file, max_trajectories=5, random_selection=True)

    # 2. 绘制热力图
    print("绘制轨迹点热力图...")
    heatmap_file = os.path.join(output_dir, "trajectory_heatmap.html")
    plot_heatmap(df, output_file=heatmap_file)

    # 3. 轨迹持续时间和长度分布分析
    print("分析轨迹持续时间和长度分布...")
    duration_file = os.path.join(output_dir, "trajectory_duration_distribution.png")
    plot_trajectory_duration_distribution(df, output_file=duration_file)

    # 4. 采样间隔分布分析
    print("分析采样间隔分布...")
    interval_file = os.path.join(output_dir, "sampling_interval_distribution.png")
    plot_sampling_interval_distribution(df, output_file=interval_file)

    # 5. 用户活跃度分布分析
    print("分析用户活跃度分布...")
    activity_dist_file = os.path.join(output_dir, "user_activity_distribution.png")
    plot_user_activity_distribution(df, output_file=activity_dist_file)

    # 6. 随机选择一个用户，绘制其轨迹速度图
    user_ids = df['user_id'].unique()
    random_user = np.random.choice(user_ids)

    print(f"为用户 {random_user} 绘制轨迹速度图...")
    user_df = df[df['user_id'] == random_user]

    # 确定使用哪个轨迹ID列
    traj_id_col = None
    for col in ['segment_trajectory_id', 'trajectory_id']:
        if col in df.columns:
            traj_id_col = col
            break

    if traj_id_col is not None:
        segment_ids = user_df[traj_id_col].unique()

        if len(segment_ids) > 0:
            random_segment = np.random.choice(segment_ids)

            # 绘制带速度的轨迹图
            speed_trajectory_file = os.path.join(output_dir, f"speed_trajectory_user_{random_user}.png")
            plot_trajectory_with_speed(df, user_id=random_user, trajectory_id=random_segment,
                                       output_file=speed_trajectory_file)

            # 绘制速度时间序列图
            speed_time_file = os.path.join(output_dir, f"speed_time_user_{random_user}.png")
            plot_speed_time_series(df, user_id=random_user, trajectory_id=random_segment,
                                   output_file=speed_time_file)

    # 7. 绘制用户活动模式热图
    print("绘制用户活动模式热图...")
    activity_file = os.path.join(output_dir, f"activity_patterns_user_{random_user}.png")
    plot_user_activity_patterns(df, user_id=random_user, output_file=activity_file)

    # 8. 绘制所有用户的活动模式热图
    print("绘制所有用户的活动模式热图...")
    all_activity_file = os.path.join(output_dir, "activity_patterns_all_users.png")
    plot_user_activity_patterns(df, output_file=all_activity_file)

    # 9. 绘制速度分布图
    print("绘制速度分布图...")
    speed_dist_file = os.path.join(output_dir, "speed_distribution_all_users.png")
    plot_speed_distribution(df, output_file=speed_dist_file)

    print("可视化完成！所有图表已保存到:", output_dir)


if __name__ == "__main__":
    main()
