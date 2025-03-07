import pandas as pd
import numpy as np
import os
from datetime import timedelta

from numpy import dtype
from tqdm import tqdm
import time
import sys

# Check for proper file paths
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = "../data/processed/geolife_feature_data.csv"

# Check if the file exists
if not os.path.exists(input_file):
    print(f"Error: Input file not found: {input_file}")
    print("Please specify the correct file path as an argument.")
    sys.exit(1)

# Try to use GPU acceleration
print("Checking for GPU acceleration...")
try:
    import torch

    if torch.cuda.is_available():
        USE_GPU = True
        device = torch.device("cuda:0")
        print(f"GPU available: {torch.cuda.get_device_name(0)}")

        try:
            import cupy as cp

            print("CuPy is available for additional GPU acceleration")
        except ImportError:
            print("CuPy not found, some operations may be slower")
    else:
        USE_GPU = False
        print("PyTorch found but CUDA not available. Using CPU mode.")
except ImportError:
    USE_GPU = False
    print("PyTorch not found. Using CPU mode.")

# 定义北京市的地理边界（经纬度范围）
BEIJING_BOUNDS = {
    'lat_min': 39.4,  # 北京市纬度范围下限
    'lat_max': 41.1,  # 北京市纬度范围上限
    'lon_min': 115.7,  # 北京市经度范围下限
    'lon_max': 117.4  # 北京市经度范围上限
}


def haversine_distance_cpu(lat1, lon1, lat2, lon2):
    """
    计算两点之间的Haversine距离(米) - CPU版本
    """
    R = 6371000  # 地球半径(米)

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c


def haversine_distance_gpu(lat1, lon1, lat2, lon2):
    """
    计算两点之间的Haversine距离(米) - GPU版本
    """
    # 确保输入是PyTorch张量并且在GPU上
    if not isinstance(lat1, torch.Tensor):
        lat1 = torch.tensor(lat1, device=device, dtype=torch.float32)
        lon1 = torch.tensor(lon1, device=device, dtype=torch.float32)
        lat2 = torch.tensor(lat2, device=device, dtype=torch.float32)
        lon2 = torch.tensor(lon2, device=device, dtype=torch.float32)

    R = 6371000.0  # 地球半径(米)

    # 将度转换为弧度
    lat1_rad = torch.deg2rad(lat1)
    lon1_rad = torch.deg2rad(lon1)
    lat2_rad = torch.deg2rad(lat2)
    lon2_rad = torch.deg2rad(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = torch.sin(dlat / 2.0) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlon / 2.0) ** 2
    c = 2.0 * torch.atan2(torch.sqrt(a), torch.sqrt(1.0 - a))

    return R * c


def batch_calculate_distance(df, batch_size=10000):
    """
    批量计算Haversine距离，根据可用性使用GPU或CPU
    """
    n = len(df) - 1
    distances = np.zeros(len(df))

    # 第一个点的距离设为0
    distances[0] = 0

    if USE_GPU:
        # GPU版本
        for i in range(0, n, batch_size):
            end_idx = min(i + batch_size, n)
            # 获取当前批次的坐标
            lat1 = df['latitude'].iloc[i:end_idx].values
            lon1 = df['longitude'].iloc[i:end_idx].values
            lat2 = df['latitude'].iloc[i + 1:end_idx + 1].values
            lon2 = df['longitude'].iloc[i + 1:end_idx + 1].values

            # 转换为GPU张量
            lat1_tensor = torch.tensor(lat1, device=device, dtype=torch.float32)
            lon1_tensor = torch.tensor(lon1, device=device, dtype=torch.float32)
            lat2_tensor = torch.tensor(lat2, device=device, dtype=torch.float32)
            lon2_tensor = torch.tensor(lon2, device=device, dtype=torch.float32)

            # 计算距离
            batch_distances = haversine_distance_gpu(lat1_tensor, lon1_tensor, lat2_tensor, lon2_tensor)

            # 将结果移回CPU并存储
            distances[i + 1:end_idx + 1] = batch_distances.cpu().numpy()
    else:
        # CPU版本 - 使用向量化操作提高效率
        for i in range(1, len(df)):
            distances[i] = haversine_distance_cpu(
                df['latitude'].iloc[i - 1], df['longitude'].iloc[i - 1],
                df['latitude'].iloc[i], df['longitude'].iloc[i]
            )

    return distances


def clean_geolife_data(input_file, output_file, max_speed_kmh=150,
                       time_gap_minutes=30, min_trajectory_points=10):
    """
    清理GeoLife轨迹数据，只保留北京区域的轨迹
    """
    print(f"读取数据: {input_file}")
    try:
        start_time = time.time()
        df = pd.read_csv(input_file)
        print(f"数据读取完成，耗时: {time.time() - start_time:.2f}秒")
    except Exception as e:
        print(f"读取数据出错: {e}")
        print(f"请确保文件路径正确: {input_file}")
        return None

    # 确保datetime列为datetime类型
    df['datetime'] = pd.to_datetime(df['datetime'])

    original_count = len(df)
    print(f"原始数据点数: {original_count}")

    # 1. 只保留北京区域内的轨迹点
    print("过滤非北京区域的轨迹点...")
    beijing_mask = ((df['latitude'] >= BEIJING_BOUNDS['lat_min']) &
                    (df['latitude'] <= BEIJING_BOUNDS['lat_max']) &
                    (df['longitude'] >= BEIJING_BOUNDS['lon_min']) &
                    (df['longitude'] <= BEIJING_BOUNDS['lon_max']))

    df_beijing = df[beijing_mask].copy()
    print(f"北京区域内的数据点: {len(df_beijing)} (占原始数据的 {len(df_beijing) / original_count * 100:.2f}%)")

    # 2. 处理异常速度值
    # 将速度转换为km/h进行过滤
    df_beijing['speed_kmh'] = df_beijing['speed'] * 3.6  # m/s转换为km/h
    speed_mask = (df_beijing['speed_kmh'] <= max_speed_kmh) | df_beijing['speed_kmh'].isna()
    df_cleaned = df_beijing[speed_mask].copy()
    print(f"移除超速点后剩余: {len(df_cleaned)} (移除了 {len(df_beijing) - len(df_cleaned)} 点)")

    # 3. 处理缺失值
    # 确保轨迹按用户、轨迹ID和时间排序
    df_cleaned = df_cleaned.sort_values(['user_id', 'trajectory_id', 'datetime'])

    # 处理缺失的距离和速度
    # 对于轨迹的第一个点，距离和速度通常是NaN，设置为0
    first_points = df_cleaned.groupby(['user_id', 'trajectory_id']).head(1).index
    df_cleaned.loc[first_points, 'distance'] = 0
    df_cleaned.loc[first_points, 'speed'] = 0
    df_cleaned.loc[first_points, 'speed_kmh'] = 0

    # 4. 轨迹分段: 如果时间间隔超过阈值，创建新的轨迹段
    print("执行轨迹分段...")
    df_cleaned['time_gap'] = df_cleaned.groupby(['user_id', 'trajectory_id'])['datetime'].diff()
    df_cleaned['new_segment'] = (df_cleaned['time_gap'] > timedelta(minutes=time_gap_minutes)) | (
        df_cleaned['time_gap'].isna())

    # 为每个轨迹创建新的segment_id
    df_cleaned['segment_id'] = df_cleaned.groupby(['user_id', 'trajectory_id'])['new_segment'].cumsum()
    df_cleaned['segment_trajectory_id'] = df_cleaned['trajectory_id'].astype(str) + '_' + df_cleaned[
        'segment_id'].astype(str)

    # 5. 识别并处理停留点
    if USE_GPU:
        df_cleaned = process_stay_points_gpu(df_cleaned)
    else:
        df_cleaned = process_stay_points_cpu(df_cleaned)

    # 6. 移除点数过少的轨迹段
    segment_counts = df_cleaned.groupby(['user_id', 'segment_trajectory_id']).size()
    valid_segments = segment_counts[segment_counts >= min_trajectory_points]

    # 创建多级索引
    df_cleaned_indexed = df_cleaned.set_index(['user_id', 'segment_trajectory_id'])

    # 过滤保留有效的轨迹段
    df_cleaned_filtered = df_cleaned_indexed.loc[valid_segments.index].reset_index()

    # 删除辅助列
    df_cleaned_filtered = df_cleaned_filtered.drop(
        ['new_segment', 'segment_id', 'time_gap', 'is_stay_point', 'stay_point_id'], axis=1,
        errors='ignore')

    print(f"清理后的数据点数: {len(df_cleaned_filtered)}")
    print(f"共有 {len(segment_counts)} 个轨迹段, 其中 {len(valid_segments)} 个有效")

    # 7. 保存清理后的数据
    try:
        print(f"保存清理后的数据: {output_file}")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df_cleaned_filtered.to_csv(output_file, index=False)
        print(f"数据已成功保存")
    except Exception as e:
        print(f"保存数据出错: {e}")
        print(f"请确保输出目录存在并有写入权限: {os.path.dirname(output_file)}")

    return df_cleaned_filtered

def process_stay_points_cpu(df, distance_threshold=50, time_threshold=300, merge_stay_points=True):
    """
    识别停留点并进行处理 - CPU版本
    """
    print("识别并处理停留点（CPU模式）...")

    # 初始化停留点标记
    df['is_stay_point'] = False
    df['stay_point_id'] = -1

    # 按用户和轨迹段分组处理
    stay_point_count = 0
    result_dfs = []

    for (user, traj), group in tqdm(df.groupby(['user_id', 'segment_trajectory_id'])):
        group = group.reset_index(drop=True).copy()

        # 初始化变量
        i = 0
        stay_points = []
        current_stay_start = None

        while i < len(group) - 1:
            # 查找可能的停留点起始位置
            current_stay_start = i

            # 计算后续点与当前点的距离
            ref_lat = group.iloc[current_stay_start]['latitude']
            ref_lon = group.iloc[current_stay_start]['longitude']
            ref_time = group.iloc[current_stay_start]['datetime']

            j = i + 1
            while j < len(group):
                # 提取当前点
                curr_lat = group.iloc[j]['latitude']
                curr_lon = group.iloc[j]['longitude']
                curr_time = group.iloc[j]['datetime']

                # 计算距离
                dist = haversine_distance_cpu(ref_lat, ref_lon, curr_lat, curr_lon)

                if dist > distance_threshold:
                    # 超出距离阈值，不是停留点的一部分
                    break

                j += 1

            # 判断是否形成停留点（至少两个点且时间超过阈值）
            if j > i + 1:
                time_diff = (group.iloc[j - 1]['datetime'] - ref_time).total_seconds()

                if time_diff >= time_threshold:
                    # 识别为停留点
                    stay_points.append((current_stay_start, j - 1, time_diff))

                    # 标记停留点
                    group.loc[current_stay_start:j - 1, 'is_stay_point'] = True
                    group.loc[current_stay_start:j - 1, 'stay_point_id'] = stay_point_count
                    stay_point_count += 1

                    i = j  # 跳过已处理的停留点
                    continue

            # 不是停留点，继续检查下一个点
            i += 1

        # 合并停留点（如果需要）
        if merge_stay_points and stay_points:
            # 保留非停留点
            non_stay_mask = ~group['is_stay_point']
            filtered_group = group[non_stay_mask].copy()

            # 对每个停留点，只保留开始和结束点
            for start_idx, end_idx, duration in stay_points:
                # 提取开始和结束点
                start_point = group.iloc[start_idx].copy()
                end_point = group.iloc[end_idx].copy()

                # 添加额外的停留信息
                start_point['stay_duration'] = duration
                end_point['stay_duration'] = duration

                # 将开始和结束点添加到结果中
                filtered_group = pd.concat([filtered_group, pd.DataFrame([start_point, end_point])])

            # 重新按时间排序
            filtered_group = filtered_group.sort_values('datetime').reset_index(drop=True)
            result_dfs.append(filtered_group)
        else:
            # 不合并停留点，保留所有点
            result_dfs.append(group)

    # 合并所有处理后的轨迹
    result_df = pd.concat(result_dfs, ignore_index=True)

    print(f"共识别 {stay_point_count} 个停留点")
    print(f"处理后的数据点数: {len(result_df)}")

    return result_df


def process_stay_points_gpu(df, distance_threshold=50, time_threshold=300, merge_stay_points=True):
    print("识别并处理停留点（GPU加速）...")

    df['is_stay_point'] = False
    df['stay_point_id'] = -1

    stay_point_count = 0
    result_dfs = []

    for (user, traj), group in tqdm(df.groupby(['user_id', 'segment_trajectory_id'])):
        group = group.reset_index(drop=True).copy()

        if len(group) <= 1:
            result_dfs.append(group)
            continue

        # 采用分块计算距离，避免创建过大的矩阵
        lats = torch.tensor(group['latitude'].values, device=device, dtype=torch.float32)
        lons = torch.tensor(group['longitude'].values, device=device, dtype=torch.float32)

        batch_size = 256  # 适当减少 batch_size，避免显存溢出

        distances_np = np.zeros((len(group), len(group)), dtype=np.float32)  # 这里仍然占用较大内存
        for i in range(0, len(group), batch_size):
            end_i = min(i + batch_size, len(group))
            for j in range(i, len(group), batch_size):
                end_j = min(j + batch_size, len(group))

                # 计算小批次的距离矩阵
                batch_distances = haversine_distance_gpu(
                    lats[i:end_i].unsqueeze(1), lons[i:end_i].unsqueeze(1),
                    lats[j:end_j].unsqueeze(0), lons[j:end_j].unsqueeze(0)
                )

                distances_np[i:end_i, j:end_j] = batch_distances.cpu().numpy()
                distances_np[j:end_j, i:end_i] = distances_np[i:end_i, j:end_j].T  # 矩阵对称性

        # 继续后续停留点识别的逻辑
        i = 0
        stay_points = []

        while i < len(group) - 1:
            curr_distances = distances_np[i, i + 1:]
            close_points = np.where(curr_distances <= distance_threshold)[0]

            if len(close_points) > 0:
                close_points = close_points + i + 1
                j = close_points[-1] + 1

                time_diff = (group.iloc[j - 1]['datetime'] - group.iloc[i]['datetime']).total_seconds()
                if time_diff >= time_threshold:
                    stay_points.append((i, j - 1, time_diff))
                    group.loc[i:j - 1, 'is_stay_point'] = True
                    group.loc[i:j - 1, 'stay_point_id'] = stay_point_count
                    stay_point_count += 1
                    i = j
                    continue

            i += 1

        if merge_stay_points and stay_points:
            non_stay_mask = ~group['is_stay_point']
            filtered_group = group[non_stay_mask].copy()
            for start_idx, end_idx, duration in stay_points:
                start_point = group.iloc[start_idx].copy()
                end_point = group.iloc[end_idx].copy()
                start_point['stay_duration'] = duration
                end_point['stay_duration'] = duration
                filtered_group = pd.concat([filtered_group, pd.DataFrame([start_point, end_point])])

            filtered_group = filtered_group.sort_values('datetime').reset_index(drop=True)
            result_dfs.append(filtered_group)
        else:
            result_dfs.append(group)

    result_df = pd.concat(result_dfs, ignore_index=True)
    print(f"共识别 {stay_point_count} 个停留点")
    print(f"处理后的数据点数: {len(result_df)}")

    return result_df

def interpolate_missing_points(df, max_gap_seconds=300):
    """
    对轨迹中的时间间隙进行插值 (CPU或GPU)
    """
    print(f"开始插值处理 ({'GPU' if USE_GPU else 'CPU'}模式)...")
    # 按用户和轨迹段分组处理
    groups = df.groupby(['user_id', 'segment_trajectory_id'])
    interpolated_dfs = []

    for name, group in tqdm(groups):
        # 排序并重置索引
        group = group.sort_values('datetime').reset_index(drop=True)

        # 计算时间差(秒)
        group['time_diff'] = group['datetime'].diff().dt.total_seconds()

        # 查找需要插值的间隙（大于5秒但小于阈值）
        gaps = group[(group['time_diff'] > 5) & (group['time_diff'] <= max_gap_seconds)]

        if gaps.empty:
            # 没有需要插值的间隙
            interpolated_dfs.append(group.drop('time_diff', axis=1))
            continue

        # 创建插值后的新数据框
        new_rows = []

        # 处理所有间隙
        for idx, gap in gaps.iterrows():
            prev_idx = idx - 1
            prev_row = group.loc[prev_idx]

            # 计算需要插入的点数
            time_diff = gap['time_diff']
            num_points = int(time_diff / 5) - 1  # 每5秒一个点，减1是因为不包括端点

            if num_points <= 0:
                continue

            # 创建时间点
            start_time = prev_row['datetime']
            time_points = [start_time + timedelta(seconds=(i + 1) * 5) for i in range(num_points)]

            # 获取坐标
            lat1, lon1 = prev_row['latitude'], prev_row['longitude']
            lat2, lon2 = gap['latitude'], gap['longitude']

            if USE_GPU:
                # 使用GPU加速线性插值
                t = torch.linspace(0, 1, num_points + 2, device=device)[1:-1]
                lat_interp = (lat1 + (lat2 - lat1) * t).cpu().numpy()
                lon_interp = (lon1 + (lon2 - lon1) * t).cpu().numpy()
            else:
                # CPU线性插值
                lat_interp = np.linspace(lat1, lat2, num_points + 2)[1:-1]
                lon_interp = np.linspace(lon1, lon2, num_points + 2)[1:-1]

            # 线性插值高度
            if 'altitude' in group.columns:
                alt1, alt2 = prev_row['altitude'], gap['altitude']
                alt_interp = np.linspace(alt1, alt2, num_points + 2)[1:-1]
            else:
                alt_interp = [0] * num_points

            # 创建新行
            for i in range(num_points):
                new_row = prev_row.copy()
                new_row['datetime'] = time_points[i]
                new_row['latitude'] = lat_interp[i]
                new_row['longitude'] = lon_interp[i]
                if 'altitude' in group.columns:
                    new_row['altitude'] = alt_interp[i]

                # 计算插值点的距离和速度
                if i == 0:
                    # 第一个插值点与前一个实际点的距离
                    dt = 5  # 5秒
                    if USE_GPU:
                        coords = torch.tensor([[lat1, lon1], [lat_interp[i], lon_interp[i]]], device=device,
                                              dtype=torch.float32)
                        dx = haversine_distance_gpu(coords[0, 0], coords[0, 1], coords[1, 0], coords[1, 1]).item()
                    else:
                        dx = haversine_distance_cpu(lat1, lon1, lat_interp[i], lon_interp[i])
                else:
                    # 与前一个插值点的距离
                    dt = 5  # 5秒
                    if USE_GPU:
                        coords = torch.tensor([[lat_interp[i - 1], lon_interp[i - 1]], [lat_interp[i], lon_interp[i]]],
                                              device=device, dtype=torch.float32)
                        dx = haversine_distance_gpu(coords[0, 0], coords[0, 1], coords[1, 0], coords[1, 1]).item()
                    else:
                        dx = haversine_distance_cpu(lat_interp[i - 1], lon_interp[i - 1], lat_interp[i], lon_interp[i])

                new_row['distance'] = dx
                new_row['speed'] = dx / dt
                new_row['speed_kmh'] = new_row['speed'] * 3.6

                # 重置停留点标记（插值点不是停留点）
                if 'is_stay_point' in new_row:
                    new_row['is_stay_point'] = False
                    new_row['stay_point_id'] = -1

                new_rows.append(new_row)

        # 将新行添加到原始组并重新排序
        if new_rows:
            new_df = pd.concat([group] + [pd.DataFrame([row]) for row in new_rows])
            new_df = new_df.sort_values('datetime').reset_index(drop=True)

            # 重新计算所有点的时间差、距离和速度
            if USE_GPU:
                new_df = recalculate_temporal_features_gpu(new_df)
            else:
                new_df = recalculate_temporal_features_cpu(new_df)

            interpolated_dfs.append(new_df.drop('time_diff', axis=1))

        else:
            interpolated_dfs.append(group.drop('time_diff', axis=1))

            # 合并所有组
    result = pd.concat(interpolated_dfs, ignore_index=True)
    print(f"插值后的数据点数: {len(result)}")

    return result


def recalculate_temporal_features_cpu(df):
    """
    使用CPU重新计算时间差、距离和速度
    """
    # 确保按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)

    # 计算时间差(秒)
    df['time_diff'] = df['datetime'].diff().dt.total_seconds()

    # 第一个点的时间差设为0
    df.loc[0, 'time_diff'] = 0

    # 计算距离
    prev_lat = df['latitude'].shift(1)
    prev_lon = df['longitude'].shift(1)

    # 第一个点距离设为0
    df.loc[0, 'distance'] = 0

    for i in range(1, len(df)):
        df.loc[i, 'distance'] = haversine_distance_cpu(
            prev_lat[i], prev_lon[i], df.loc[i, 'latitude'], df.loc[i, 'longitude']
        )

    # 计算速度
    df.loc[0, 'speed'] = 0
    df.loc[0, 'speed_kmh'] = 0

    mask = df['time_diff'] > 0
    df.loc[mask, 'speed'] = df.loc[mask, 'distance'] / df.loc[mask, 'time_diff']
    df.loc[mask, 'speed_kmh'] = df.loc[mask, 'speed'] * 3.6

    # 替换无穷大和NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def recalculate_temporal_features_gpu(df):
    """
    使用GPU重新计算时间差、距离和速度
    """
    # 确保按时间排序
    df = df.sort_values('datetime').reset_index(drop=True)

    # 计算时间差(秒)
    df['time_diff'] = df['datetime'].diff().dt.total_seconds()

    # 第一个点的时间差设为0
    df.loc[0, 'time_diff'] = 0

    # 使用GPU批量计算距离
    distances = batch_calculate_distance(df)
    df['distance'] = distances

    # 计算速度
    df.loc[0, 'speed'] = 0
    df.loc[0, 'speed_kmh'] = 0

    # 使用矢量化操作计算速度
    mask = df['time_diff'] > 0
    df.loc[mask, 'speed'] = df.loc[mask, 'distance'] / df.loc[mask, 'time_diff']
    df.loc[mask, 'speed_kmh'] = df.loc[mask, 'speed'] * 3.6

    # 替换无穷大和NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    return df


def main():
    # 文件路径
    input_dir = os.path.join("..", "data", "processed")
    output_dir = os.path.join("..", "data", "processed")

    # 检查命令行参数
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    else:
        input_file = os.path.join(input_dir, "geolife_feature_data.csv")

    # 构建输出文件路径
    cleaned_file = os.path.join(output_dir, "geolife_beijing_cleaned_data.csv")
    interpolated_file = os.path.join(output_dir, "geolife_beijing_interpolated_data.csv")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 记录总处理时间
    total_start_time = time.time()

    # 1. 清理数据
    print("开始数据清理过程...")
    start_time = time.time()
    df_cleaned = clean_geolife_data(
        input_file,
        cleaned_file,
        max_speed_kmh=150,  # 最大允许速度(km/h)
        time_gap_minutes=30,  # 轨迹分段的时间间隔阈值(分钟)
        min_trajectory_points=10  # 有效轨迹的最小点数
    )

    if df_cleaned is None:
        print("数据清理失败，程序终止")
        sys.exit(1)

    print(f"数据清理完成，耗时: {time.time() - start_time:.2f}秒")

    # 2. 插值处理
    print("开始插值处理...")
    start_time = time.time()
    df_interpolated = interpolate_missing_points(
        df_cleaned,
        max_gap_seconds=300  # 最大允许插值的时间间隙(5分钟)
    )
    print(f"插值处理完成，耗时: {time.time() - start_time:.2f}秒")

    # 3. 保存插值后的数据
    try:
        print(f"保存插值后的数据: {interpolated_file}")
        df_interpolated.to_csv(interpolated_file, index=False)
        print(f"插值数据已成功保存")
    except Exception as e:
        print(f"保存插值数据出错: {e}")

    print(f"数据清理和插值处理完成! 总耗时: {time.time() - total_start_time:.2f}秒")
    if USE_GPU:
        print(f"GPU加速处理完成!")
    else:
        print(f"CPU处理完成! 若需更快处理速度，请安装支持CUDA的PyTorch版本")


if __name__ == "__main__":
    main()