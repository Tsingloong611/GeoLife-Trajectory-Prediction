import os
import pandas as pd
import glob
from datetime import datetime
import numpy as np


def parse_plt_file(file_path, user_id):
    """
    解析单个PLT文件并返回DataFrame

    参数:
    file_path: PLT文件路径
    user_id: 用户ID

    返回:
    DataFrame包含轨迹点数据
    """
    # 读取PLT文件，跳过前6行(文件头)
    try:
        df = pd.read_csv(file_path, skiprows=6, header=None,
                         names=['latitude', 'longitude', 'zero', 'altitude',
                                'date_days', 'date', 'time'])

        # 提取文件名作为轨迹ID（不包含路径和扩展名）
        trajectory_id = os.path.splitext(os.path.basename(file_path))[0]

        # 合并日期和时间
        df['datetime'] = df['date'] + ' ' + df['time']
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')

        # 添加用户ID和轨迹ID
        df['user_id'] = user_id
        df['trajectory_id'] = trajectory_id

        # 只保留需要的列
        df = df[['user_id', 'trajectory_id', 'datetime', 'latitude', 'longitude', 'altitude']]

        return df

    except Exception as e:
        print(f"解析文件 {file_path} 时出错: {e}")
        return pd.DataFrame()


def process_all_users(data_dir):
    """
    处理所有用户的PLT文件

    参数:
    data_dir: GeoLife数据集根目录

    返回:
    合并的DataFrame包含所有用户的轨迹数据
    """
    all_trajectories = []

    # 获取用户目录列表
    user_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

    for user_id in user_dirs:
        user_path = os.path.join(data_dir, user_id)
        trajectory_path = os.path.join(user_path, 'Trajectory')

        if not os.path.exists(trajectory_path):
            continue

        # 获取所有PLT文件
        plt_files = glob.glob(os.path.join(trajectory_path, '**', '*.plt'), recursive=True)

        print(f"正在处理用户 {user_id}，共有 {len(plt_files)} 个轨迹文件")

        # 处理每个PLT文件
        for plt_file in plt_files:
            df = parse_plt_file(plt_file, user_id)
            if not df.empty:
                all_trajectories.append(df)

    # 合并所有DataFrame
    if all_trajectories:
        merged_df = pd.concat(all_trajectories, ignore_index=True)
        return merged_df
    else:
        return pd.DataFrame()


def add_time_features(df):
    """
    添加时间相关特征
    """
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['day_part'] = pd.cut(df['hour'],
                            bins=[0, 6, 12, 18, 24],
                            labels=['night', 'morning', 'afternoon', 'evening'],
                            include_lowest=True)
    return df


def add_temporal_features(df):
    """
    添加时间差和速度等时序特征
    """
    # 确保按用户、轨迹和时间排序
    df = df.sort_values(['user_id', 'trajectory_id', 'datetime'])

    # 计算时间差(秒)
    df['time_diff'] = df.groupby(['user_id', 'trajectory_id'])['datetime'].diff().dt.total_seconds()

    # 计算Haversine距离(米)
    df['prev_lat'] = df.groupby(['user_id', 'trajectory_id'])['latitude'].shift(1)
    df['prev_lon'] = df.groupby(['user_id', 'trajectory_id'])['longitude'].shift(1)

    # 使用haversine函数计算距离，这里简化为直线距离
    # 实际实现时应使用haversine公式计算球面距离
    df['distance'] = np.nan
    mask = ~df['prev_lat'].isna()

    # 计算两点之间的距离(米)，使用Haversine公式
    R = 6371000  # 地球半径(米)
    df.loc[mask, 'distance'] = np.vectorize(lambda lat1, lon1, lat2, lon2:
                                            2 * R * np.arcsin(np.sqrt(
                                                np.sin((np.radians(lat2) - np.radians(lat1)) / 2) ** 2 +
                                                np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) *
                                                np.sin((np.radians(lon2) - np.radians(lon1)) / 2) ** 2
                                            ))
                                            )(df.loc[mask, 'prev_lat'], df.loc[mask, 'prev_lon'],
                                              df.loc[mask, 'latitude'], df.loc[mask, 'longitude'])

    # 计算速度(m/s)
    df['speed'] = df['distance'] / df['time_diff']

    # 删除辅助列
    df = df.drop(['prev_lat', 'prev_lon'], axis=1)

    return df


def main():
    # 数据目录路径
    data_dir = "../data/raw/Geolife Trajectories 1.3/Data"
    output_dir = "../data/processed"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 处理所有用户数据
    print("开始处理GeoLife数据集...")
    df = process_all_users(data_dir)

    if not df.empty:
        # 保存原始解析数据
        print("保存原始解析数据...")
        df.to_csv(os.path.join(output_dir, "geolife_data.csv"), index=False)

        # 添加时间特征
        print("添加时间特征...")
        df = add_time_features(df)

        # 添加时序特征
        print("添加时序特征...")
        df = add_temporal_features(df)

        # 保存带特征的数据
        print("保存带特征的数据...")
        df.to_csv(os.path.join(output_dir, "geolife_feature_data.csv"), index=False)

        print(f"处理完成! 总共 {len(df)} 条记录")
    else:
        print("未找到任何数据，请检查数据目录路径")


if __name__ == "__main__":
    main()