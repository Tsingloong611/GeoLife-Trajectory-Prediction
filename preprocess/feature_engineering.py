import os
import math
import time
import warnings
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm
import numpy as np

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in reduce")
warnings.filterwarnings("ignore", category=FutureWarning)

# 全局变量：如果设置为True，将强制使用CPU模式
RAPIDS_FALLBACK = False
USE_RAPIDS = True

try:
    import cudf
    import cupy as cp
    import cuspatial
    from cuml.preprocessing import StandardScaler, RobustScaler
    from cuml.model_selection import train_test_split as cuml_train_test_split
    import xgboost as xgb
    from numba import cuda, jit, float32, float64

    # 为了确保在RAPIDS模式下也可以使用sklearn的函数作为备选
    from sklearn.model_selection import train_test_split as sklearn_train_test_split
    from sklearn.preprocessing import StandardScaler as SklearnStandardScaler
    from sklearn.preprocessing import RobustScaler as SklearnRobustScaler

    USE_RAPIDS = True
    print("RAPIDS库已加载，将使用GPU加速计算")

    # 创建CUDA上下文以确保GPU被激活
    cuda_context = cuda.current_context()
    print(f"CUDA设备: {cuda.get_current_device().name}")

    # 获取GPU内存信息
    mempool = cp.get_default_memory_pool()
    print(
        f"GPU初始内存使用情况: {mempool.used_bytes() / 1024 ** 2:.2f} MB / {mempool.total_bytes() / 1024 ** 2:.2f} MB")

    # 使用更大的块大小以提高占用率
    DEFAULT_THREADS_PER_BLOCK = 512  # 可以调整为128, 256, 512等，取决于GPU架构


    # 定义train_test_split函数，优先使用cuML版本
    def train_test_split(*args, **kwargs):
        try:
            return cuml_train_test_split(*args, **kwargs)
        except Exception as e:
            print(f"cuML train_test_split失败，使用sklearn版本: {str(e)}")
            return sklearn_train_test_split(*args, **kwargs)

except ImportError:
    import pandas as cudf  # 使用pandas作为后备
    import numpy as cp
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    import xgboost as xgb

    USE_RAPIDS = False
    print("未找到RAPIDS库，将使用CPU进行计算")



def timer_decorator(func):
    """计时装饰器，用于测量函数执行时间"""

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 执行时间: {end_time - start_time:.2f} 秒")
        return result

    return wrapper


def safe_div(a, b, default=0.0):
    """安全除法，处理被零除的情况"""
    if USE_RAPIDS and isinstance(a, (cp.ndarray, cuda.devicearray.DeviceNDArray)):
        result = cp.zeros_like(a, dtype=cp.float32)
        mask = (b != 0)
        if mask.any():
            result[mask] = a[mask] / b[mask]
        return result
    else:
        return np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=(b != 0))


def safe_values(series):
    if USE_RAPIDS and isinstance(series, cudf.Series):
        try:
            val = series.values
            # 如果是 Cupy 数组，使用 .get() 显式转换为 NumPy 数组
            if hasattr(val, "get"):
                return val.get()
            return val
        except TypeError as e:
            if "String Arrays is not yet implemented" in str(e):
                return np.array([item for item in series])
            else:
                raise
    else:
        return series.values


# 在to_cudf函数中添加调试信息
def to_cudf(df):
    global RAPIDS_FALLBACK, USE_RAPIDS

    if RAPIDS_FALLBACK:
        USE_RAPIDS = False
        return df

    if USE_RAPIDS and not isinstance(df, cudf.DataFrame):
        try:
            print("尝试转换为cuDF，数据形状:", df.shape)
            result = cudf.DataFrame.from_pandas(df)
            print("转换成功，cuDF形状:", result.shape)
            # 检查GPU内存
            mempool = cp.get_default_memory_pool()
            print(f"GPU内存使用: {mempool.used_bytes() / 1024 ** 2:.2f} MB")
            return result
        except Exception as e:
            print(f"转换为cuDF失败: {str(e)}")
            RAPIDS_FALLBACK = True
            USE_RAPIDS = False
            return df
    return df


def to_pandas(df_or_series):
    """将cuDF DataFrame/Series转换为pandas DataFrame/Series，如果已经是pandas则直接返回"""
    if USE_RAPIDS:
        if isinstance(df_or_series, cudf.DataFrame):
            return df_or_series.to_pandas()
        elif isinstance(df_or_series, cudf.Series):
            try:
                return df_or_series.to_pandas()
            except TypeError as e:
                # 如果是字符串数组错误，使用特殊处理
                if "String Arrays is not yet implemented" in str(e):
                    # 转换为列表然后创建pandas Series
                    import pandas as pd
                    return pd.Series([item for item in df_or_series])
                else:
                    raise
        else:
            return df_or_series
    return df_or_series


def is_nan_cudf(value):
    """安全检查cuDF中的值是否为NaN"""
    try:
        # 尝试转换为float并检查是否为NaN
        return np.isnan(float(value))
    except (TypeError, ValueError):
        # 如果无法转换为float，则视为NaN
        return True


def safe_float_cudf(value, default=0.0):
    """安全地将cuDF值转换为float"""
    try:
        float_val = float(value)
        return float_val if not np.isnan(float_val) else default
    except (TypeError, ValueError):
        return default


def safe_compare_cudf(value, threshold, comparison='gt'):
    """
    安全地比较cuDF中的值与阈值

    参数:
    value: 要比较的值
    threshold: 阈值
    comparison: 比较类型, 'gt'(大于), 'lt'(小于), 'abs_gt'(绝对值大于)

    返回:
    比较结果的布尔值
    """
    try:
        float_val = float(value)
        if np.isnan(float_val):
            return False

        if comparison == 'gt':
            return float_val > threshold
        elif comparison == 'lt':
            return float_val < threshold
        elif comparison == 'abs_gt':
            return abs(float_val) > threshold
        else:
            return False
    except (TypeError, ValueError):
        return False


def haversine_distance_gpu(lat1, lon1, lat2, lon2):
    """使用GPU计算Haversine距离（米）"""
    if USE_RAPIDS:
        try:
            result = haversine_distance_cupy(lat1, lon1, lat2, lon2)

            # 如果结果是标量，转换为Series
            if not hasattr(result, 'iloc'):
                result = cudf.Series([float(result)])

            return result
        except Exception as e:
            print(f"GPU Haversine计算出错，回退到numpy: {str(e)}")
            # 转换为numpy数组
            try:
                lat1_np = safe_values(lat1)
                lon1_np = safe_values(lon1)
                lat2_np = safe_values(lat2)
                lon2_np = safe_values(lon2)
                return haversine_distance_numpy(lat1_np, lon1_np, lat2_np, lon2_np)
            except:
                # 如果转换失败，逐个处理
                if isinstance(lat1, cudf.Series):
                    lat1_list = [float(item) for item in lat1.values_host]
                    lon1_list = [float(item) for item in lon1.values_host]
                    lat2_list = [float(item) for item in lat2.values_host]
                    lon2_list = [float(item) for item in lon2.values_host]
                else:
                    lat1_list = [float(item) for item in lat1]
                    lon1_list = [float(item) for item in lon1]
                    lat2_list = [float(item) for item in lat2]
                    lon2_list = [float(item) for item in lon2]
                return haversine_distance_numpy(lat1_list, lon1_list, lat2_list, lon2_list)
    else:
        return haversine_distance_numpy(lat1, lon1, lat2, lon2)


def haversine_distance_cupy(lat1, lon1, lat2, lon2):
    """使用cupy计算Haversine距离（米）"""
    R = 6371000  # 地球半径(米)

    try:
        # 转换为cupy数组
        if isinstance(lat1, cudf.Series):
            lat1_cp = cp.asarray(safe_values(lat1), dtype=cp.float32)
        else:
            lat1_cp = cp.asarray(lat1, dtype=cp.float32)

        if isinstance(lon1, cudf.Series):
            lon1_cp = cp.asarray(safe_values(lon1), dtype=cp.float32)
        else:
            lon1_cp = cp.asarray(lon1, dtype=cp.float32)

        if isinstance(lat2, cudf.Series):
            lat2_cp = cp.asarray(safe_values(lat2), dtype=cp.float32)
        else:
            lat2_cp = cp.asarray(lat2, dtype=cp.float32)

        if isinstance(lon2, cudf.Series):
            lon2_cp = cp.asarray(safe_values(lon2), dtype=cp.float32)
        else:
            lon2_cp = cp.asarray(lon2, dtype=cp.float32)

        # 转换为弧度
        lat1_rad = cp.radians(lat1_cp)
        lon1_rad = cp.radians(lon1_cp)
        lat2_rad = cp.radians(lat2_cp)
        lon2_rad = cp.radians(lon2_cp)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = cp.sin(dlat / 2) ** 2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon / 2) ** 2
        a = cp.minimum(a, 1.0)  # 确保a不大于1，防止数值问题
        c = 2 * cp.arctan2(cp.sqrt(a), cp.sqrt(1.0 - a))

        distance = R * c

        # 将结果转回cuDF Series
        if isinstance(lat1, cudf.Series):
            return cudf.Series(cp.asnumpy(distance))
        return cp.asnumpy(distance)
    except Exception as e:
        print(f"cupy Haversine计算出错，回退到numpy: {str(e)}")
        # 转换为numpy计算
        return haversine_distance_numpy(
            lat1.to_pandas() if isinstance(lat1, cudf.Series) else lat1,
            lon1.to_pandas() if isinstance(lon1, cudf.Series) else lon1,
            lat2.to_pandas() if isinstance(lat2, cudf.Series) else lat2,
            lon2.to_pandas() if isinstance(lon2, cudf.Series) else lon2
        )


def haversine_distance_numpy(lat1, lon1, lat2, lon2):
    """使用numpy计算Haversine距离（米）"""
    R = 6371000  # 地球半径(米)

    # 转换为numpy数组
    lat1_np = np.asarray(lat1, dtype=np.float32)
    lon1_np = np.asarray(lon1, dtype=np.float32)
    lat2_np = np.asarray(lat2, dtype=np.float32)
    lon2_np = np.asarray(lon2, dtype=np.float32)

    # 转换为弧度
    lat1_rad = np.radians(lat1_np)
    lon1_rad = np.radians(lon1_np)
    lat2_rad = np.radians(lat2_np)
    lon2_rad = np.radians(lon2_np)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    a = np.minimum(a, 1.0)  # 确保a不大于1，防止数值问题
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


def calculate_direction_angle(lat1, lon1, lat2, lon2):
    """计算两点之间的方向角（方位角）
    返回值范围: 0-360度，北为0度，顺时针增加
    """
    if USE_RAPIDS:
        try:
            # 使用cupy计算
            lat1_cp = cp.asarray(lat1, dtype=cp.float32)
            lon1_cp = cp.asarray(lon1, dtype=cp.float32)
            lat2_cp = cp.asarray(lat2, dtype=cp.float32)
            lon2_cp = cp.asarray(lon2, dtype=cp.float32)

            # 转换为弧度
            lat1_rad = cp.radians(lat1_cp)
            lon1_rad = cp.radians(lon1_cp)
            lat2_rad = cp.radians(lat2_cp)
            lon2_rad = cp.radians(lon2_cp)

            # 计算方位角
            y = cp.sin(lon2_rad - lon1_rad) * cp.cos(lat2_rad)
            x = cp.cos(lat1_rad) * cp.sin(lat2_rad) - cp.sin(lat1_rad) * cp.cos(lat2_rad) * cp.cos(lon2_rad - lon1_rad)
            bearing = cp.degrees(cp.arctan2(y, x))

            # 转换为0-360度范围
            bearing = (bearing + 360) % 360

            # 将结果转回CPU
            if isinstance(lat1, cudf.Series):
                return cudf.Series(cp.asnumpy(bearing))
            return cp.asnumpy(bearing)
        except Exception as e:
            print(f"GPU方向角计算出错，回退到numpy: {str(e)}")
            # 回退到numpy
            return calculate_direction_angle_numpy(lat1, lon1, lat2, lon2)
    else:
        return calculate_direction_angle_numpy(lat1, lon1, lat2, lon2)


def calculate_direction_angle_numpy(lat1, lon1, lat2, lon2):
    """使用numpy计算方向角"""
    # 使用numpy计算
    lat1_np = np.asarray(lat1, dtype=np.float32)
    lon1_np = np.asarray(lon1, dtype=np.float32)
    lat2_np = np.asarray(lat2, dtype=np.float32)
    lon2_np = np.asarray(lon2, dtype=np.float32)

    # 转换为弧度
    lat1_rad = np.radians(lat1_np)
    lon1_rad = np.radians(lon1_np)
    lat2_rad = np.radians(lat2_np)
    lon2_rad = np.radians(lon2_np)

    # 计算方位角
    y = np.sin(lon2_rad - lon1_rad) * np.cos(lat2_rad)
    x = np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
    bearing = np.degrees(np.arctan2(y, x))

    # 转换为0-360度范围
    bearing = (bearing + 360) % 360

    return bearing


def calculate_turning_angle(prev_angle, curr_angle):
    """计算转弯角度（两个方向角之间的差）
    返回值范围: -180到180度，负值表示左转，正值表示右转
    """
    if USE_RAPIDS:
        try:
            # 使用cupy计算
            prev_angle_cp = cp.asarray(prev_angle, dtype=cp.float32)
            curr_angle_cp = cp.asarray(curr_angle, dtype=cp.float32)

            # 计算角度差
            diff = curr_angle_cp - prev_angle_cp

            # 归一化到-180到180范围
            diff = cp.where(diff > 180, diff - 360, diff)
            diff = cp.where(diff < -180, diff + 360, diff)

            # 将结果转回CPU/cuDF
            if isinstance(prev_angle, cudf.Series):
                return cudf.Series(cp.asnumpy(diff))
            return cp.asnumpy(diff)
        except Exception as e:
            print(f"GPU转弯角度计算出错，回退到numpy: {str(e)}")
            return calculate_turning_angle_numpy(prev_angle, curr_angle)
    else:
        return calculate_turning_angle_numpy(prev_angle, curr_angle)


def calculate_turning_angle_numpy(prev_angle, curr_angle):
    """使用numpy计算转弯角度"""
    # 使用numpy计算
    prev_angle_np = np.asarray(prev_angle, dtype=np.float32)
    curr_angle_np = np.asarray(curr_angle, dtype=np.float32)

    # 计算角度差
    diff = curr_angle_np - prev_angle_np

    # 归一化到-180到180范围
    diff = np.where(diff > 180, diff - 360, diff)
    diff = np.where(diff < -180, diff + 360, diff)

    return diff


def check_rapids_compatibility():
    """检查RAPIDS库是否与当前数据集兼容"""
    global USE_RAPIDS, RAPIDS_FALLBACK

    if RAPIDS_FALLBACK:
        print("由于先前的错误，强制使用CPU模式")
        USE_RAPIDS = False
        return

    if not USE_RAPIDS:
        return

    # 尝试创建一个简单的cuDF DataFrame并执行基本操作
    try:
        test_df = cudf.DataFrame({
            'a': [1, 2, 3],
            'b': [4.0, 5.0, 6.0],
            'c': ['x', 'y', 'z']
        })

        # 尝试一些基本操作
        test_df['d'] = test_df['a'] * 2
        test_df = test_df.sort_values('b')

        # 尝试to_pandas
        test_pd = test_df.to_pandas()

        # 尝试Series操作
        s1 = test_df['a']
        s2 = s1 * 3

        print("RAPIDS兼容性检查通过")
    except Exception as e:
        print(f"RAPIDS兼容性检查失败: {str(e)}")
        print("切换到CPU模式")
        USE_RAPIDS = False
        RAPIDS_FALLBACK = True


# 将函数移到模块级别
def haversine_distance_for_features(lat1, lon1, lat2, lon2):
    """
    计算Haversine距离的通用函数
    """
    R = 6371000.0  # 地球半径(米)

    # 转换为弧度
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    a = min(a, 1.0)  # 确保a不大于1
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return R * c


def calculate_direction_angle_for_features(lat1, lon1, lat2, lon2):
    """
    计算方向角的通用函数
    """
    y = np.sin(np.radians(lon2 - lon1)) * np.cos(np.radians(lat2))
    x = np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) - \
        np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(lon2 - lon1))

    bearing = np.degrees(np.arctan2(y, x))
    return (bearing + 360) % 360


def calculate_turning_angle_for_features(prev_angle, curr_angle):
    """
    计算转弯角度的通用函数
    """
    diff = curr_angle - prev_angle
    if diff > 180:
        diff -= 360
    elif diff < -180:
        diff += 360
    return diff


def calculate_curvature_for_features(lat1, lon1, lat2, lon2, lat3, lon3):
    """
    计算曲率的通用函数
    """

    def haversine_distance(lat_a, lon_a, lat_b, lon_b):
        R = 6371000.0  # 地球半径(米)
        lat1, lon1 = np.radians(lat_a), np.radians(lon_a)
        lat2, lon2 = np.radians(lat_b), np.radians(lon_b)
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    # 计算三点间距离
    a = haversine_distance(lat1, lon1, lat2, lon2)
    b = haversine_distance(lat2, lon2, lat3, lon3)
    c = haversine_distance(lat1, lon1, lat3, lon3)

    # 检查三角形合法性
    if a <= 0 or b <= 0 or c <= 0 or a + b <= c or a + c <= b or b + c <= a:
        return np.nan

    # 海伦公式计算面积
    s = (a + b + c) / 2
    area_squared = s * (s - a) * (s - b) * (s - c)

    if area_squared <= 0:
        return np.nan

    area = np.sqrt(area_squared)
    radius = (a * b * c) / (4 * area)

    return 1 / radius if radius > 1e-6 else np.nan


def process_single_trajectory_group(group):
    """
    处理单个轨迹组的特征提取
    注意：需要使用模块级别的函数
    """
    if len(group) < 3:
        return group

    # 深拷贝以避免修改原始数据
    group = group.copy()

    # 计算方向角
    for i in range(1, len(group)):
        try:
            lat1, lon1 = group.iloc[i - 1]['latitude'], group.iloc[i - 1]['longitude']
            lat2, lon2 = group.iloc[i]['latitude'], group.iloc[i]['longitude']

            # 避免重复计算
            if abs(lat1 - lat2) < 1e-8 and abs(lon1 - lon2) < 1e-8:
                continue

            # 计算方向角
            direction_angle = calculate_direction_angle_for_features(lat1, lon1, lat2, lon2)
            group.loc[group.index[i], 'direction_angle'] = direction_angle
        except Exception:
            continue

    # 计算转弯角度
    direction_angles = group['direction_angle'].values
    for i in range(2, len(group)):
        try:
            prev_angle = direction_angles[i - 1]
            curr_angle = direction_angles[i]

            if np.isnan(prev_angle) or np.isnan(curr_angle):
                continue

            # 计算转弯角度
            turning = calculate_turning_angle_for_features(prev_angle, curr_angle)
            group.loc[group.index[i], 'turning_angle'] = turning

            # 识别转弯点
            if abs(turning) > 30:
                group.loc[group.index[i], 'is_turn'] = 1
                group.loc[group.index[i], 'turn_sharpness'] = abs(turning)
        except Exception:
            continue

    # 计算曲率
    for i in range(1, len(group) - 1):
        try:
            lat1, lon1 = group.iloc[i - 1]['latitude'], group.iloc[i - 1]['longitude']
            lat2, lon2 = group.iloc[i]['latitude'], group.iloc[i]['longitude']
            lat3, lon3 = group.iloc[i + 1]['latitude'], group.iloc[i + 1]['longitude']

            curvature = calculate_curvature_for_features(lat1, lon1, lat2, lon2, lat3, lon3)
            group.loc[group.index[i], 'curvature'] = curvature
        except Exception:
            continue

    # 计算直线度
    start_lat, start_lon = group.iloc[0]['latitude'], group.iloc[0]['longitude']
    end_lat, end_lon = group.iloc[-1]['latitude'], group.iloc[-1]['longitude']

    # 计算起点到终点的直线距离
    straight_distance = haversine_distance_for_features(start_lat, start_lon, end_lat, end_lon)
    path_length = group['distance'].sum()

    # 计算直线度
    straightness = straight_distance / path_length if path_length > 0 else 0
    group['straightness'] = min(straightness, 1.0)

    return group


@timer_decorator
def extract_spatial_features(df):
    """提取空间特征：方向角、转弯角度、轨迹形状等"""
    print("提取空间特征...")

    # 确认RAPIDS兼容性
    check_rapids_compatibility()

    # 确保数据是pandas格式
    df = to_pandas(df)

    # 确保按用户、轨迹和时间排序
    df = df.sort_values(['user_id', 'segment_trajectory_id', 'datetime'])

    # 初始化新特征
    df['direction_angle'] = np.nan
    df['turning_angle'] = np.nan
    df['curvature'] = np.nan
    df['is_turn'] = 0
    df['turn_sharpness'] = np.nan
    df['straightness'] = np.nan

    # 使用更高效的方法处理分组
    processed_groups = []
    for (user, traj), group in tqdm(df.groupby(['user_id', 'segment_trajectory_id']),
                                    desc="提取空间特征"):
        if len(group) >= 3:
            processed_groups.append(process_single_trajectory_group(group))
        else:
            processed_groups.append(group)

    # 合并处理后的数据
    if processed_groups:
        df = pd.concat(processed_groups).sort_index()

    # 如果原本使用RAPIDS，尝试将结果转回cuDF
    if USE_RAPIDS:
        try:
            df = cudf.DataFrame.from_pandas(df)
        except Exception as e:
            print(f"将处理后的数据转回cuDF时出错: {str(e)}")

    return df


def calculate_curvature_cpu(group, pdf):
    """CPU版本计算曲率，用于在RAPIDS版本失败时回退"""
    try:
        print(f"calculate_curvature_cpu调用：group类型={type(group)}")

        if not isinstance(group, pd.DataFrame):
            # If it's an array, convert back to DataFrame
            print("需要将group转换为DataFrame")
            # 确保group是array-like且可转为DataFrame
            if hasattr(group, 'shape') and hasattr(group, 'dtype'):
                try:
                    group = pd.DataFrame(group, columns=pdf.columns)
                    print(f"成功转换：新group类型={type(group)}")
                except Exception as conv_error:
                    print(f"转换失败: {str(conv_error)}")
                    return  # 返回，不继续处理
            else:
                print(f"group不是array-like: {type(group)}")
                return  # 返回，不继续处理
        for i in range(1, len(group) - 1):
            p1 = (group.iloc[i - 1]['latitude'], group.iloc[i - 1]['longitude'])
            p2 = (group.iloc[i]['latitude'], group.iloc[i]['longitude'])
            p3 = (group.iloc[i + 1]['latitude'], group.iloc[i + 1]['longitude'])

            # 计算三点间的距离
            a_result = haversine_distance_gpu(p1[0], p1[1], p2[0], p2[1])
            b_result = haversine_distance_gpu(p2[0], p2[1], p3[0], p3[1])
            c_result = haversine_distance_gpu(p1[0], p1[1], p3[0], p3[1])

            # 安全地获取标量值
            a = float(a_result.iloc[0]) if hasattr(a_result, 'iloc') else float(a_result)
            b = float(b_result.iloc[0]) if hasattr(b_result, 'iloc') else float(b_result)
            c = float(c_result.iloc[0]) if hasattr(c_result, 'iloc') else float(c_result)

            # 检查三角形是否合法
            if a <= 0 or b <= 0 or c <= 0 or a + b <= c or a + c <= b or b + c <= a:
                continue

            # 海伦公式计算三角形面积
            s = (a + b + c) / 2
            area_squared = s * (s - a) * (s - b) * (s - c)

            # 检查面积是否为正
            if area_squared <= 0:
                continue

            area = np.sqrt(area_squared)

            # 计算曲率（半径的倒数）
            radius = (a * b * c) / (4 * area)
            if radius > 1e-6:  # 避免非常小的半径导致的极大曲率
                curvature = 1 / radius
                pdf.loc[group.index[i], 'curvature'] = curvature
    except Exception as e:
        print(f"CPU计算曲率出错: {str(e)}")
        print(f"错误位置: group类型={type(group)}")
        # 打印更详细的错误堆栈
        import traceback
        traceback.print_exc()


@timer_decorator
def extract_time_features(df):
    """提取时间特征：周期性编码、时段分类等"""
    print("提取时间特征...")

    # 确认RAPIDS兼容性
    check_rapids_compatibility()

    # 确保数据是cuDF格式
    USE_RAPIDS = True
    if USE_RAPIDS:
        try:
            df = to_cudf(df)
        except Exception as e:
            print(f"转换为cuDF格式失败: {str(e)}")
            global RAPIDS_FALLBACK
            RAPIDS_FALLBACK = True
            USE_RAPIDS = False

    # 提取基本时间组件
    # 注意：cuDF的datetime处理与pandas略有不同
    if USE_RAPIDS:
        try:
            # 如果使用cuDF，需要单独处理每个时间分量
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month

            # 转为pandas处理is_weekend（因为cuDF不支持isin）
            pdf = df.to_pandas()
            pdf['is_weekend'] = pdf['day_of_week'].isin([5, 6]).astype(int)
            df = cudf.DataFrame.from_pandas(pdf)
        except Exception as e:
            print(f"使用cuDF处理时间分量失败: {str(e)}")
            # 回退到pandas
            df = to_pandas(df)
            df['hour'] = df['datetime'].dt.hour
            df['minute'] = df['datetime'].dt.minute
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['day_of_month'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    else:
        # pandas版本
        df['hour'] = df['datetime'].dt.hour
        df['minute'] = df['datetime'].dt.minute
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['month'] = df['datetime'].dt.month
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    # 周期性时间编码（正弦和余弦变换）
    # 小时（24小时周期）
    hours = df['hour'] + df['minute'] / 60.0  # 小时的小数表示

    if USE_RAPIDS:
        try:
            # 使用cupy加速三角函数计算
            hours_cp = cp.array(to_pandas(hours).values, dtype=cp.float32)
            df['hour_sin'] = cudf.Series(cp.sin(2 * cp.pi * hours_cp / 24).get())
            df['hour_cos'] = cudf.Series(cp.cos(2 * cp.pi * hours_cp / 24).get())

            # 星期（7天周期）
            days_cp = cp.array(to_pandas(df['day_of_week']).values, dtype=cp.float32)
            df['day_sin'] = cudf.Series(cp.sin(2 * cp.pi * days_cp / 7).get())
            df['day_cos'] = cudf.Series(cp.cos(2 * cp.pi * days_cp / 7).get())

            # 月份（12个月周期）
            months_cp = cp.array(to_pandas(df['month']).values, dtype=cp.float32)
            df['month_sin'] = cudf.Series(cp.sin(2 * cp.pi * months_cp / 12).get())
            df['month_cos'] = cudf.Series(cp.cos(2 * cp.pi * months_cp / 12).get())
        except Exception as e:
            print(f"使用cupy加速三角函数计算失败: {str(e)}")
            # 回退到numpy
            df = to_pandas(df)
            df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
            df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    else:
        # CPU 计算
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)

        # 星期（7天周期）
        days = df['day_of_week']
        df['day_sin'] = np.sin(2 * np.pi * days / 7)
        df['day_cos'] = np.cos(2 * np.pi * days / 7)

        # 月份（12个月周期）
        months = df['month']
        df['month_sin'] = np.sin(2 * np.pi * months / 12)
        df['month_cos'] = np.cos(2 * np.pi * months / 12)

    # 一天中的时段分类
    # 由于需要多次使用条件逻辑，对cuDF较复杂，转为pandas处理
    pdf = to_pandas(df)

    # 0-6: 深夜，6-9: 早晨，9-12: 上午，12-14: 中午，14-18: 下午，18-22: 晚上，22-24: 夜晚
    hour_ranges = [(0, 6, 'late_night'), (6, 9, 'early_morning'),
                   (9, 12, 'morning'), (12, 14, 'noon'),
                   (14, 18, 'afternoon'), (18, 22, 'evening'),
                   (22, 24, 'night')]

    # 初始化时段列
    pdf['day_part'] = None

    # 根据小时确定时段
    for start, end, label in hour_ranges:
        mask = (pdf['hour'] >= start) & (pdf['hour'] < end)
        pdf.loc[mask, 'day_part'] = label

    # 通勤时间（工作日早上7-9点，下午17-19点）
    morning_commute = (pdf['hour'] >= 7) & (pdf['hour'] < 9) & ~pdf['is_weekend']
    evening_commute = (pdf['hour'] >= 17) & (pdf['hour'] < 19) & ~pdf['is_weekend']
    pdf['is_commute_time'] = (morning_commute | evening_commute).astype(int)

    # 添加工作日/休息日特征
    holidays = [
        # 添加中国法定节假日，这里仅作示例
        # 可根据实际数据时间范围添加相应的节假日
        "2008-01-01", "2008-02-07", "2008-02-08", "2008-02-09",  # 2008年部分节假日
        "2008-05-01", "2008-06-08", "2008-09-15", "2008-10-01",
        "2009-01-01", "2009-01-26", "2009-01-27", "2009-01-28",  # 2009年部分节假日
        "2009-05-01", "2009-05-28", "2009-10-01", "2009-10-02"
    ]
    pdf['is_holiday'] = pdf['datetime'].dt.strftime('%Y-%m-%d').isin(holidays).astype(int)

    # 合并周末和节假日
    pdf['is_rest_day'] = ((pdf['is_weekend'] == 1) | (pdf['is_holiday'] == 1)).astype(int)

    # 将处理后的pandas DataFrame转回cuDF
    if USE_RAPIDS:
        try:
            df = cudf.DataFrame.from_pandas(pdf)
        except Exception as e:
            print(f"将处理后的数据转回cuDF时出错: {str(e)}")
            df = pdf
    else:
        df = pdf

    return df


def calculate_speed_trend_for_features(speeds):
    """
    使用滑动窗口计算速度变化趋势的通用函数
    返回速度趋势数组：1(加速)、-1(减速)、0(保持)
    """
    if len(speeds) < 5:
        return np.zeros(len(speeds), dtype=np.int8)

    # 初始化趋势数组
    trends = np.zeros(len(speeds), dtype=np.int8)

    # 使用滑动窗口计算
    window_size = 3
    for i in range(2, len(speeds)):
        # 使用最近3个点的速度判断趋势
        recent_speeds = speeds[i - 2:i + 1]

        # 线性回归斜率
        x = np.arange(window_size, dtype=np.float32)
        try:
            # 使用numpy的polyfit进行线性回归
            slope = np.polyfit(x, recent_speeds, 1)[0]

            # 判断趋势
            if slope > 0.5:
                trends[i] = 1  # 加速
            elif slope < -0.5:
                trends[i] = -1  # 减速
        except Exception:
            continue

    return trends


def calculate_displacement_for_features(group):
    """
    计算相对于轨迹起点的位移的通用函数
    """
    try:
        # 获取起点坐标
        start_lat = group.iloc[0]['latitude']
        start_lon = group.iloc[0]['longitude']

        # 跳过无效坐标
        if pd.isna(start_lat) or pd.isna(start_lon):
            return pd.Series(np.nan, index=group.index)

        # 计算每个点到起点的距离
        curr_lats = group['latitude'].values
        curr_lons = group['longitude'].values

        # 使用Haversine距离计算
        distances = np.array([
            haversine_distance_for_features(start_lat, start_lon, lat, lon)
            for lat, lon in zip(curr_lats, curr_lons)
        ])

        # 确保返回的Series使用原始索引
        return pd.Series(distances, index=group.index)
    except Exception as e:
        print(f"位移计算出错: {str(e)}")
        return pd.Series(np.nan, index=group.index)

def process_single_sequential_group(group):
    """
    处理单个轨迹组的序列特征提取的通用函数
    """
    # 深拷贝以避免修改原始数据
    group = group.copy()

    # 计算速度变化率和加速度
    group['speed_change'] = group['speed_kmh'].diff()

    # 安全计算加速度（处理时间差为0的情况）
    time_diff = group['time_diff'].diff()
    valid_mask = (time_diff > 1e-6)
    group.loc[valid_mask, 'acceleration'] = group.loc[valid_mask, 'speed_change'] / time_diff[valid_mask]

    # 计算滑动窗口特征
    window_sizes = [3, 5, 10]
    for size in window_sizes:
        # 速度滑动窗口统计
        group[f'speed_mean_{size}'] = group['speed_kmh'].rolling(window=size, min_periods=1).mean()
        group[f'speed_std_{size}'] = group['speed_kmh'].rolling(window=size, min_periods=1).std().fillna(0)
        group[f'speed_max_{size}'] = group['speed_kmh'].rolling(window=size, min_periods=1).max()
        group[f'speed_min_{size}'] = group['speed_kmh'].rolling(window=size, min_periods=1).min()

        # 距离滑动窗口统计
        group[f'distance_sum_{size}'] = group['distance'].rolling(window=size, min_periods=1).sum()

    # 速度趋势计算
    group['speed_trend'] = calculate_speed_trend_for_features(
        group['speed_kmh'].fillna(method='ffill').fillna(method='bfill').values
    )

    # 转弯特征处理
    if 'turning_angle' in group.columns:
        group['turning_angle'] = pd.to_numeric(group['turning_angle'], errors='coerce').fillna(0)
        for size in window_sizes:
            group[f'turn_mean_{size}'] = group['turning_angle'].rolling(window=size, min_periods=1).mean()
            group[f'turn_abs_mean_{size}'] = group['turning_angle'].abs().rolling(window=size, min_periods=1).mean()

    return group


@timer_decorator
def extract_sequential_features(df):
    """提取序列特征：滑动窗口统计特征、速度趋势等"""
    print("提取序列特征...")

    # 确认RAPIDS兼容性
    check_rapids_compatibility()

    # 确保数据是pandas格式
    df = to_pandas(df)

    # 确保按用户、轨迹段和时间排序
    df = df.sort_values(['user_id', 'segment_trajectory_id', 'datetime'])

    # 使用更高效的方法处理分组
    processed_groups = []
    for (user, traj), group in tqdm(df.groupby(['user_id', 'segment_trajectory_id']),
                                    desc="提取序列特征"):
        if len(group) >= 3:
            processed_groups.append(process_single_sequential_group(group))
        else:
            processed_groups.append(group)

    # 合并处理后的数据
    if processed_groups:
        df = pd.concat(processed_groups).sort_index()

    # 计算位移特征 - 修改方法以确保索引兼容
    displacement_series = df.groupby(['user_id', 'segment_trajectory_id']).apply(
        lambda group: calculate_displacement_for_features(group)
    ).reset_index(level=[0, 1], drop=True)

    # 确保插入的列具有完全匹配的索引
    df['displacement'] = displacement_series

    # 如果原本使用RAPIDS，尝试将结果转回cuDF
    if USE_RAPIDS:
        try:
            df = cudf.DataFrame.from_pandas(df)
        except Exception as e:
            print(f"将处理后的数据转回cuDF时出错: {str(e)}")

    return df


@timer_decorator
def feature_selection_with_xgboost(df, target_col='next_latitude', n_features=20):
    """
    使用XGBoost进行特征选择

    参数:
    df: 带有所有特征的DataFrame
    target_col: 目标列（例如：'next_latitude'、'next_longitude'）
    n_features: 选择的特征数量

    返回:
    重要特征列表及其重要性得分
    """
    print(f"使用XGBoost选择预测{target_col}的重要特征...")

    # 确认RAPIDS兼容性
    check_rapids_compatibility()

    # 将数据转为pandas，确保与scikit-learn兼容
    df = to_pandas(df)

    # 准备特征和目标列
    # 将所有数值特征作为输入特征
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # 移除不应该作为特征的列
    exclude_cols = ['user_id', 'trajectory_id', 'segment_trajectory_id', target_col,
                    'latitude', 'longitude', 'datetime', 'next_latitude', 'next_longitude']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    # 确保所有特征列在数据中
    feature_cols = [col for col in feature_cols if col in df.columns]

    # 处理特征数据
    # 复制数据以避免修改原始DataFrame
    X_df = df[feature_cols].copy()

    # 替换无穷值和NAType为NaN
    if USE_RAPIDS:
        # cuDF处理
        for col in X_df.columns:
            # 检测和处理无限值
            X_df[col] = X_df[col].astype('float32')  # 确保类型一致
            try:
                # 尝试查找无限值
                mask_inf = (X_df[col] == float('inf')) | (X_df[col] == float('-inf'))
                X_df[col] = X_df[col].mask(mask_inf, None)  # 使用None替换无穷值
            except Exception:
                # 如果无法直接比较，使用更安全的方法
                X_df[col] = X_df[col].replace([np.inf, -np.inf], np.nan)
    else:
        # pandas处理
        X_df = X_df.replace([np.inf, -np.inf], np.nan)

    # 使用各列的中位数填充NaN值
    for col in X_df.columns:
        try:
            median_val = X_df[col].median()
            if pd.isna(median_val):  # 如果中位数是NaN
                median_val = 0.0
            X_df[col] = X_df[col].fillna(median_val)
        except Exception as e:
            print(f"填充列 {col} 的NaN值时出错: {str(e)}")
            X_df[col] = X_df[col].fillna(0)  # 出错时用0填充

    # 准备目标变量
    y = df[target_col]

    # 删除目标变量中的NaN行
    mask = ~y.isna()
    X = X_df[mask]
    y = y[mask]

    # 划分训练和测试集
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        print(f"使用标准train_test_split失败: {str(e)}")
        # 如果导入的train_test_split失败，直接使用sklearn版本
        from sklearn.model_selection import train_test_split as sklearn_tts
        X_train, X_test, y_train, y_test = sklearn_tts(X, y, test_size=0.2, random_state=42)

    # 创建XGBoost模型，如果可用则使用GPU
    try:
        if USE_RAPIDS:
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                tree_method='gpu_hist',  # 使用GPU加速的树构建方法
                gpu_id=0,  # 指定GPU ID
                max_depth=6,  # 限制树深度
                subsample=0.8,  # 使用部分样本
                colsample_bytree=0.8  # 使用部分特征
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8
            )

        # 训练模型（使用early_stopping可提前停止训练）
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=10,
            verbose=False
        )

        # 获取特征重要性
        importance = model.feature_importances_

        # 创建特征重要性DataFrame
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': importance
        })

        # 按重要性排序
        feature_importance = feature_importance.sort_values('importance', ascending=False)

        # 选择前N个重要特征
        top_features = feature_importance.head(n_features)

        print(f"选择的前{n_features}个特征:")
        for i, (feature, importance) in enumerate(zip(top_features['feature'], top_features['importance'])):
            print(f"{i + 1}. {feature}: {importance:.6f}")

        # 如果使用RAPIDS，转回cuDF格式
        if USE_RAPIDS:
            try:
                top_features = cudf.DataFrame.from_pandas(top_features)
            except Exception as e:
                print(f"将特征重要性结果转回cuDF失败: {str(e)}")

        return top_features
    except Exception as e:
        print(f"XGBoost特征选择失败: {str(e)}")
        # 返回输入特征的前N个作为备选
        feature_importance = pd.DataFrame({
            'feature': feature_cols[:n_features],
            'importance': [1.0 / i for i in range(1, min(len(feature_cols), n_features) + 1)]
        })
        return feature_importance


@timer_decorator
def prepare_prediction_dataset(df, window_size=5):
    """
    准备用于预测模型的数据集，添加下一个位置作为目标变量

    参数:
    df: 带有所有特征的DataFrame
    window_size: 用于创建特征窗口的大小

    返回:
    准备好的数据集
    """
    print("准备预测数据集...")

    # 确认RAPIDS兼容性
    check_rapids_compatibility()

    # 确保按用户、轨迹段和时间排序
    if USE_RAPIDS:
        try:
            df = to_cudf(df)
            df = df.sort_values(['user_id', 'segment_trajectory_id', 'datetime'])

            # 添加下一个位置作为目标变量
            # cuDF的shift操作有所不同，需要分组处理
            pdf = df.to_pandas()
            pdf['next_latitude'] = pdf.groupby(['user_id', 'segment_trajectory_id'])['latitude'].shift(-1)
            pdf['next_longitude'] = pdf.groupby(['user_id', 'segment_trajectory_id'])['longitude'].shift(-1)

            # 移除最后一个点（没有下一个位置）
            pdf = pdf.dropna(subset=['next_latitude', 'next_longitude'])

            # 转回cuDF
            try:
                df = cudf.DataFrame.from_pandas(pdf)
            except Exception as e:
                print(f"将预测数据集转回cuDF失败: {str(e)}")
                df = pdf
        except Exception as e:
            print(f"使用cuDF处理预测数据集失败: {str(e)}")
            # 回退到pandas
            df = to_pandas(df)
            df = df.sort_values(['user_id', 'segment_trajectory_id', 'datetime'])
            df['next_latitude'] = df.groupby(['user_id', 'segment_trajectory_id'])['latitude'].shift(-1)
            df['next_longitude'] = df.groupby(['user_id', 'segment_trajectory_id'])['longitude'].shift(-1)
            df = df.dropna(subset=['next_latitude', 'next_longitude'])
    else:
        df = df.sort_values(['user_id', 'segment_trajectory_id', 'datetime'])
        df['next_latitude'] = df.groupby(['user_id', 'segment_trajectory_id'])['latitude'].shift(-1)
        df['next_longitude'] = df.groupby(['user_id', 'segment_trajectory_id'])['longitude'].shift(-1)
        df = df.dropna(subset=['next_latitude', 'next_longitude'])

    return df


@timer_decorator
def encode_categorical_features(df):
    """对分类特征进行编码"""
    print("对分类特征进行编码...")

    # 确认RAPIDS兼容性
    check_rapids_compatibility()

    # 获取所有分类特征
    cat_features = ['day_part']

    # 对每个分类特征进行One-Hot编码
    if USE_RAPIDS:
        try:
            # cuDF的get_dummies与pandas略有不同
            df = to_cudf(df)

            for feature in cat_features:
                if feature in df.columns:
                    # 过滤出非空值
                    valid_mask = (df[feature].fillna('') != '')
                    if valid_mask.sum() > 0:
                        # 创建哑变量（One-Hot编码）
                        try:
                            # cuDF的get_dummies
                            dummies = cudf.get_dummies(df[feature].fillna(''), prefix=feature)
                            # 合并回原数据框
                            df = df.join(dummies)
                        except Exception as e:
                            print(f"使用cuDF.get_dummies失败: {str(e)}")
                            # 回退到pandas
                            pdf = df.to_pandas()
                            dummies = pd.get_dummies(pdf[feature].fillna(''), prefix=feature)
                            pdf = pd.concat([pdf, dummies], axis=1)
                            df = cudf.DataFrame.from_pandas(pdf)
        except Exception as e:
            print(f"使用cuDF处理分类特征失败: {str(e)}")
            # 回退到pandas
            df = to_pandas(df)
            for feature in cat_features:
                if feature in df.columns:
                    # 过滤出非空值
                    valid_mask = df[feature].notna() & (df[feature] != '')
                    if valid_mask.sum() > 0:
                        # 创建哑变量（One-Hot编码）
                        dummies = pd.get_dummies(df.loc[valid_mask, feature], prefix=feature)
                        # 合并回原数据框
                        df = pd.concat([df, dummies], axis=1)
    else:
        for feature in cat_features:
            if feature in df.columns:
                # 过滤出非空值
                valid_mask = df[feature].notna() & (df[feature] != '')
                if valid_mask.sum() > 0:
                    # 创建哑变量（One-Hot编码）
                    dummies = pd.get_dummies(df.loc[valid_mask, feature], prefix=feature)
                    # 合并回原数据框
                    df = pd.concat([df, dummies], axis=1)

    return df


@timer_decorator
def normalize_features(df, output_file):
    """对数值特征进行标准化/归一化，处理异常值"""
    print("标准化数值特征...")

    # 确认RAPIDS兼容性
    check_rapids_compatibility()

    # 转为pandas处理，确保兼容性
    df = to_pandas(df)

    # 选择需要标准化的数值特征
    # 排除ID、日期时间和目标变量
    exclude_cols = ['user_id', 'trajectory_id', 'segment_trajectory_id',
                    'datetime', 'next_latitude', 'next_longitude',
                    'latitude', 'longitude']

    # 获取所有数值列
    num_cols = df.select_dtypes(include=['number']).columns

    # 过滤需要标准化的列
    scale_cols = [col for col in num_cols if col not in exclude_cols]

    # 创建副本
    df_scaled = df.copy()

    # 对数值特征先进行极值和异常值处理
    for col in scale_cols:
        try:
            # 替换无穷值为NaN
            df_scaled[col] = df_scaled[col].replace([np.inf, -np.inf], np.nan)

            # 计算中位数和四分位距用于异常值检测
            # 先剔除NaN值再计算统计量
            temp_series = df_scaled[col].dropna()
            if temp_series.empty:
                # 如果列全是NaN，用0填充并跳过异常值处理
                df_scaled[col] = df_scaled[col].fillna(0)
                continue

            median_val = temp_series.median()
            q1 = temp_series.quantile(0.25)
            q3 = temp_series.quantile(0.75)
            iqr = q3 - q1

            # 定义异常值上下限
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr

            # 将异常值替换为上下限值
            df_scaled.loc[df_scaled[col] < lower_bound, col] = lower_bound
            df_scaled.loc[df_scaled[col] > upper_bound, col] = upper_bound

            # 用中位数填充NaN值
            df_scaled[col] = df_scaled[col].fillna(median_val)
        except Exception as e:
            print(f"处理列 {col} 的异常值时出错: {str(e)}")
            # 出错时简单地用0填充
            df_scaled[col] = df_scaled[col].fillna(0)

    try:
        # 尝试使用RobustScaler
        if USE_RAPIDS:
            try:
                from cuml.preprocessing import RobustScaler
                # 尝试使用cuML的RobustScaler
                scaler = RobustScaler()
                # 转为numpy数组进行处理
                scale_data = df_scaled[scale_cols].values
                scaled_values = scaler.fit_transform(scale_data)
                # 将结果写回DataFrame
                df_scaled[scale_cols] = scaled_values
            except (ImportError, Exception) as e:
                print(f"cuML RobustScaler失败，使用sklearn版本: {str(e)}")
                # 如果cuML版本失败，使用sklearn版本
                from sklearn.preprocessing import RobustScaler
                scaler = RobustScaler()
                df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        else:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
    except Exception as e:
        print(f"RobustScaler失败，尝试StandardScaler: {str(e)}")
        try:
            # 如果RobustScaler失败，尝试StandardScaler
            if USE_RAPIDS:
                try:
                    from cuml.preprocessing import StandardScaler as CumlStandardScaler
                    scaler = CumlStandardScaler()
                    scale_data = df_scaled[scale_cols].values
                    scaled_values = scaler.fit_transform(scale_data)
                    df_scaled[scale_cols] = scaled_values
                except (ImportError, Exception) as e:
                    print(f"cuML StandardScaler失败，使用sklearn版本: {str(e)}")
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                    df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
            else:
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                df_scaled[scale_cols] = scaler.fit_transform(df_scaled[scale_cols])
        except Exception as e2:
            print(f"标准化失败，使用Z-score手动标准化: {str(e2)}")
            # 如果StandardScaler也失败，手动实现简单的Z-score标准化
            for col in scale_cols:
                mean_val = df_scaled[col].mean()
                std_val = df_scaled[col].std()
                if std_val > 0:  # 避免除以零
                    df_scaled[col] = (df_scaled[col] - mean_val) / std_val

    # 再次检查是否还有无穷值
    for col in scale_cols:
        if (df_scaled[col] == np.inf).any() or (df_scaled[col] == -np.inf).any():
            print(f"警告: 列 {col} 在标准化后仍包含无穷值，将它们替换为0")
            df_scaled[col] = df_scaled[col].replace([np.inf, -np.inf], 0)

    # 保存缩放后的数据
    try:
        df_scaled.to_csv(output_file, index=False)
        print(f"标准化后的数据已保存至: {output_file}")
    except Exception as e:
        print(f"保存标准化数据时出错: {str(e)}")
        # 尝试更安全的保存方式
        backup_file = output_file.replace('.csv', '_backup.csv')
        df_scaled.to_csv(backup_file, index=False, na_rep='0')
        print(f"数据已保存到备份文件: {backup_file}")

    # 如果使用RAPIDS，转回cuDF格式
    if USE_RAPIDS:
        try:
            df_scaled = cudf.DataFrame.from_pandas(df_scaled)
        except Exception as e:
            print(f"转换回cuDF失败: {str(e)}")

    return df_scaled


def prepare_prediction_sample(group_data, input_length, output_length, feature_cols):
    """
    为单个轨迹组准备预测样本的通用函数

    参数:
    group_data: 单个轨迹组的DataFrame
    input_length: 输入序列长度
    output_length: 输出序列长度
    feature_cols: 特征列表

    返回:
    样本列表
    """
    # 确保按时间排序
    group = group_data.sort_values('datetime').reset_index(drop=True)

    # 跳过长度不足的轨迹
    if len(group) < input_length + output_length:
        return []

    # 预分配当前轨迹的样本列表
    trajectory_samples = []

    # 使用滑动窗口
    for i in range(len(group) - input_length - output_length + 1):
        # 输入窗口
        input_window = group.iloc[i:i + input_length]

        # 输出窗口 (只关注经纬度)
        output_window = group.iloc[i + input_length:i + input_length + output_length][['latitude', 'longitude']]

        # 创建样本字典
        sample = {
            'user_id': group.iloc[0]['user_id'],
            'traj_id': group.iloc[0]['segment_trajectory_id'],
            'start_time': input_window.iloc[0]['datetime'],
            'end_time': input_window.iloc[-1]['datetime']
        }

        # 添加输入特征
        for t in range(input_length):
            for feat in feature_cols:
                if feat in input_window.columns:
                    col_name = f"in_{feat}_{t}"
                    sample[col_name] = input_window.iloc[t][feat]

        # 添加输出目标
        for t in range(output_length):
            sample[f'out_lat_{t}'] = output_window.iloc[t]['latitude']
            sample[f'out_lon_{t}'] = output_window.iloc[t]['longitude']

        trajectory_samples.append(sample)

    return trajectory_samples


@timer_decorator
def create_prediction_samples(df, output_dir, input_length, output_length, feature_cols):
    """
    创建滑动窗口预测样本的优化版本

    参数:
    df: 输入数据框
    output_dir: 输出目录
    input_length: 输入序列长度
    output_length: 输出序列长度
    feature_cols: 要使用的特征列表
    """
    print(f"创建预测样本: 输入长度={input_length}, 输出长度={output_length}")

    # 确认RAPIDS兼容性
    check_rapids_compatibility()

    # 转为pandas处理，确保兼容性
    df = to_pandas(df)

    # 预分配样本列表的内存
    samples_data = []

    # 按用户和轨迹ID分组
    grouped = df.groupby(['user_id', 'segment_trajectory_id'])

    # 顺序处理轨迹组（避免多进程序列化问题）
    for (user, traj), group in tqdm(grouped, desc=f"创建样本 (输入:{input_length}, 输出:{output_length})"):
        if len(group) >= input_length + output_length:
            try:
                # 使用模块级别的函数准备样本
                trajectory_samples = prepare_prediction_sample(
                    group,
                    input_length,
                    output_length,
                    feature_cols
                )
                samples_data.extend(trajectory_samples)
            except Exception as e:
                print(f"处理轨迹 ({user}, {traj}) 时出错: {str(e)}")

    # 创建样本数据框
    if not samples_data:
        print(f"警告: 未能创建任何样本 (输入:{input_length}, 输出:{output_length})")
        return

    # 使用更高效的DataFrame创建方式
    samples_df = pd.DataFrame(samples_data)

    # 保存样本
    try:
        samples_file = os.path.join(output_dir, f"prediction_samples_in{input_length}_out{output_length}.csv")

        # 使用更高效的CSV写入方法
        samples_df.to_csv(samples_file, index=False, compression='gzip')

        print(f"创建了 {len(samples_df)} 个预测样本，已保存至: {samples_file}")
    except Exception as e:
        print(f"保存预测样本失败: {str(e)}")

        # 备份保存
        backup_file = os.path.join(output_dir, f"prediction_samples_in{input_length}_out{output_length}_backup.csv.gz")
        samples_df.to_csv(backup_file, index=False, compression='gzip')
        print(f"预测样本已保存至备份文件: {backup_file}")

    return samples_df

@timer_decorator
def main():
    # 文件路径
    input_dir = "../data/processed"
    output_dir = "../data/processed"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    USE_RAPIDS = True

    # 读取清理后的北京数据
    try:
        input_file = os.path.join(input_dir, "geolife_beijing_cleaned_data.csv")
        print(f"读取数据: {input_file}")

        try:
            if USE_RAPIDS:
                # 使用cuDF读取CSV文件
                df = cudf.read_csv(input_file,nrows=1000)
                # 打印行数
                print(f"行数: {len(df)}")
                # 确保datetime列为datetime类型
                df['datetime'] = cudf.to_datetime(df['datetime'])
            else:
                df = pd.read_csv(input_file)
                # 确保datetime列为datetime类型
                df['datetime'] = pd.to_datetime(df['datetime'])
        except Exception as e:
            print(f"使用标准方式读取数据失败，尝试更安全的读取方式: {str(e)}")
            # 使用pandas读取然后转换为cuDF
            df = pd.read_csv(input_file, parse_dates=['datetime'])
            if USE_RAPIDS:
                try:
                    df = cudf.DataFrame.from_pandas(df)
                except Exception as inner_e:
                    print(f"转换为cuDF失败: {str(inner_e)}")
                    global RAPIDS_FALLBACK
                    RAPIDS_FALLBACK = True
                    USE_RAPIDS = False
    except FileNotFoundError:
        try:
            input_file = os.path.join(input_dir, "geolife_beijing_interpolated_data.csv")
            print(f"找不到清理后的数据，尝试读取插值后的数据: {input_file}")

            if USE_RAPIDS:
                df = cudf.read_csv(input_file)
                df['datetime'] = cudf.to_datetime(df['datetime'])
            else:
                df = pd.read_csv(input_file)
                df['datetime'] = pd.to_datetime(df['datetime'])
        except FileNotFoundError:
            print("找不到必要的北京数据文件，请先运行北京数据清理脚本")
            return

    # 打印RAPIDS加速状态
    if USE_RAPIDS:
        print("RAPIDS加速已启用，将使用GPU进行计算")
        # 显示GPU内存使用情况
        if hasattr(cp, 'cuda'):
            mempool = cp.get_default_memory_pool()
            print(
                f"GPU内存使用情况: {mempool.used_bytes() / 1024 ** 2:.2f} MB / {mempool.total_bytes() / 1024 ** 2:.2f} MB")
    else:
        print("RAPIDS加速未启用，将使用CPU进行计算")

    try:
        # 1. 添加空间特征
        print("\n步骤1: 提取空间特征")
        df = extract_spatial_features(df)

        # 2. 添加时间特征
        print("\n步骤2: 提取时间特征")
        df = extract_time_features(df)

        # 3. 添加序列特征
        print("\n步骤3: 提取序列特征")
        df = extract_sequential_features(df)

        # 4. 准备预测数据集（添加目标变量）
        print("\n步骤4: 准备预测数据集")
        df = prepare_prediction_dataset(df)

        # 5. 编码分类特征
        print("\n步骤5: 编码分类特征")
        df = encode_categorical_features(df)

        # 保存带特征的数据
        enhanced_file = os.path.join(output_dir, "geolife_beijing_enhanced_features.csv")
        print(f"\n保存增强特征数据: {enhanced_file}")

        try:
            if USE_RAPIDS:
                # 转为pandas保存，确保兼容性
                to_pandas(df).to_csv(enhanced_file, index=False)
            else:
                df.to_csv(enhanced_file, index=False)
            print(f"增强特征数据已保存至: {enhanced_file}")
        except Exception as e:
            print(f"保存增强特征数据失败: {str(e)}")
            # 尝试备份保存
            backup_file = os.path.join(output_dir, "geolife_beijing_enhanced_features_backup.csv")
            to_pandas(df).to_csv(backup_file, index=False)
            print(f"增强特征数据已保存至备份文件: {backup_file}")

        # 6. 特征选择 - 为经度和纬度预测分别进行特征选择
        print("\n步骤6: 使用XGBoost进行特征选择")

        try:
            # 纬度预测的特征选择
            lat_features = feature_selection_with_xgboost(df, target_col='next_latitude', n_features=20)
            lat_features_file = os.path.join(output_dir, "latitude_prediction_features.csv")

            if USE_RAPIDS:
                to_pandas(lat_features).to_csv(lat_features_file, index=False)
            else:
                lat_features.to_csv(lat_features_file, index=False)

            print(f"纬度预测的重要特征已保存至: {lat_features_file}")

            # 经度预测的特征选择
            lon_features = feature_selection_with_xgboost(df, target_col='next_longitude', n_features=20)
            lon_features_file = os.path.join(output_dir, "longitude_prediction_features.csv")

            if USE_RAPIDS:
                to_pandas(lon_features).to_csv(lon_features_file, index=False)
            else:
                lon_features.to_csv(lon_features_file, index=False)

            print(f"经度预测的重要特征已保存至: {lon_features_file}")

            # 合并特征（取并集）
            if USE_RAPIDS:
                lat_features_list = to_pandas(lat_features)['feature'].values
                lon_features_list = to_pandas(lon_features)['feature'].values
                selected_features = pd.concat([
                    pd.Series(lat_features_list),
                    pd.Series(lon_features_list)
                ]).unique().tolist()
            else:
                selected_features = pd.concat([
                    lat_features['feature'],
                    lon_features['feature']
                ]).unique().tolist()

            print(f"合并后的特征数量: {len(selected_features)}")

            # 7. 保存选择的特征
            selected_features_df = pd.DataFrame({'feature': selected_features})
            selected_features_file = os.path.join(output_dir, "selected_features.csv")
            selected_features_df.to_csv(selected_features_file, index=False)
            print(f"选择的特征列表已保存至: {selected_features_file}")

        except Exception as e:
            print(f"特征选择过程中出错: {str(e)}")
            # 如果特征选择失败，使用一些默认特征
            print("使用默认特征列表...")
            selected_features = [
                'latitude', 'longitude', 'speed_kmh', 'distance', 'time_diff',
                'direction_angle', 'turning_angle', 'is_turn', 'hour', 'day_of_week',
                'is_weekend', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'
            ]

        # 8. 标准化特征
        print("\n步骤7: 标准化特征")
        scaled_file = os.path.join(output_dir, "geolife_beijing_scaled_features.csv")
        df_scaled = normalize_features(df, scaled_file)

        # 9. 创建只包含选定特征的数据集
        print("\n步骤8: 创建最终预测数据集")
        # 必要的列（ID、时间、坐标和目标变量）
        essential_cols = ['user_id', 'segment_trajectory_id', 'datetime',
                          'latitude', 'longitude', 'next_latitude', 'next_longitude']

        # 创建最终特征集
        final_cols = essential_cols + selected_features

        # 确保所有列都在数据框中
        final_cols = [col for col in final_cols if col in df_scaled.columns]

        # 创建最终数据集
        if USE_RAPIDS:
            df_final = df_scaled[final_cols].copy()
        else:
            df_final = df_scaled[final_cols]

        # 保存最终数据集
        final_file = os.path.join(output_dir, "geolife_beijing_prediction_dataset.csv")

        if USE_RAPIDS:
            to_pandas(df_final).to_csv(final_file, index=False)
        else:
            df_final.to_csv(final_file, index=False)

        print(f"最终预测数据集已保存至: {final_file}")

        # 10. 创建滑动窗口预测样本
        print("\n步骤9: 创建滑动窗口预测样本")

        # 为不同的输入和输出长度创建样本
        input_output_configs = [
            (10, 1),  # 前10点预测后1点
            (20, 5),  # 前20点预测后5点
            (30, 10)  # 前30点预测后10点
        ]

        for input_length, output_length in input_output_configs:
            try:
                # 创建预测样本函数
                create_prediction_samples(
                    df_final,
                    output_dir,
                    input_length,
                    output_length,
                    selected_features
                )
            except Exception as e:
                print(f"创建预测样本 ({input_length}, {output_length}) 时出错: {str(e)}")

        print("\n特征工程和数据准备完成!")
        print(f"原始特征数: {len(df.columns)}")
        print(f"选择的特征数: {len(selected_features)}")

        # 打印GPU加速使用情况总结
        if USE_RAPIDS:
            print("\nGPU加速总结:")
            print("- XGBoost特征选择使用GPU加速")
            print("- 地理计算（Haversine距离、方向角等）使用GPU加速")
            print("- 数值计算（三角函数、统计特征等）使用CuPy加速")
            print("- 部分批量处理使用Numba CUDA加速")
        else:
            print("\n提示: 安装GPU加速库可显著提高处理速度")
            print("建议安装: xgboost cupy-cuda11x numba")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()