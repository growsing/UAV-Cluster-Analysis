"""
数据加载模块
"""

import pandas as pd
import os
import glob
from config import DATA_DIR, EXPERIMENT_TIME

def detect_uav_files(data_dir=None, experiment_time=None):
    """
    自动检测raw_data文件夹中的无人机文件
    
    Args:
        data_dir: 数据目录，如果为None则使用config中的设置
        experiment_time: 实验时间，如果为None则使用config中的设置
    
    Returns:
        list: 排序后的无人机ID列表
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if experiment_time is None:
        experiment_time = EXPERIMENT_TIME
    
    pattern = os.path.join(data_dir, f"uav_*_experiment_{experiment_time}.csv")
    uav_files = glob.glob(pattern)
    
    # 提取无人机ID并排序
    uav_ids = []
    for file_path in uav_files:
        filename = os.path.basename(file_path)
        # 从文件名中提取无人机ID，如 uav_0_experiment_20251125_111743.csv -> 0
        parts = filename.split('_')
        if len(parts) >= 2 and parts[0] == 'uav' and parts[1].isdigit():
            uav_ids.append(int(parts[1]))
    
    uav_ids.sort()
    return uav_ids

def read_uav_data(file_path, uav_id):
    """
    读取单个无人机的CSV数据文件
    
    Args:
        file_path: 文件路径
        uav_id: 无人机ID
    
    Returns:
        pd.DataFrame: 无人机数据，失败返回None
    """
    try:
        df = pd.read_csv(file_path)
        df['uav_id'] = uav_id
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def load_all_uav_data(data_dir=None, experiment_time=None):
    """
    加载所有无人机的数据
    
    Args:
        data_dir: 数据目录，如果为None则使用config中的设置
        experiment_time: 实验时间，如果为None则使用config中的设置
    
    Returns:
        list: 包含所有无人机数据的DataFrame列表
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if experiment_time is None:
        experiment_time = EXPERIMENT_TIME
    
    all_data = []
    
    # 自动检测无人机文件
    uav_ids = detect_uav_files(data_dir, experiment_time)
    
    if not uav_ids:
        print(f"No UAV files found in {data_dir}")
        return all_data
    
    print(f"Found {len(uav_ids)} UAV files: {uav_ids}")
    
    for uav_id in uav_ids:
        filename = f"uav_{uav_id}_experiment_{experiment_time}.csv"
        file_path = os.path.join(data_dir, filename)
        
        df = read_uav_data(file_path, uav_id)
        if df is not None:
            all_data.append(df)
            print(f"Successfully loaded data for UAV {uav_id}: {len(df)} records")
        else:
            print(f"Warning: Failed to load data for UAV {uav_id}")
    
    return all_data

def get_time_range(all_uav_data):
    """
    获取所有无人机数据的时间范围
    
    Args:
        all_uav_data: 所有无人机数据列表
    
    Returns:
        tuple: (min_time, max_time)
    """
    all_sim_times = []
    for uav_data in all_uav_data:
        if not uav_data.empty:
            all_sim_times.extend(uav_data['sim_time'].values)
    
    if not all_sim_times:
        return None, None
    
    min_time = min(all_sim_times)
    max_time = max(all_sim_times)
    return min_time, max_time