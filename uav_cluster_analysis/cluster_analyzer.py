"""
相关簇分析模块
"""

import numpy as np
from utils import are_correlated, calculate_probability_distribution
from config import TIME_STEP, START_TIME, END_TIME, PROGRESS_INTERVAL_DIVISOR

def find_correlated_clusters(uavs_data, distance_thresh, angle_thresh):
    """
    在单个时间步长中查找相关簇
    
    Args:
        uavs_data: 无人机数据列表
        distance_thresh: 距离阈值
        angle_thresh: 角度阈值
    
    Returns:
        list: 相关簇列表，每个簇包含无人机索引
    """
    n_uavs = len(uavs_data)
    visited = [False] * n_uavs
    clusters = []
    
    def dfs(uav_idx, cluster):
        visited[uav_idx] = True
        cluster.append(uav_idx)
        
        # 查找未访问的相关邻居
        for other_idx in range(n_uavs):
            if not visited[other_idx]:
                if are_correlated(uavs_data[uav_idx], uavs_data[other_idx], 
                                distance_thresh, angle_thresh):
                    dfs(other_idx, cluster)
        
        return cluster
    
    for i in range(n_uavs):
        if not visited[i]:
            cluster = dfs(i, [])
            clusters.append(cluster)
    
    return clusters

def find_closest_data_point(uav_data, target_time, time_step):
    """
    在无人机数据中找到最接近目标时间的数据点
    
    Args:
        uav_data: 无人机数据
        target_time: 目标时间
        time_step: 时间步长
    
    Returns:
        pd.Series: 最接近的数据点，如果没有则返回None
    """
    if uav_data.empty:
        return None
    
    # 计算时间差
    time_diffs = np.abs(uav_data['sim_time'] - target_time)
    min_idx = time_diffs.idxmin()
    
    # 如果时间差太大，可能没有有效数据
    if time_diffs[min_idx] > time_step / 2:  # 允许半个时间步长的误差
        return None
    
    return uav_data.iloc[min_idx]

def get_unified_timestep_data(all_uav_data, target_time, time_step):
    """
    获取所有无人机在指定时间步长的数据
    
    Args:
        all_uav_data: 所有无人机数据
        target_time: 目标时间
        time_step: 时间步长
    
    Returns:
        list: 统一时间步长的无人机数据
    """
    unified_data = []
    
    for uav_id, uav_df in enumerate(all_uav_data):
        closest_data = find_closest_data_point(uav_df, target_time, time_step)
        if closest_data is not None:
            unified_data.append(closest_data)
    
    return unified_data

def analyze_cluster_sizes(all_uav_data, distance_thresh, angle_thresh, time_step=None, start_time=None, end_time=None):
    """
    分析所有时间步的相关簇大小分布（统一时间步长）
    
    Args:
        all_uav_data: 所有无人机数据
        distance_thresh: 距离阈值
        angle_thresh: 角度阈值
        time_step: 时间步长，如果为None则使用config中的设置
        start_time: 起始时间，如果为None则使用config中的设置
        end_time: 结束时间，如果为None则使用config中的设置
    
    Returns:
        list: 簇大小列表
    """
    if time_step is None:
        time_step = TIME_STEP
    if start_time is None:
        start_time = START_TIME
    if end_time is None:
        end_time = END_TIME
    
    avalanche_sizes = []
    
    # 确定时间范围
    from data_loader import get_time_range
    min_time, max_time = get_time_range(all_uav_data)
    
    if min_time is None:
        return avalanche_sizes
    
    # 调整起始时间，确保不小于最小时间
    actual_start_time = max(min_time, start_time)
    
    # 调整结束时间，确保不大于最大时间
    if end_time is None:
        actual_end_time = max_time
    else:
        actual_end_time = min(max_time, end_time)
    
    # 检查时间范围是否有效
    if actual_start_time >= actual_end_time:
        print(f"Error: Invalid time range. Start: {actual_start_time:.2f}s, End: {actual_end_time:.2f}s")
        return avalanche_sizes
    
    # 生成统一的时间步长序列（从start_time到end_time）
    time_steps = np.arange(actual_start_time, actual_end_time, time_step)
    
    print(f"Analyzing {len(time_steps)} timesteps (from {actual_start_time:.2f}s to {actual_end_time:.2f}s)...")
    print(f"Original time range: {min_time:.2f}s - {max_time:.2f}s")
    print(f"Start time set to: {start_time}s, actual start: {actual_start_time}s")
    if end_time is not None:
        print(f"End time set to: {end_time}s, actual end: {actual_end_time}s")
    
    # 计算进度显示间隔
    progress_interval = max(1, len(time_steps) // PROGRESS_INTERVAL_DIVISOR)
    
    for i, target_time in enumerate(time_steps):
        # 进度显示
        if i % progress_interval == 0 or i == len(time_steps) - 1:
            print(f"Processing timestep {i}/{len(time_steps)} (time = {target_time:.2f}s)")
        
        # 获取当前时间步的所有无人机数据
        current_data = get_unified_timestep_data(all_uav_data, target_time, time_step)
        
        if len(current_data) < 2:  # 至少需要2架无人机才能形成簇
            continue
        
        # 查找相关簇
        clusters = find_correlated_clusters(current_data, distance_thresh, angle_thresh)
        
        # 记录簇大小（排除大小为1的簇）
        for cluster in clusters:
            if len(cluster) > 1:  # 只考虑大小大于1的簇
                avalanche_sizes.append(len(cluster))
    
    return avalanche_sizes

def analyze_cluster_duration(all_uav_data, distance_thresh, angle_thresh, time_step=None, start_time=None, end_time=None):
    """
    分析相关簇的持续时间分布（统一时间步长）
    
    Args:
        all_uav_data: 所有无人机数据
        distance_thresh: 距离阈值
        angle_thresh: 角度阈值
        time_step: 时间步长，如果为None则使用config中的设置
        start_time: 起始时间，如果为None则使用config中的设置
        end_time: 结束时间，如果为None则使用config中的设置
    
    Returns:
        list: 簇持续时间列表
    """
    if time_step is None:
        time_step = TIME_STEP
    if start_time is None:
        start_time = START_TIME
    if end_time is None:
        end_time = END_TIME
    
    # 确定时间范围
    from data_loader import get_time_range
    min_time, max_time = get_time_range(all_uav_data)
    
    if min_time is None:
        return []
    
    # 调整起始时间，确保不小于最小时间
    actual_start_time = max(min_time, start_time)
    
    # 调整结束时间，确保不大于最大时间
    if end_time is None:
        actual_end_time = max_time
    else:
        actual_end_time = min(max_time, end_time)
    
    # 检查时间范围是否有效
    if actual_start_time >= actual_end_time:
        print(f"Error: Invalid time range. Start: {actual_start_time:.2f}s, End: {actual_end_time:.2f}s")
        return []
    
    # 生成统一的时间步长序列（从start_time到end_time）
    time_steps = np.arange(actual_start_time, actual_end_time, time_step)
    
    cluster_evolution = {}  # 跟踪簇的演变
    cluster_history = []    # 记录每个时间步的簇
    
    print(f"Tracking cluster evolution over {len(time_steps)} timesteps...")
    print(f"Time range: {actual_start_time:.2f}s - {actual_end_time:.2f}s")
    if end_time is not None:
        print(f"End time set to: {end_time}s, actual end: {actual_end_time}s")
    
    # 计算进度显示间隔
    progress_interval = max(1, len(time_steps) // PROGRESS_INTERVAL_DIVISOR)
    
    for i, target_time in enumerate(time_steps):
        # 进度显示
        if i % progress_interval == 0 or i == len(time_steps) - 1:
            print(f"Processing timestep {i}/{len(time_steps)} (time = {target_time:.2f}s)")
        
        current_data = get_unified_timestep_data(all_uav_data, target_time, time_step)
        
        if len(current_data) < 2:
            cluster_history.append([])
            continue
        
        # 查找当前时间步的簇
        current_clusters = find_correlated_clusters(current_data, distance_thresh, angle_thresh)
        current_cluster_ids = [tuple(sorted([current_data[idx]['uav_id'] for idx in cluster])) 
                             for cluster in current_clusters if len(cluster) > 1]
        
        cluster_history.append(current_cluster_ids)
        
        # 更新簇跟踪
        for cluster_id in current_cluster_ids:
            if cluster_id in cluster_evolution:
                if cluster_evolution[cluster_id]['active']:
                    cluster_evolution[cluster_id]['duration'] += 1
                    cluster_evolution[cluster_id]['last_seen'] = target_time
                else:
                    # 簇重新出现，开始新的持续时间记录
                    cluster_evolution[cluster_id] = {
                        'start': target_time,
                        'last_seen': target_time,
                        'duration': 1,
                        'active': True
                    }
            else:
                cluster_evolution[cluster_id] = {
                    'start': target_time,
                    'last_seen': target_time,
                    'duration': 1,
                    'active': True
                }
        
        # 标记在当前时间步未出现的簇为不活跃
        for cluster_id in cluster_evolution:
            if cluster_evolution[cluster_id]['active'] and cluster_id not in current_cluster_ids:
                cluster_evolution[cluster_id]['active'] = False
    
    # 提取持续时间
    durations = [info['duration'] for info in cluster_evolution.values()]
    return durations