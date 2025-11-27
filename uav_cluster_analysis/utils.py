"""
工具函数模块
"""

import numpy as np
import os
from config import OUTPUT_DIR, DISTANCE_THRESHOLD, ANGLE_THRESHOLD, TIME_STEP, START_TIME, END_TIME

def angle_between(v1_x, v1_y, v2_x, v2_y):
    """
    计算两个向量的夹角（角度制）
    
    Args:
        v1_x, v1_y: 向量1的x,y分量
        v2_x, v2_y: 向量2的x,y分量
    
    Returns:
        float: 夹角角度
    """
    v1 = np.array([v1_x, v1_y])
    v2 = np.array([v2_x, v2_y])
    
    # 检查零向量
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0
    
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    
    cos_theta = dot_product / norms
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保在有效范围内
    angle_in_radians = np.arccos(cos_theta)
    angle_in_degrees = np.degrees(angle_in_radians)
    return angle_in_degrees

def calculate_distance(pos1_x, pos1_y, pos2_x, pos2_y):
    """
    计算两个无人机之间的距离
    
    Args:
        pos1_x, pos1_y: 无人机1的位置
        pos2_x, pos2_y: 无人机2的位置
    
    Returns:
        float: 距离
    """
    return np.sqrt((pos2_x - pos1_x) ** 2 + (pos2_y - pos1_y) ** 2)

def are_correlated(uav1_data, uav2_data, distance_thresh, angle_thresh):
    """
    检查两个无人机在xy平面上是否相关
    
    Args:
        uav1_data: 无人机1数据
        uav2_data: 无人机2数据
        distance_thresh: 距离阈值
        angle_thresh: 角度阈值
    
    Returns:
        bool: 是否相关
    """
    # 计算距离
    dist = calculate_distance(uav1_data['pos_x'], uav1_data['pos_y'], 
                            uav2_data['pos_x'], uav2_data['pos_y'])
    
    # 计算速度夹角
    angle = angle_between(uav1_data['vel_x'], uav1_data['vel_y'],
                         uav2_data['vel_x'], uav2_data['vel_y'])
    
    return (dist < distance_thresh) and (angle < angle_thresh)

def calculate_probability_distribution(values):
    """
    计算值的概率分布
    
    Args:
        values: 数值列表
    
    Returns:
        tuple: (unique_values, probabilities)
    """
    if len(values) == 0:
        return np.array([]), np.array([])
    
    unique_values, counts = np.unique(values, return_counts=True)
    probabilities = counts / len(values)
    return unique_values, probabilities

def create_output_dir():
    """
    创建参数特定的输出目录
    
    Returns:
        str: 参数特定的输出目录路径
    """
    # 创建参数特定的子目录
    if END_TIME is None:
        param_dir = f"d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}"
    else:
        param_dir = f"d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}_end{END_TIME}"
    
    specific_output_dir = os.path.join(OUTPUT_DIR, param_dir)
    os.makedirs(specific_output_dir, exist_ok=True)
    print(f"Created output directory: {specific_output_dir}")
    return specific_output_dir

def get_output_path(filename):
    """
    获取输出文件的完整路径（在参数特定目录下）
    
    Args:
        filename: 文件名
    
    Returns:
        str: 完整文件路径
    """
    if END_TIME is None:
        param_dir = f"d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}"
    else:
        param_dir = f"d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}_end{END_TIME}"
    
    specific_output_dir = os.path.join(OUTPUT_DIR, param_dir)
    return os.path.join(specific_output_dir, filename)