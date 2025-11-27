"""
配置参数文件
"""

# 相关簇分析参数
DISTANCE_THRESHOLD = 7.0    # 距离阈值（米）
ANGLE_THRESHOLD = 60        # 角度阈值（度）
TIME_STEP = 1.0             # 时间步长（秒）
START_TIME = 70            # 计算起始时间（秒）
END_TIME = 900             # 计算结束时间（秒），None表示使用数据最大时间

# 实验配置
EXPERIMENT_TIME = '20251126_213448'
DATA_DIR = f'uav_data/experiment_{EXPERIMENT_TIME}/raw_data'
OUTPUT_DIR = f'uav_data/experiment_{EXPERIMENT_TIME}/process_result'

# 进度显示设置
PROGRESS_INTERVAL_DIVISOR = 5  # 进度显示间隔除数（1/N）