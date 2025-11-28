"""
主程序入口
"""

import numpy as np
import os
from config import *
from data_loader import load_all_uav_data, get_time_range
from cluster_analyzer import analyze_cluster_sizes, analyze_cluster_duration, analyze_individual_cluster_sizes
from visualization import plot_cluster_size_distribution, plot_cluster_duration_distribution, plot_individual_cluster_sizes
from utils import create_output_dir, get_output_path, calculate_probability_distribution

def main():
    """
    主函数
    """
    print("=== UAV Cluster Analysis ===")
    print(f"Experiment: {EXPERIMENT_TIME}")
    if END_TIME is None:
        print(f"Parameters: d={DISTANCE_THRESHOLD}, a={ANGLE_THRESHOLD}, dt={TIME_STEP}, start={START_TIME}")
    else:
        print(f"Parameters: d={DISTANCE_THRESHOLD}, a={ANGLE_THRESHOLD}, dt={TIME_STEP}, start={START_TIME}, end={END_TIME}")
    
    # 创建参数特定的输出目录
    output_dir = create_output_dir()
    print(f"Output directory: {output_dir}")
    
    # 加载数据
    print("\nLoading UAV data...")
    all_uav_data = load_all_uav_data()
    
    # 检查是否有有效数据
    if not all_uav_data or all(df.empty for df in all_uav_data):
        print("No valid data available for analysis!")
        return
    
    # 统计信息
    valid_uavs = sum(1 for df in all_uav_data if not df.empty)
    total_data_points = sum(len(df) for df in all_uav_data if not df.empty)
    
    print(f"\nData Summary:")
    print(f"Loaded data from {valid_uavs} UAVs")
    print(f"Total data points: {total_data_points}")
    
    # 确定时间范围
    min_time, max_time = get_time_range(all_uav_data)
    if min_time is None or max_time is None:
        print("Error: Could not determine time range from data!")
        return
    
    print(f"Original time range: {min_time:.2f}s - {max_time:.2f}s")
    
    # 检查起始时间和结束时间是否合理
    if START_TIME < min_time:
        print(f"Warning: Start time {START_TIME}s is before data start time {min_time:.2f}s")
        print(f"Using actual start time: {min_time:.2f}s")
    elif START_TIME > max_time:
        print(f"Error: Start time {START_TIME}s is after data end time {max_time:.2f}s")
        print("No data to analyze!")
        return
    
    if END_TIME is not None:
        if END_TIME < min_time:
            print(f"Error: End time {END_TIME}s is before data start time {min_time:.2f}s")
            print("No data to analyze!")
            return
        elif END_TIME > max_time:
            print(f"Warning: End time {END_TIME}s is after data end time {max_time:.2f}s")
            print(f"Using actual end time: {max_time:.2f}s")
    
    # 分析每个个体簇大小随时间变化
    print("\n" + "="*50)
    print("Starting individual cluster size analysis...")
    cluster_sizes_time, time_steps_used = analyze_individual_cluster_sizes(
        all_uav_data, DISTANCE_THRESHOLD, ANGLE_THRESHOLD,
        time_step=TIME_STEP, start_time=START_TIME, end_time=END_TIME
    )
    
    if cluster_sizes_time.size > 0:
        # 绘制个体簇大小变化图
        plot_individual_cluster_sizes(
            cluster_sizes_time, time_steps_used, 
            DISTANCE_THRESHOLD, ANGLE_THRESHOLD, TIME_STEP, START_TIME,
            show_plot=False
        )
        
        # 保存个体簇大小数据（使用新格式）
        if END_TIME is None:
            output_file = f'individual_cluster_sizes_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}.txt'
        else:
            output_file = f'individual_cluster_sizes_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}_end{END_TIME}.txt'
        
        from cluster_analyzer import save_individual_cluster_sizes
        save_individual_cluster_sizes(cluster_sizes_time, time_steps_used, get_output_path(output_file))
        
        print(f"Individual cluster size analysis completed. Data shape: {cluster_sizes_time.shape}")
        print(f"Time range: {time_steps_used[0]:.2f}s - {time_steps_used[-1]:.2f}s")

    # 分析簇大小分布
    print("\n" + "="*50)
    print("Starting cluster size analysis...")
    cluster_sizes = analyze_cluster_sizes(all_uav_data, DISTANCE_THRESHOLD, ANGLE_THRESHOLD, 
                                         time_step=TIME_STEP, start_time=START_TIME, end_time=END_TIME)
    
    if cluster_sizes:
        # 绘制和保存簇大小分布（不显示窗口）
        size_values, size_probs = plot_cluster_size_distribution(
            cluster_sizes, DISTANCE_THRESHOLD, ANGLE_THRESHOLD, TIME_STEP, START_TIME,
            show_plot=False  # 不弹出显示窗口
        )
        
        # 保存数据
        if END_TIME is None:
            np.savetxt(get_output_path(f'uav_cluster_sizes_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}.txt'), 
                       cluster_sizes, fmt='%d')
            np.savetxt(get_output_path(f'uav_cluster_size_values_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}.txt'), 
                       size_values, fmt='%d')
            np.savetxt(get_output_path(f'uav_cluster_size_probs_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}.txt'), 
                       size_probs, fmt='%f')
        else:
            np.savetxt(get_output_path(f'uav_cluster_sizes_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}_end{END_TIME}.txt'), 
                       cluster_sizes, fmt='%d')
            np.savetxt(get_output_path(f'uav_cluster_size_values_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}_end{END_TIME}.txt'), 
                       size_values, fmt='%d')
            np.savetxt(get_output_path(f'uav_cluster_size_probs_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}_end{END_TIME}.txt'), 
                       size_probs, fmt='%f')
        
        print(f"Cluster size analysis completed. Found {len(cluster_sizes)} clusters.")
        print(f"Size range: {min(cluster_sizes)} - {max(cluster_sizes)}")
        print(f"Average cluster size: {np.mean(cluster_sizes):.2f}")
    else:
        print("No clusters found in size analysis!")
    
    # 分析簇持续时间分布
    print("\n" + "="*50)
    print("Starting cluster duration analysis...")
    cluster_durations = analyze_cluster_duration(all_uav_data, DISTANCE_THRESHOLD, ANGLE_THRESHOLD,
                                                time_step=TIME_STEP, start_time=START_TIME, end_time=END_TIME)
    
    if cluster_durations:
        # 绘制和保存簇持续时间分布（不显示窗口）
        dur_values, dur_probs = plot_cluster_duration_distribution(
            cluster_durations, DISTANCE_THRESHOLD, ANGLE_THRESHOLD, TIME_STEP, START_TIME,
            show_plot=False  # 不弹出显示窗口
        )
        
        # 保存数据
        if END_TIME is None:
            np.savetxt(get_output_path(f'uav_cluster_durations_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}.txt'), 
                       cluster_durations, fmt='%d')
            np.savetxt(get_output_path(f'uav_cluster_dur_values_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}.txt'), 
                       dur_values, fmt='%d')
            np.savetxt(get_output_path(f'uav_cluster_dur_probs_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}.txt'), 
                       dur_probs, fmt='%f')
        else:
            np.savetxt(get_output_path(f'uav_cluster_durations_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}_end{END_TIME}.txt'), 
                       cluster_durations, fmt='%d')
            np.savetxt(get_output_path(f'uav_cluster_dur_values_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}_end{END_TIME}.txt'), 
                       dur_values, fmt='%d')
            np.savetxt(get_output_path(f'uav_cluster_dur_probs_d{DISTANCE_THRESHOLD}_a{ANGLE_THRESHOLD}_ts{TIME_STEP}_start{START_TIME}_end{END_TIME}.txt'), 
                       dur_probs, fmt='%f')
        
        print(f"Cluster duration analysis completed. Found {len(cluster_durations)} cluster instances.")
        print(f"Duration range: {min(cluster_durations)} - {max(cluster_durations)} timesteps")
        print(f"Average cluster duration: {np.mean(cluster_durations):.2f} timesteps")
    else:
        print("No cluster durations found!")
    
    # 输出最终统计信息
    print("\n" + "="*50)
    print("Final Statistical Summary:")
    print(f"Total UAVs analyzed: {valid_uavs}")
    
    # 计算实际分析的时间范围
    actual_start_time = max(min_time, START_TIME)
    if END_TIME is None:
        actual_end_time = max_time
    else:
        actual_end_time = min(max_time, END_TIME)
    
    print(f"Total timesteps analyzed: {int((actual_end_time - actual_start_time) / TIME_STEP)}")
    print(f"Time step size: {TIME_STEP}s")
    print(f"Start time: {START_TIME}s (actual: {actual_start_time:.2f}s)")
    if END_TIME is not None:
        print(f"End time: {END_TIME}s (actual: {actual_end_time:.2f}s)")
    print(f"Distance threshold: {DISTANCE_THRESHOLD}m")
    print(f"Angle threshold: {ANGLE_THRESHOLD}°")
    
    if cluster_sizes:
        print(f"Total clusters found: {len(cluster_sizes)}")
        print(f"Cluster size statistics: min={min(cluster_sizes)}, max={max(cluster_sizes)}, mean={np.mean(cluster_sizes):.2f}")
    
    if cluster_durations:
        print(f"Cluster duration statistics: min={min(cluster_durations)}, max={max(cluster_durations)}, mean={np.mean(cluster_durations):.2f}")
    
    print(f"\nAll results saved to: {output_dir}")

if __name__ == "__main__":
    main()