"""
可视化模块
"""

import matplotlib.pyplot as plt
import numpy as np
from utils import calculate_probability_distribution, get_output_path
from config import END_TIME

def plot_distribution(values, probabilities, title, xlabel, save_path=None, show_plot=False):  # 修改：默认不显示图表
    """
    绘制概率分布图
    
    Args:
        values: x轴值
        probabilities: y轴概率值
        title: 图表标题
        xlabel: x轴标签
        save_path: 保存路径，如果为None则不保存
        show_plot: 是否显示图表（默认False，后台运行）
    """
    if len(values) == 0:
        print("No data to plot!")
        return
    
    plt.figure(figsize=(6, 6))
    plt.plot(values, probabilities, 'o-', markersize=6)
    plt.xlabel(xlabel)
    plt.ylabel('Probability')
    plt.title(title)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    if show_plot:
        plt.show()  # 只有显式设置为True时才显示窗口
    else:
        plt.close()  # 不显示窗口，直接关闭图形释放内存

def plot_cluster_size_distribution(cluster_sizes, distance_thresh, angle_thresh, time_step, start_time, save_plot=True, show_plot=False):
    """
    绘制簇大小分布图
    """
    if not cluster_sizes:
        print("No cluster size data to plot!")
        return None, None
    
    size_values, size_probs = calculate_probability_distribution(cluster_sizes)
    
    # 根据是否有结束时间设置标题
    if END_TIME is None:
        title = f'Cluster Size Distribution\n(d={distance_thresh}, a={angle_thresh}, dt={time_step}s, start={start_time}s)'
    else:
        title = f'Cluster Size Distribution\n(d={distance_thresh}, a={angle_thresh}, dt={time_step}s, start={start_time}s, end={END_TIME}s)'
    
    xlabel = 'Cluster Size'
    
    save_path = None
    if save_plot:
        if END_TIME is None:
            filename = f'uav_cluster_size_dist_d{distance_thresh}_a{angle_thresh}_ts{time_step}_start{start_time}.png'
        else:
            filename = f'uav_cluster_size_dist_d{distance_thresh}_a{angle_thresh}_ts{time_step}_start{start_time}_end{END_TIME}.png'
        save_path = get_output_path(filename)
    
    plot_distribution(size_values, size_probs, title, xlabel, save_path, show_plot)
    
    return size_values, size_probs

def plot_cluster_duration_distribution(cluster_durations, distance_thresh, angle_thresh, time_step, start_time, save_plot=True, show_plot=False):
    """
    绘制簇持续时间分布图
    """
    if not cluster_durations:
        print("No cluster duration data to plot!")
        return None, None
    
    dur_values, dur_probs = calculate_probability_distribution(cluster_durations)
    
    # 根据是否有结束时间设置标题
    if END_TIME is None:
        title = f'Cluster Duration Distribution\n(d={distance_thresh}, a={angle_thresh}, dt={time_step}s, start={start_time}s)'
    else:
        title = f'Cluster Duration Distribution\n(d={distance_thresh}, a={angle_thresh}, dt={time_step}s, start={start_time}s, end={END_TIME}s)'
    
    xlabel = 'Cluster Duration (timesteps)'
    
    save_path = None
    if save_plot:
        if END_TIME is None:
            filename = f'uav_cluster_duration_dist_d{distance_thresh}_a{angle_thresh}_ts{time_step}_start{start_time}.png'
        else:
            filename = f'uav_cluster_duration_dist_d{distance_thresh}_a{angle_thresh}_ts{time_step}_start{start_time}_end{END_TIME}.png'
        save_path = get_output_path(filename)
    
    plot_distribution(dur_values, dur_probs, title, xlabel, save_path, show_plot)
    
    return dur_values, dur_probs

def plot_individual_cluster_sizes(cluster_sizes_time, time_steps, distance_thresh, angle_thresh, time_step, start_time, save_plot=True, show_plot=False):
    """
    绘制每个个体簇大小随时间变化图
    
    Args:
        cluster_sizes_time: 二维数组，shape=(时间步数, 无人机数量)
        time_steps: 时间步列表
        distance_thresh: 距离阈值
        angle_thresh: 角度阈值
        time_step: 时间步长
        start_time: 起始时间
        save_plot: 是否保存图表
        show_plot: 是否显示图表（默认False）
    """
    if cluster_sizes_time.size == 0:
        print("No individual cluster size data to plot!")
        return
    
    n_timesteps, n_uavs = cluster_sizes_time.shape
    
    plt.figure(figsize=(16, 12))
    
    # 绘制每个个体的簇大小变化
    for uav_id in range(n_uavs):
        plt.plot(time_steps, cluster_sizes_time[:, uav_id], 
                linestyle='-', linewidth=2, marker='o', markersize=1, 
                label=f'UAV {uav_id}')
    
    plt.xlabel('Time (s)')
    plt.ylabel('Cluster Size')
    
    # 根据是否有结束时间设置标题
    if END_TIME is None:
        title = f'Individual Cluster Sizes Over Time\n(d={distance_thresh}, a={angle_thresh}, dt={time_step}s, start={start_time}s)'
    else:
        title = f'Individual Cluster Sizes Over Time\n(d={distance_thresh}, a={angle_thresh}, dt={time_step}s, start={start_time}s, end={END_TIME}s)'
    
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # 由于无人机数量可能很多，图例会很大，可以选择不显示图例或简化显示
    if n_uavs <= 20:  # 只有无人机数量较少时才显示图例
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.text(0.02, 0.98, f'Total {n_uavs} UAVs', transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_plot:
        if END_TIME is None:
            filename = f'individual_cluster_sizes_d{distance_thresh}_a{angle_thresh}_ts{time_step}_start{start_time}.png'
        else:
            filename = f'individual_cluster_sizes_d{distance_thresh}_a{angle_thresh}_ts{time_step}_start{start_time}_end{END_TIME}.png'
        save_path = get_output_path(filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Individual cluster sizes plot saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()