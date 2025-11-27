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