#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功率谱密度分析脚本
根据 Sun et al., 2025 的方法复现PSD分析
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy import signal
from scipy.optimize import curve_fit

def load_individual_cluster_data(filename):
    """
    加载个体簇大小数据
    
    Args:
        filename: 数据文件路径
    
    Returns:
        tuple: (time_array, cluster_sizes, uav_ids)
            time_array: 时间数组
            cluster_sizes: 簇大小数据，shape=(时间步数, 无人机数量)
            uav_ids: 无人机ID列表
    """
    # 检查文件是否存在
    if not os.path.exists(filename):
        raise FileNotFoundError(f"文件不存在: {filename}")
    
    print(f"正在加载文件: {filename}")
    
    # 读取文件头信息
    with open(filename, 'r') as f:
        header_lines = []
        for i, line in enumerate(f):
            if line.startswith('#'):
                header_lines.append(line.strip())
            else:
                break
    
    # 解析头信息
    n_timesteps = None
    n_uavs = None
    for line in header_lines:
        if 'Time steps:' in line:
            n_timesteps = int(line.split(':')[1].strip())
        elif 'Number of UAVs:' in line:
            n_uavs = int(line.split(':')[1].strip())
    
    # 读取数据
    data = np.loadtxt(filename, delimiter=',', comments='#')
    
    # 提取时间和簇大小数据
    time_array = data[:, 0]
    cluster_sizes = data[:, 1:1+n_uavs]
    
    # 生成无人机ID列表
    uav_ids = list(range(n_uavs))
    
    print(f"数据加载完成: {n_timesteps} 个时间步, {n_uavs} 架无人机")
    print(f"时间范围: {time_array[0]:.2f}s - {time_array[-1]:.2f}s")
    
    return time_array, cluster_sizes, uav_ids

def calculate_psd(signal_data, sampling_interval):
    """
    计算功率谱密度
    
    Args:
        signal_data: 信号数据
        sampling_interval: 采样间隔（秒）
    
    Returns:
        tuple: (frequencies, psd)
    """
    Fs = 1.0 / sampling_interval  # 采样频率 (Hz)
    N = len(signal_data)
    
    # 使用Welch方法计算PSD（比简单FFT更稳定）
    frequencies, psd = signal.welch(signal_data, fs=Fs, nperseg=min(256, N//4))
    
    return frequencies, psd

def power_law_func(f, A, beta):
    """
    幂律函数：P(f) = A * f^(-beta)
    """
    return A * np.power(f, -beta)

def fit_power_law(frequencies, psd, fit_range=None):
    """
    对PSD进行幂律拟合
    
    Args:
        frequencies: 频率数组
        psd: 功率谱密度数组
        fit_range: 拟合频率范围 (min_freq, max_freq)，None表示自动选择
    
    Returns:
        tuple: (A, beta, fit_psd)
            A: 幂律系数
            beta: 幂律指数
            fit_psd: 拟合的PSD值
    """
    if fit_range is None:
        # 自动选择拟合范围（忽略极低和极高频率）
        min_freq = frequencies[1]  # 忽略直流
        max_freq = frequencies[-1] * 0.5  # 使用一半的最高频率
    else:
        min_freq, max_freq = fit_range
    
    # 选择拟合范围内的数据点
    mask = (frequencies >= min_freq) & (frequencies <= max_freq)
    fit_freq = frequencies[mask]
    fit_psd_data = psd[mask]
    
    if len(fit_freq) < 2:
        print("警告: 拟合范围内的数据点不足，跳过幂律拟合")
        return None, None, None
    
    try:
        # 在对数空间中进行线性拟合
        log_f = np.log(fit_freq)
        log_psd = np.log(fit_psd_data)
        
        # 使用曲线拟合
        popt, pcov = curve_fit(power_law_func, fit_freq, fit_psd_data, 
                              p0=[psd[1], 1.0], maxfev=5000)
        A, beta = popt
        
        # 计算拟合的PSD
        fit_psd = power_law_func(frequencies, A, beta)
        
        print(f"幂律拟合成功: β = {beta:.3f}")
        return A, beta, fit_psd
    
    except Exception as e:
        print(f"幂律拟合失败: {e}")
        return None, None, None

def analyze_individual_psd(filename, uav_id, sampling_interval, output_dir=None, fit_range=None):
    """
    分析单个无人机的功率谱密度
    
    Args:
        filename: 数据文件路径
        uav_id: 要分析的无人机ID
        sampling_interval: 采样间隔（秒）
        output_dir: 输出目录
        fit_range: 拟合频率范围
    
    Returns:
        dict: 分析结果
    """
    # 加载数据
    time_array, cluster_sizes, uav_ids = load_individual_cluster_data(filename)
    
    if uav_id not in uav_ids:
        print(f"错误: 数据中找不到无人机 {uav_id}。可用的无人机ID: {uav_ids}")
        return None
    
    # 提取指定无人机的簇大小数据
    signal_data = cluster_sizes[:, uav_id]
    
    print(f"分析无人机 {uav_id} 的簇大小数据...")
    print(f"数据长度: {len(signal_data)}")
    print(f"簇大小范围: {np.min(signal_data)} - {np.max(signal_data)}")
    
    # 计算PSD
    frequencies, psd = calculate_psd(signal_data, sampling_interval)
    
    # 幂律拟合
    A, beta, fit_psd = fit_power_law(frequencies, psd, fit_range)
    
    # 绘制图形 - 修改图片大小为 (6, 6)
    plt.figure(figsize=(6, 6))
    plt.loglog(frequencies, psd, 'b.', markersize=6, alpha=0.7, label='PSD数据')
    
    if beta is not None:
        plt.loglog(frequencies, fit_psd, 'r--', linewidth=2, 
                  label=f'幂律拟合 (β = {beta:.2f})')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('频率 (Hz)')
    plt.ylabel('功率/频率 (a.u./Hz)')
    plt.title(f'功率谱密度 - 无人机 {uav_id}')
    plt.legend()
    plt.tight_layout()
    
    # 保存图形和数据 - 修改保存路径
    if output_dir is None:
        # 如果没有指定输出目录，使用数据文件所在目录
        output_dir = os.path.dirname(filename)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 从文件名提取参数信息
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # 保存图形
    plot_filename = os.path.join(output_dir, f'{base_name}_uav_{uav_id}_psd.png')
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"PSD图形已保存: {plot_filename}")
    
    # 保存PSD数据
    data_filename = os.path.join(output_dir, f'{base_name}_uav_{uav_id}_psd_data.txt')
    psd_data = np.column_stack((frequencies, psd))
    np.savetxt(data_filename, psd_data, 
              header='Frequency(Hz) Power_Spectral_Density', fmt='%.6f')
    print(f"PSD数据已保存: {data_filename}")
    
    plt.show()
    
    # 返回分析结果
    result = {
        'uav_id': uav_id,
        'frequencies': frequencies,
        'psd': psd,
        'power_law_A': A,
        'power_law_beta': beta,
        'fit_psd': fit_psd
    }
    
    return result

def analyze_all_uavs_psd(filename, sampling_interval, output_dir=None, fit_range=None):
    """
    分析所有无人机的功率谱密度
    
    Args:
        filename: 数据文件路径
        sampling_interval: 采样间隔（秒）
        output_dir: 输出目录
        fit_range: 拟合频率范围
    
    Returns:
        dict: 所有无人机的分析结果
    """
    # 加载数据
    time_array, cluster_sizes, uav_ids = load_individual_cluster_data(filename)
    
    all_results = {}
    beta_values = []
    
    # 如果没有指定输出目录，使用数据文件所在目录
    if output_dir is None:
        output_dir = os.path.dirname(filename)
    
    for uav_id in uav_ids:
        print(f"\n正在分析无人机 {uav_id} 的PSD...")
        
        # 分析单个无人机
        result = analyze_individual_psd(filename, uav_id, sampling_interval, 
                                      output_dir, fit_range)
        
        if result is not None and result['power_law_beta'] is not None:
            all_results[uav_id] = result
            beta_values.append(result['power_law_beta'])
    
    # 统计幂律指数
    if beta_values:
        print(f"\n幂律指数统计:")
        print(f"平均 β: {np.mean(beta_values):.3f}")
        print(f"标准差 β: {np.std(beta_values):.3f}")
        print(f"最小 β: {np.min(beta_values):.3f}")
        print(f"最大 β: {np.max(beta_values):.3f}")
        
        # 保存统计结果
        if output_dir:
            stats_filename = os.path.join(output_dir, 'power_law_statistics.txt')
            with open(stats_filename, 'w') as f:
                f.write("Power Law Exponent Statistics\n")
                f.write("=============================\n")
                f.write(f"Mean β: {np.mean(beta_values):.3f}\n")
                f.write(f"Std β: {np.std(beta_values):.3f}\n")
                f.write(f"Min β: {np.min(beta_values):.3f}\n")
                f.write(f"Max β: {np.max(beta_values):.3f}\n")
                f.write(f"Number of UAVs: {len(beta_values)}\n")
            print(f"统计结果已保存: {stats_filename}")
    
    return all_results

def main():
    """
    主函数 - 用户可修改参数
    """
    # --- 用户需要修改的参数 ---
    # 使用完整的文件路径
    filename = r'uav_data\experiment_20251126_213448\process_result\d6.0_a60_ts1.0_start400\individual_cluster_sizes_d6.0_a60_ts1.0_start400.txt'
    sampling_interval = 1.0  # 采样间隔（秒）
    uav_id_to_analyze = 0    # 要分析的无人机ID（设为None则分析所有）
    fit_range = (0.001, 0.1) # 幂律拟合频率范围 (Hz)，None表示自动选择
    output_dir = None  # 设为None，自动使用数据文件所在目录
    # -------------------------
    
    print("=== 无人机集群功率谱密度分析 ===")
    print(f"数据文件: {filename}")
    print(f"采样间隔: {sampling_interval} 秒")
    
    # 确保使用绝对路径
    if not os.path.isabs(filename):
        # 如果使用相对路径，转换为绝对路径
        filename = os.path.abspath(filename)
        print(f"使用绝对路径: {filename}")
    
    # 设置输出目录（如果不指定，使用数据文件所在目录）
    if output_dir is None:
        output_dir = os.path.dirname(filename)
        print(f"输出目录: {output_dir}")
    
    try:
        if uav_id_to_analyze is not None:
            # 分析单个无人机
            print(f"分析单个无人机: UAV {uav_id_to_analyze}")
            result = analyze_individual_psd(filename, uav_id_to_analyze, 
                                          sampling_interval, output_dir, fit_range)
        else:
            # 分析所有无人机
            print("分析所有无人机...")
            all_results = analyze_all_uavs_psd(filename, sampling_interval, 
                                             output_dir, fit_range)
        
        print("\n分析完成!")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
        print("请检查文件路径是否正确。")
        print("当前工作目录:", os.getcwd())
        
        # 列出当前目录下的文件
        print("\n当前目录下的文件:")
        for file in os.listdir('.'):
            if file.endswith('.txt'):
                print(f"  {file}")
        
        # 如果文件在子目录中，尝试查找
        print("\n搜索相关文件...")
        for root, dirs, files in os.walk('.'):
            for file in files:
                if 'individual_cluster_sizes' in file and file.endswith('.txt'):
                    print(f"找到相关文件: {os.path.join(root, file)}")

if __name__ == "__main__":
    main()