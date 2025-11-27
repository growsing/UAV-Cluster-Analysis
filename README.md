# UAV Cluster Analysis

无人机集群相关簇分析工具，用于分析无人机集群在时空中的群体行为模式。

## 功能特性

- 自动检测并加载多无人机实验数据
- 基于距离和速度夹角的相关簇识别
- 分析相关簇大小和持续时间的概率分布
- 生成可视化图表和统计数据

## 快速开始

0. 导入数据：将无人机集群实验数据放在 `uav_data/` 目录下。格式如示例数据 `uav_1_experiment_20251125_111743.csv`  
1. 配置参数：修改 `config.py` 中的实验设置
2. 运行分析：`python main.py`
3. 查看结果：结果保存在 `process_result` 目录

## 数据要求

数据文件格式：`uav_{id}_experiment_{time}.csv`
必需字段：`timestamp`, `sim_time`, `pos_x`, `pos_y`, `vel_x`, `vel_y`