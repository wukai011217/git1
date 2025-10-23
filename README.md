# WJob - 催化剂研究辅助工具

## 项目概述

WJob是一个用于催化剂研究的综合工具集，它提供了从数据收集、处理到分析的完整工作流程。该项目旨在辅助化学领域催化剂相关研究，实现从服务器(SLURM)计算DFT数据，数据到特征的转换，以及机器学习的训练与解释。

## 特点

- SLURM服务器资源管理与监控
- DFT数据自动化计算与处理
- 催化剂特征提取与分析
- 机器学习模型构建与解释
- 可视化分析与结果展示

## 安装方法

```bash
git clone https://github.com/yourusername/wjob.git
cd wjob
pip install -e .
```

## 使用示例

### 查找空闲计算资源

```bash
python -m wjob.slurm.check_idle_cores --threshold 4
```

### 运行DFT计算

```bash
python -m wjob.dft.run_calculation --input-file path/to/input.xyz
```

### 训练机器学习模型

```bash
python -m wjob.ml.train --data-path path/to/features.csv --model xgboost
```

## 项目结构

```
wjob/
├── src/wjob/          # 源代码
│   ├── slurm/         # SLURM相关功能
│   ├── data/          # 数据处理功能
│   ├── models/        # 模型定义
│   └── visualization/ # 可视化功能
├── dft/               # DFT计算相关
├── ml/                # 机器学习相关
├── catalysts/         # 催化剂特定代码
├── notebooks/         # Jupyter笔记本
├── data/              # 数据文件
└── results/           # 结果输出
```

## 许可证

MIT

## 引用

如果您在研究中使用了此工具，请引用：

```
待添加
```

## 贡献

欢迎提交问题和拉取请求。
