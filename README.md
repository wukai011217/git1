# WJob - 催化剂研究辅助工具集

![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-orange)

一个用于催化剂研究的综合工具集，提供从数据收集、处理到分析的完整工作流程。

## 📋 目录

- [功能特性](#-功能特性)
- [安装说明](#-安装说明)
- [快速开始](#️-快速开始)
- [项目结构](#-项目结构)
- [主要模块](#-主要模块)
- [使用示例](#-使用示例)
- [依赖项](#-依赖项)
- [贡献指南](#-贡献指南)
- [许可证](#-许可证)

## 🚀 功能特性

- **结构分析**: 催化剂结构的几何分析和角度计算
- **机器学习**: 集成多种ML算法用于催化剂性能预测
- **数据可视化**: 丰富的图表和可视化工具
- **不确定性估计**: 模型预测的不确定性量化
- **VASP集成**: 支持VASP计算的自动化工作流程
- **特征工程**: 催化剂描述符的自动生成和选择

## 📦 安装说明

### 环境要求

- Python 3.7+
- NumPy, Pandas, Scikit-learn
- ASE (Atomic Simulation Environment)

### 安装方法

1. **克隆仓库**

```bash
git clone https://github.com/wukai011217/git1.git
cd git1
```

1. **创建虚拟环境（推荐）**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

1. **安装依赖**

```bash
pip install -r requirements.txt
```

1. **安装包**

```bash
pip install -e .
```

## 🏃‍♂️ 快速开始

```python
import wjob

# 基本使用示例
from wjob.analysis import StructureAnalyzer
from wjob.ml import CatalystPredictor

# 结构分析
analyzer = StructureAnalyzer()
results = analyzer.analyze_structure("path/to/structure.cif")

# 机器学习预测
predictor = CatalystPredictor()
prediction = predictor.predict(features)
```

## 📁 项目结构

```text
w_git/
├── src/
│   ├── wjob/                    # 主要包目录
│   │   ├── __init__.py
│   │   ├── config.py            # 全局配置
│   │   ├── analysis/            # 分析模块
│   │   ├── data/                # 数据处理模块
│   │   ├── ml/                  # 机器学习模块
│   │   ├── utils/               # 工具函数
│   │   └── visualization/       # 可视化模块
│   └── features/                # 特征工程脚本
├── notebooks/                   # Jupyter笔记本
├── data/                        # 数据文件
├── requirements.txt             # 依赖列表
├── setup.py                     # 安装配置
└── README.md                    # 项目说明
```

## 🔧 主要模块

### 分析模块 (analysis/)

- 结构几何分析
- 角度和距离计算
- 结构对比和聚类

### 机器学习模块 (ml/)

- 多种ML算法集成
- 特征重要性分析
- 模型解释和可视化
- 不确定性量化

### 数据处理模块 (data/)

- 数据清洗和预处理
- 格式转换
- 数据验证

### 可视化模块 (visualization/)

- 结构可视化
- 数据图表
- 分析结果展示

## 💡 使用示例

### 结构分析示例

```python
from wjob.analysis import AngleAnalyzer

# 分析O-Ce-O角度
analyzer = AngleAnalyzer()
angles = analyzer.analyze_o_ce_o_angles("structure.cif")
print(f"平均角度: {angles.mean():.2f}°")
```

### 机器学习示例

```python
from wjob.ml import UncertaintyEstimator
import pandas as pd

# 加载数据
data = pd.read_csv("catalyst_data.csv")

# 不确定性估计
estimator = UncertaintyEstimator()
predictions, uncertainties = estimator.predict_with_uncertainty(data)
```

### 可视化示例

```python
from wjob.visualization import LearningCurveAnalyzer

# 学习曲线分析
analyzer = LearningCurveAnalyzer()
analyzer.plot_learning_curves(X, y, models=['rf', 'xgb', 'svm'])
```

## 📋 依赖项

### 核心依赖

- `numpy>=1.20.0` - 数值计算
- `pandas>=1.3.0` - 数据处理
- `scikit-learn>=1.0.0` - 机器学习
- `ase>=3.22.0` - 原子模拟环境

### 可视化

- `matplotlib>=3.4.0` - 基础绘图
- `seaborn>=0.11.0` - 统计可视化

### 机器学习增强

- `xgboost>=1.5.0` - 梯度提升
- `shap>=0.40.0` - 模型解释
- `dscribe>=1.2.0` - 材料描述符

### 开发工具

- `jupyter>=1.0.0` - 交互式开发
- `pytest>=6.0.0` - 测试框架
- `sphinx>=4.0.0` - 文档生成

## 📚 文档和教程

详细的使用教程和API文档请参考 `notebooks/` 目录中的Jupyter笔记本：

- `jupy.ipynb` - 基础使用教程
- `plot.ipynb` - 可视化示例
- 其他专题教程...

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

## 🐛 问题反馈

如果您遇到任何问题或有功能建议，请在 [Issues](../../issues) 页面提交。

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢所有为本项目做出贡献的开发者和研究人员。

---


