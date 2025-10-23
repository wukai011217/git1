#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
留一结构测试模块
实现留一结构交叉验证并通过特征分布可视化解释性能差异
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn.ensemble import ExtraTreesRegressor

from wjob.config import DEFAULT_ML_MODEL

# 设置中文字体支持
def setup_chinese_font():
    """设置matplotlib支持中文字体"""
    # 检查系统中可用的字体
    font_names = [f.name for f in fm.fontManager.ttflist]
    
    # 尝试找到支持中文的字体
    chinese_fonts = ['SimHei', 'Microsoft YaHei', 'PingFang SC', 'STHeiti', 'WenQuanYi Micro Hei', 'Hiragino Sans GB']
    
    # 在系统中寻找第一个匹配的中文字体
    font_to_use = None
    for font in chinese_fonts:
        if font in font_names:
            font_to_use = font
            break
    
    # 如果找到了中文字体，则设置为默认字体
    if font_to_use:
        plt.rcParams['font.family'] = font_to_use
    else:
        # 如果没有找到中文字体，则输出警告
        print("警告: 未找到支持中文的字体，图表中的中文可能无法正确显示")
        print("系统中的字体:", font_names[:5], "...（共" + str(len(font_names)) + "个字体）")
    
    # 修复负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

# 设置绘图样式
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

def load_data(file_path):
    """
    加载数据并进行基础预处理
    
    Parameters
    ----------
    file_path : str
        数据文件路径
        
    Returns
    -------
    pd.DataFrame
        预处理后的数据框
    """
    # 加载数据
    data = pd.read_csv(file_path)
    
    # 检查缺失值
    if data.isnull().sum().sum() > 0:
        print(f"警告：数据集中存在 {data.isnull().sum().sum()} 个缺失值")
        # 填充缺失值或删除含缺失值的行
        data = data.dropna()
        print(f"已删除含缺失值的行，剩余 {data.shape[0]} 行数据")
    
    return data


def get_features_target_groups(data):
    """
    分离特征、目标变量和分组信息
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据框
        
    Returns
    -------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    groups : pd.Series
        分组信息，用于留一结构交叉验证
    """
    # 排除非特征列
    non_feature_cols = ['element', 'structure_type', 'H2_adsorption_energy']
    
    # 提取特征、目标变量和分组信息
    X = data.drop(columns=non_feature_cols)
    y = data['H2_adsorption_energy']
    groups = data['structure_type']
    
    return X, y, groups


def find_top_features(X, y, n_features=10):
    """
    训练随机森林模型并获取最重要的特征
    
    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    n_features : int, optional
        返回的重要特征数量，默认为10
        
    Returns
    -------
    list
        最重要的n个特征名称列表
    pd.Series
        所有特征的重要性得分，按重要性降序排列
    """
    # 初始化并训练随机森林模型
    if DEFAULT_ML_MODEL.lower() == 'extra_tree':
        model = ExtraTreesRegressor(max_features=0.3, min_samples_split=3, n_estimators=179,
                    n_jobs=-1, random_state=42)
    else:
        # 默认使用随机森林
        model = ExtraTreesRegressor(max_features=0.3, min_samples_split=3, n_estimators=179,
                    n_jobs=-1, random_state=42)
    
    model.fit(X, y)
    
    # 获取特征重要性并排序
    feature_importance = pd.Series(model.feature_importances_, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)
    
    # 返回最重要的n个特征名称
    top_features = feature_importance.index[:n_features].tolist()
    
    return top_features, feature_importance

def perform_loso_cv(X, y, groups, n_features=None):
    """
    执行留一结构交叉验证
    
    Parameters
    ----------
    X : pd.DataFrame
        特征矩阵
    y : pd.Series
        目标变量
    groups : pd.Series
        分组信息，用于留一结构交叉验证
    n_features : int, optional
        使用的top特征数量，如果None则使用全部特征
        
    Returns
    -------
    dict
        每个结构的性能指标
    """
    # 如果指定了特征数量，先获取top特征
    if n_features is not None:
        top_features, _ = find_top_features(X, y, n_features=n_features)
        X = X[top_features]
    
    # 初始化模型
    if DEFAULT_ML_MODEL.lower() == 'random_forest':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        # 默认使用随机森林
        model = ExtraTreesRegressor(max_features=0.3, min_samples_split=3, n_estimators=179,
                    n_jobs=-1, random_state=42)
    
    # 初始化留一结构交叉验证
    logo = LeaveOneGroupOut()
    
    # 存储每个结构的性能指标
    structure_performance = {}
    
    # 标准化处理器
    scaler = StandardScaler()
    
    # 执行留一结构交叉验证
    for train_idx, test_idx in logo.split(X, y, groups):
        # 分割数据
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # 获取当前测试结构
        test_structure = groups.iloc[test_idx].unique()[0]
        
        # 标准化特征
        X_train = pd.DataFrame(scaler.fit_transform(X_train), 
                               columns=X_train.columns, 
                               index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), 
                             columns=X_test.columns, 
                             index=X_test.index)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 计算性能指标
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 存储当前结构的性能
        structure_performance[test_structure] = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'y_true': y_test.values,
            'y_pred': y_pred
        }
    
    return structure_performance


def plot_structure_performance(structure_performance, save_dir=None):
    """
    绘制每个结构的性能对比图
    
    Parameters
    ----------
    structure_performance : dict
        留一结构交叉验证的性能指标
    save_dir : str, optional
        图像保存路径，如果为None则不保存
    """
    # 提取每个结构的RMSE和R2
    structures = list(structure_performance.keys())
    rmse_values = [structure_performance[s]['RMSE'] for s in structures]
    r2_values = [structure_performance[s]['R2'] for s in structures]
    
    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 绘制RMSE柱状图
    sns.barplot(x=structures, y=rmse_values, ax=ax1, hue=None, palette="muted")
    ax1.set_title('每个结构的RMSE')
    ax1.set_xlabel('结构')
    ax1.set_ylabel('RMSE')
    # 修复 set_ticklabels 警告
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 绘制R2柱状图
    sns.barplot(x=structures, y=r2_values, ax=ax2, hue=None, palette="muted")
    ax2.set_title('每个结构的R2')
    ax2.set_xlabel('结构')
    ax2.set_ylabel('R2')
    # 修复 set_ticklabels 警告
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图像
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, 'structure_performance.png'), dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_feature_distributions(data, top_features, structure_column='structure_type', target_column='H2_adsorption_energy', save_dir=None):
    """
    为前2个最重要特征绘制所有结构的分布图
    
    Parameters
    ----------
    data : pd.DataFrame
        输入数据框
    top_features : list
        最重要特征列表
    structure_column : str, optional
        结构列名称，默认为'structure_type'
    target_column : str, optional
        目标列名称，默认为'H2_adsorption_energy'
    save_dir : str, optional
        图像保存路径，如果为None则不保存
    """
    # 为保存图像创建目录
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # 只获取最重要的2个特征
    top_2_features = top_features[:2] if len(top_features) >= 2 else top_features
    
    # 获取唯一的结构类型
    structures = data[structure_column].unique()
    
    # 为每个重要特征绘制分布图
    for feature in top_2_features:
        # 创建图形，为每个特征创建单独的图
        plt.figure(figsize=(14, 10))
        
        # 创建小提琴图，显示所有结构下特征的分布
        ax = sns.violinplot(x=structure_column, y=feature, data=data, inner="quartile", 
                          hue=None, palette="Set3", linewidth=1.5)
        
        # 添加散点图，按吸附能着色
        sns.stripplot(x=structure_column, y=feature, data=data, 
                     hue=target_column, palette="viridis", 
                     size=8, jitter=True, alpha=0.7)
        
        # 添加标题和标签
        plt.title(f'不同结构下的 {feature} 分布', fontsize=18, fontweight='bold')
        plt.xlabel('结构类型', fontsize=16)
        plt.ylabel(feature, fontsize=16)
        # 避免使用plt.xticks来设置旋转，改为直接设置当前轴的标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=14)
        plt.yticks(fontsize=14)
        
        # 添加网格线增强可读性
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 调整图例
        plt.legend(title='H₂吸附能 (eV)', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图像
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'{feature}_distribution.png'), dpi=300, bbox_inches='tight')
        
        plt.close()
    
    # 如果有至少2个特征，创建这两个特征的联合分布图
    if len(top_2_features) == 2:
        plt.figure(figsize=(15, 12))
        
        # 创建联合分布图
        g = sns.jointplot(
            data=data,
            x=top_2_features[0],
            y=top_2_features[1],
            hue=structure_column,
            kind="scatter",
            palette="Set2",
            height=10,
            ratio=3,
            s=100,
            alpha=0.8,
            edgecolor="w",
            linewidth=0.5
        )
        
        # 添加标题
        g.fig.suptitle(f'最重要两个特征的联合分布：{top_2_features[0]} vs {top_2_features[1]}', 
                      fontsize=16, fontweight='bold', y=1.02)
        
        # 调整轴标签
        g.ax_joint.set_xlabel(top_2_features[0], fontsize=14)
        g.ax_joint.set_ylabel(top_2_features[1], fontsize=14)
        
        # 移动图例到图外
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 保存图像
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'top_2_features_joint_distribution.png'), dpi=300, bbox_inches='tight')
        
        plt.close()


def explain_performance_differences(structure_performance, top_features_data, top_feature_names, save_dir=None):
    """
    分析并解释不同结构的性能差异
    
    Parameters
    ----------
    structure_performance : dict
        每个结构的性能指标
    top_features_data : pd.DataFrame
        包含顶级特征和结构信息的数据
    top_feature_names : list
        重要特征名称列表
    save_dir : str, optional
        图像保存路径，如果为None则不保存
    """
    # 获取性能最好和最差的结构
    structures = list(structure_performance.keys())
    rmse_values = [structure_performance[s]['RMSE'] for s in structures]
    best_structure = structures[np.argmin(rmse_values)]
    worst_structure = structures[np.argmax(rmse_values)]
    
    print(f"\n性能分析与解释:")
    print(f"性能最好的结构: {best_structure}, RMSE: {structure_performance[best_structure]['RMSE']:.4f}")
    print(f"性能最差的结构: {worst_structure}, RMSE: {structure_performance[worst_structure]['RMSE']:.4f}")
    
    # 为每个特征比较最好和最差结构的分布
    plt.figure(figsize=(15, 10))
    
    for i, feature in enumerate(top_feature_names):
        plt.subplot(2, 3, i+1)
        
        # 提取最好和最差结构的特征数据
        best_data = top_features_data[top_features_data['structure_type'] == best_structure][feature]
        worst_data = top_features_data[top_features_data['structure_type'] == worst_structure][feature]
        
        # 绘制特征分布的密度图
        sns.kdeplot(best_data, label=f'{best_structure}', fill=True, alpha=0.4)
        sns.kdeplot(worst_data, label=f'{worst_structure}', fill=True, alpha=0.4)
        
        plt.title(feature)
        plt.legend()
        
        if i >= 5:  # 限制在一页最多显示6个特征
            break
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'best_worst_structure_comparison.png'), dpi=300, bbox_inches='tight')
    
    plt.show()


def main():
    """
    主函数，执行留一结构测试并分析结果
    """
    # 设置支持中文的字体
    setup_chinese_font()
    
    # 设置文件路径
    data_file = "/Users/wukai/Desktop/project/wjob/data/fea/final/reduced_dataset_90pct.csv"
    save_dir = "/Users/wukai/Desktop/project/wjob/reports/figures/loso_analysis"
    
    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    print("加载数据...")
    data = load_data(data_file)
    print(f"数据集维度: {data.shape}")
    
    # 分离特征、目标和分组
    X, y, groups = get_features_target_groups(data)
    print(f"特征数量: {X.shape[1]}")
    print(f"目标变量: {y.name}")
    print(f"唯一结构: {groups.unique().tolist()}")
    
    # 找出最重要的特征
    n_top_features = 10
    print(f"\n识别{n_top_features}个最重要特征...")
    top_features, feature_importance = find_top_features(X, y, n_features=n_top_features)
    
    print("最重要的特征 (按重要性排序):")
    for i, (feature, importance) in enumerate(feature_importance[:n_top_features].items(), 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 6))
    feature_importance[:15].plot(kind='bar')
    plt.title("特征重要性 (Top 15)")
    plt.xlabel("特征")
    plt.ylabel("重要性评分")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    
    # 执行留一结构交叉验证
    print("\n执行留一结构交叉验证...")
    structure_performance = perform_loso_cv(X, y, groups, n_features=n_top_features)
    
    # 输出各结构的性能
    print("\n各结构的性能指标:")
    all_rmse = []
    all_r2 = []
    for struct, metrics in structure_performance.items():
        rmse = metrics['RMSE']
        r2 = metrics['R2']
        all_rmse.append(rmse)
        all_r2.append(r2)
        print(f"结构 {struct}: RMSE = {rmse:.4f}, R2 = {r2:.4f}")
    
    # 输出平均性能
    print(f"\n平均性能: RMSE = {np.mean(all_rmse):.4f}, R2 = {np.mean(all_r2):.4f}")
    
    # 绘制各结构的性能对比图
    print("\n绘制性能对比图...")
    plot_structure_performance(structure_performance, save_dir=save_dir)
    
    # 为分析准备数据，只包含最重要的特征
    top_features_data = data[['structure_type', 'H2_adsorption_energy'] + top_features]
    
    # 绘制特征分布图
    print("\n绘制特征分布图...")
    plot_feature_distributions(top_features_data, top_features, save_dir=save_dir)
    
    # 解释性能差异
    print("\n分析性能差异...")
    explain_performance_differences(structure_performance, top_features_data, top_features, save_dir=save_dir)
    
    print("\n分析完成! 结果和图表已保存在:", save_dir)


if __name__ == "__main__":
    main()
