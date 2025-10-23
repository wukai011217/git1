#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PCA降维可视化脚本
用于读取CSV文件，进行PCA降维并绘制可视化图形
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib.colors import ListedColormap
import argparse
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 使用支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def load_data(csv_file, target_column=None, label_column=None):
    """
    加载CSV数据文件
    
    参数:
        csv_file (str): CSV文件路径
        target_column (str, optional): 目标列名（如能量值）
        label_column (str, optional): 标签列名（如结构类型）
        
    返回:
        tuple: (特征数据, 目标值, 标签)
    """
    print(f"正在加载数据: {csv_file}")
    df = pd.read_csv(csv_file)
    
    # 如果指定了目标列，则分离目标值
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column].values
    
    # 如果指定了标签列，则提取标签
    labels = None
    if label_column and label_column in df.columns:
        labels = df[label_column].values
        # 从特征中移除标签列
        df = df.drop(columns=[label_column])
    
    # 如果指定了目标列，从特征中移除目标列
    if target_column and target_column in df.columns:
        df = df.drop(columns=[target_column])
    
    # 移除非数值列
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    X = df[numeric_cols].values
    
    return X, y, labels

def perform_pca(X, n_components=2, scale=True):
    """
    执行PCA降维
    
    参数:
        X (numpy.ndarray): 特征数据
        n_components (int): 降维后的维度数
        scale (bool): 是否进行标准化
        
    返回:
        tuple: (降维后的数据, PCA模型, 解释方差比例)
    """
    print(f"执行PCA降维，目标维度: {n_components}")
    
    # 标准化数据
    if scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = X
    
    # 执行PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 计算解释方差比例
    explained_variance_ratio = pca.explained_variance_ratio_
    total_explained_variance = np.sum(explained_variance_ratio)
    
    print(f"PCA解释方差比例: {explained_variance_ratio}")
    print(f"总解释方差: {total_explained_variance:.4f}")
    
    return X_pca, pca, explained_variance_ratio

def plot_pca_2d(X_pca, labels=None, target=None, title="PCA降维可视化", 
                save_path=None, figsize=(12, 10), dpi=300, 
                colormap='viridis', alpha=0.7, s=80):
    """
    绘制2D PCA降维图
    
    参数:
        X_pca (numpy.ndarray): 降维后的数据
        labels (numpy.ndarray, optional): 数据标签
        target (numpy.ndarray, optional): 目标值（用于颜色映射）
        title (str): 图表标题
        save_path (str, optional): 保存路径
        figsize (tuple): 图表大小
        dpi (int): 图表DPI
        colormap (str): 颜色映射
        alpha (float): 点的透明度
        s (int): 点的大小
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 如果有目标值，使用目标值作为颜色映射
    if target is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=target, cmap=colormap, 
                             alpha=alpha, s=s, edgecolors='k', linewidths=0.5)
        cbar = plt.colorbar(scatter)
        cbar.set_label('目标值', fontsize=14)
    
    # 如果有标签，使用标签作为颜色分类
    elif labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], color=colors[i], 
                       label=label, alpha=alpha, s=s, edgecolors='k', linewidths=0.5)
        
        ax.legend(title="类别", fontsize=12, title_fontsize=14)
    
    # 如果没有标签和目标值，使用默认颜色
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=alpha, s=s, 
                   edgecolors='k', linewidths=0.5)
    
    # 设置图表样式
    ax.set_xlabel(f'主成分 1', fontsize=16)
    ax.set_ylabel(f'主成分 2', fontsize=16)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 添加原点参考线
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()
    return fig

def plot_pca_3d(X_pca, labels=None, target=None, title="PCA 3D降维可视化", 
                save_path=None, figsize=(14, 12), dpi=300, 
                colormap='viridis', alpha=0.7, s=80):
    """
    绘制3D PCA降维图
    
    参数:
        X_pca (numpy.ndarray): 降维后的数据（至少3维）
        labels (numpy.ndarray, optional): 数据标签
        target (numpy.ndarray, optional): 目标值（用于颜色映射）
        title (str): 图表标题
        save_path (str, optional): 保存路径
        figsize (tuple): 图表大小
        dpi (int): 图表DPI
        colormap (str): 颜色映射
        alpha (float): 点的透明度
        s (int): 点的大小
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 如果有目标值，使用目标值作为颜色映射
    if target is not None:
        scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], 
                             c=target, cmap=colormap, alpha=alpha, s=s)
        cbar = plt.colorbar(scatter)
        cbar.set_label('目标值', fontsize=14)
    
    # 如果有标签，使用标签作为颜色分类
    elif labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], X_pca[mask, 2], 
                       color=colors[i], label=label, alpha=alpha, s=s)
        
        ax.legend(title="类别", fontsize=12, title_fontsize=14)
    
    # 如果没有标签和目标值，使用默认颜色
    else:
        ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], alpha=alpha, s=s)
    
    # 设置图表样式
    ax.set_xlabel(f'主成分 1', fontsize=16)
    ax.set_ylabel(f'主成分 2', fontsize=16)
    ax.set_zlabel(f'主成分 3', fontsize=16)
    ax.set_title(title, fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # 设置视角
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()
    return fig

def plot_explained_variance(explained_variance_ratio, title="PCA解释方差比例", 
                           save_path=None, figsize=(10, 6), dpi=300):
    """
    绘制PCA解释方差比例图
    
    参数:
        explained_variance_ratio (numpy.ndarray): 解释方差比例
        title (str): 图表标题
        save_path (str, optional): 保存路径
        figsize (tuple): 图表大小
        dpi (int): 图表DPI
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 计算累积解释方差
    cumulative_variance = np.cumsum(explained_variance_ratio)
    components = np.arange(1, len(explained_variance_ratio) + 1)
    
    # 绘制条形图
    ax.bar(components, explained_variance_ratio, alpha=0.7, color='steelblue', 
           edgecolor='black', linewidth=1)
    
    # 绘制累积曲线
    ax2 = ax.twinx()
    ax2.plot(components, cumulative_variance, 'r-', marker='o', markersize=6, 
             linewidth=2, label='累积解释方差')
    ax2.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, 
                label='90% 解释方差阈值')
    
    # 设置图表样式
    ax.set_xlabel('主成分数量', fontsize=14)
    ax.set_ylabel('解释方差比例', fontsize=14, color='steelblue')
    ax2.set_ylabel('累积解释方差', fontsize=14, color='red')
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xticks(components)
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='red')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 添加图例
    lines, labels = ax2.get_legend_handles_labels()
    ax2.legend(lines, labels, loc='lower right', fontsize=12)
    
    plt.tight_layout()
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()
    return fig

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PCA降维可视化工具')
    parser.add_argument('csv_file', help='CSV数据文件路径')
    parser.add_argument('--n_components', type=int, default=2, help='PCA降维的目标维度数（默认：2）')
    parser.add_argument('--target_column', help='目标列名（如能量值）')
    parser.add_argument('--label_column', help='标签列名（如结构类型）')
    parser.add_argument('--no_scale', action='store_true', help='不进行数据标准化')
    parser.add_argument('--save_dir', help='图表保存目录')
    parser.add_argument('--title', default='PCA降维可视化', help='图表标题')
    parser.add_argument('--colormap', default='viridis', help='颜色映射名称')
    parser.add_argument('--figsize', type=int, nargs=2, default=[12, 10], help='图表大小 (宽 高)')
    parser.add_argument('--dpi', type=int, default=300, help='图表DPI')
    parser.add_argument('--alpha', type=float, default=0.7, help='点的透明度')
    parser.add_argument('--point_size', type=int, default=80, help='点的大小')
    
    args = parser.parse_args()
    
    # 加载数据
    X, y, labels = load_data(args.csv_file, args.target_column, args.label_column)
    
    # 执行PCA降维
    X_pca, pca, explained_variance_ratio = perform_pca(
        X, n_components=args.n_components, scale=not args.no_scale
    )
    
    # 创建保存目录
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        base_filename = Path(args.csv_file).stem
        
        pca_2d_path = save_dir / f"{base_filename}_pca_2d.png"
        pca_3d_path = save_dir / f"{base_filename}_pca_3d.png"
        variance_path = save_dir / f"{base_filename}_explained_variance.png"
    else:
        pca_2d_path = None
        pca_3d_path = None
        variance_path = None
    
    # 绘制2D PCA图
    if args.n_components >= 2:
        plot_pca_2d(
            X_pca, labels=labels, target=y, 
            title=f"{args.title} (2D)",
            save_path=pca_2d_path,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            colormap=args.colormap,
            alpha=args.alpha,
            s=args.point_size
        )
    
    # 绘制3D PCA图
    if args.n_components >= 3:
        plot_pca_3d(
            X_pca, labels=labels, target=y, 
            title=f"{args.title} (3D)",
            save_path=pca_3d_path,
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            colormap=args.colormap,
            alpha=args.alpha,
            s=args.point_size
        )
    
    # 绘制解释方差比例图
    if len(explained_variance_ratio) > 1:
        plot_explained_variance(
            explained_variance_ratio,
            title=f"{args.title} - 解释方差比例",
            save_path=variance_path,
            figsize=(10, 6),
            dpi=args.dpi
        )

if __name__ == "__main__":
    main()
