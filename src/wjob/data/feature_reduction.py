#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
特征降维模块

对SOAP特征按照类别进行PCA降维
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from sklearn.decomposition import PCA
import re
import matplotlib.pyplot as plt
import matplotlib
# 中文字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

def extract_feature_prefix(feature_name: str) -> str:
    """
    从特征名称中提取前缀和后缀信息
    
    参数:
        feature_name: 特征名称，如 'M-2H_M_H_H_n10_n20_l0_atom0'
        
    返回:
        特征分组标识符，如 'M-2H_M_H_H_atom0'
    """
    parts = feature_name.split('_')
    
    # 提取后缀信息（如atom0、atom1）
    suffix = ""
    for part in parts:
        if part.startswith('atom'):
            suffix = part
            break
    
    # 检查是否包含M类型前缀（M、M-H或M-2H）
    if len(parts) >= 4 and parts[0] in ['M', 'M-H', 'M-2H']:
        # 包含M类型前缀，返回前四个部分加后缀
        prefix = f"{parts[0]}_{parts[1]}_{parts[2]}_{parts[3]}"
    elif len(parts) >= 3:
        # 没有M类型前缀，返回前三个部分（兼容旧格式）
        prefix = f"{parts[0]}_{parts[1]}_{parts[2]}"
    else:
        prefix = feature_name
    
    # 如果有后缀，将其添加到前缀中
    if suffix:
        return f"{prefix}_{suffix}"
    else:
        return prefix


def group_features_by_prefix(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    将特征按前缀分组
    
    参数:
        df: 输入数据集
        
    返回:
        按前缀分组的特征字典，格式为 {前缀: [特征名列表]}
    """
    # 排除非特征列
    non_feature_cols = ['element', 'structure_type', 'H2_adsorption_energy']
    non_feature_cols.extend([col for col in df.columns if col in [
        'Electronegativity', 'First_Ionization_Energy', 'Second_Ionization_Energy',
        'Atomic_Radius', 'Covalent_Radius', 'Period', 'Group', 'Valence_Electrons', 'Electron_Shells',
        'M_energy', 'M2H_energy'
    ]])
    
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # 按前缀分组
    grouped_features = {}
    for feature in feature_cols:
        prefix = extract_feature_prefix(feature)
        if prefix not in grouped_features:
            grouped_features[prefix] = []
        grouped_features[prefix].append(feature)
    
    return grouped_features


def apply_pca_by_group(
    df: pd.DataFrame, 
    grouped_features: Dict[str, List[str]], 
    variance_threshold: float = 0.9,
    min_features_for_pca: int = 10,
    max_components: int = None
) -> Tuple[pd.DataFrame, Dict[str, PCA]]:
    """
    对每组特征应用PCA降维，根据方差解释率自动选择主成分数量
    
    参数:
        df: 输入数据集
        grouped_features: 按前缀分组的特征字典
        variance_threshold: 方差解释率阈值，选择能解释该比例方差的最小主成分数量
        min_features_for_pca: 应用PCA的最小特征数量
        max_components: 每组最多保留的主成分数量，如果为None则不限制
        
    返回:
        包含降维后特征的数据集和PCA模型字典
    """
    # 创建结果数据框，保留非特征列
    non_feature_cols = ['element', 'structure_type', 'H2_adsorption_energy']
    non_feature_cols.extend([col for col in df.columns if col in [
        'Electronegativity', 'First_Ionization_Energy', 'Second_Ionization_Energy',
        'Atomic_Radius', 'Covalent_Radius', 'Period', 'Group', 'Valence_Electrons', 'Electron_Shells',
        'M_energy', 'M2H_energy'
    ]])
    
    result_df = df[non_feature_cols].copy()
    pca_models = {}
    
    # 对每组特征应用PCA
    for prefix, features in grouped_features.items():
        # 如果特征数量太少，不应用PCA
        if len(features) < min_features_for_pca:
            for feature in features:
                result_df[feature] = df[feature]
            continue
        
        # 首先计算全部主成分
        full_pca = PCA()
        full_pca.fit(df[features])
        
        # 计算累积方差解释率
        explained_variance = full_pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # 确定需要的主成分数量，使得方差解释率达到阈值
        n_components = np.argmax(cumulative_variance >= variance_threshold) + 1
        
        # 如果指定了最大主成分数量，则限制主成分数量
        if max_components is not None:
            n_components = min(n_components, max_components)
        
        # 确保主成分数量不超过特征数量
        n_components = min(n_components, len(features))
        
        # 重新应用PCA，使用确定的主成分数量
        pca = PCA(n_components=n_components)
        transformed = pca.fit_transform(df[features])
        
        # 添加降维后的特征到结果数据框
        for i in range(n_components):
            result_df[f"{prefix}_PC{i+1}"] = transformed[:, i]
        
        # 保存PCA模型
        pca_models[prefix] = pca
        
        # 打印方差解释率
        actual_variance = np.sum(pca.explained_variance_ratio_)
        print(f"前缀 {prefix} 的 {n_components} 个主成分解释了 {actual_variance:.2%} 的方差")
    
    return result_df, pca_models


def save_pca_components_info(pca_models: Dict[str, PCA], grouped_features: Dict[str, List[str]], output_dir: str):
    """
    保存PCA成分与原始特征的对应关系及贡献权重
    
    参数:
        pca_models: PCA模型字典
        grouped_features: 按前缀分组的特征字典
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建一个汇总文件，记录所有组的PCA信息
    summary_file = os.path.join(output_dir, "pca_components_summary.csv")
    with open(summary_file, "w") as f:
        f.write("feature_group,original_feature,principal_component,contribution\n")
    
    # 对每个特征组处理其PCA成分
    for prefix, pca in pca_models.items():
        # 获取该组的原始特征
        if prefix not in grouped_features:
            print(f"警告: 未找到特征组 {prefix} 的原始特征列表")
            continue
        
        original_features = grouped_features[prefix]
        
        # 创建该特征组的详细信息文件
        # 创建安全的文件名
        safe_prefix = re.sub(r'[^\w\-_.]', '_', prefix)
        detail_file = os.path.join(output_dir, f"pca_components_{safe_prefix}.csv")
        
        # 获取成分矩阵 (形状为 [n_components, n_features])
        # 每行是一个主成分，每列表示原始特征的贡献
        components = pca.components_
        n_components = components.shape[0]
        
        # 将成分信息写入CSV文件
        with open(detail_file, "w") as f:
            # 写入表头
            header = "principal_component," + ",".join(original_features) + "\n"
            f.write(header)
            
            # 写入每个主成分的贡献权重
            for i in range(n_components):
                component_name = f"PC{i+1}"
                row = [component_name] + [f"{weight:.6f}" for weight in components[i]]
                f.write(",".join(row) + "\n")
        
        print(f"已保存特征组 {prefix} 的PCA成分信息到 {detail_file}")
        
        # 将信息追加到汇总文件
        with open(summary_file, "a") as f:
            for i in range(n_components):
                component_name = f"{prefix}_PC{i+1}"
                for j, feat in enumerate(original_features):
                    weight = components[i, j]
                    f.write(f"{prefix},{feat},{component_name},{weight:.6f}\n")
    
    print(f"已保存所有特征组的PCA成分汇总信息到 {summary_file}")


def plot_explained_variance(pca_models: Dict[str, PCA], output_dir: str = None):
    """
    为每个特征组绘制PCA方差解释率图
    
    参数:
        pca_models: PCA模型字典
        output_dir: 图像输出目录，如果为None则不保存
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    for prefix, pca in pca_models.items():
        plt.figure(figsize=(12, 5))
        
        # 创建子图
        # plt.subplot(1, 2, 1)
        plt.bar(
            range(1, len(pca.explained_variance_ratio_) + 1),
            pca.explained_variance_ratio_,
            alpha=0.8
        )
        # 设置全局字体参数
        plt.rcParams['font.family'] = 'Arial'  # 设置字体为Arial
        plt.rcParams['font.weight'] = 'bold'  # 全局设置字体为粗体
        plt.rcParams['axes.titleweight'] = 'bold'  # 坐标轴标题加粗
        plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
        plt.rcParams['font.size'] = 24  # 全局设置字体大小
        plt.rcParams['axes.linewidth'] = 1.5  # 增加坐标轴线宽

        # 增加图表尺寸
        plt.gcf().set_size_inches(8, 12)

        plt.xlabel('Principal Component', fontsize=32,weight='bold')
        plt.ylabel('Explained Variance', fontsize=32,weight='bold')
        # plt.title('Principal Component Variance Contribution')
        
        # 确保图表有足够空间显示标签
        plt.tight_layout()
        
        # plt.subplot(1, 2, 2)
        # plt.plot(
        #     range(1, len(pca.explained_variance_ratio_) + 1),
        #     np.cumsum(pca.explained_variance_ratio_),
        #     marker='o'
        # )
        # plt.axhline(y=0.9, color='r', linestyle='-')
        # plt.xlabel('Number of Principal Components')
        # plt.ylabel('Cumulative Explained Variance')
        # plt.title('Cumulative Explained Variance')
        
        # plt.title(f'Feature Group {prefix} PCA Variance Explained')
        # plt.tight_layout()
        
        if output_dir:
            # 创建安全的文件名
            safe_prefix = re.sub(r'[^\w\-_.]', '_', prefix)
            plt.savefig(os.path.join(output_dir, f"pca_variance_{safe_prefix}.png"))
            
        plt.close()


def reduce_features(
    input_file: str,
    output_file: str = None,
    variance_threshold: float = 0.9,
    min_features_for_pca: int = 10,
    max_components: int = None,
    plot_dir: str = None
):
    """
    对数据集进行特征降维
    
    参数:
        input_file: 输入数据集文件路径
        output_file: 输出数据集文件路径
        variance_threshold: 方差解释率阈值，选择能解释该比例方差的最小主成分数量
        min_features_for_pca: 应用PCA的最小特征数量
        max_components: 每组最多保留的主成分数量，如果为None则不限制
        plot_dir: 方差解释率图表输出目录
    
    返回:
        降维后的数据集
    """
    print(f"加载数据集: {input_file}")
    df = pd.read_csv(input_file)
    print(f"原始数据集大小: {df.shape}")
    
    # 按前缀分组特征
    print("按前缀分组特征...")
    grouped_features = group_features_by_prefix(df)
    print(f"共有 {len(grouped_features)} 组特征")
    
    # 打印每组特征的数量
    for prefix, features in grouped_features.items():
        print(f"  - {prefix}: {len(features)} 个特征")
    
    # 应用PCA
    print("应用PCA降维...")
    print(f"方差解释率阈值设置为: {variance_threshold:.2%}")
    reduced_df, pca_models = apply_pca_by_group(
        df, grouped_features, variance_threshold, min_features_for_pca, max_components
    )
    print(f"降维后的数据集大小: {reduced_df.shape}")
    
    # 绘制方差解释率
    if plot_dir:
        print(f"绘制方差解释率图表到 {plot_dir}...")
        plot_explained_variance(pca_models, plot_dir)
        
        # 保存PCA成分与原始特征的对应关系
        components_dir = os.path.join(plot_dir, "components")
        save_pca_components_info(pca_models, grouped_features, components_dir)
    
    # 保存结果
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        reduced_df.to_csv(output_file, index=False)
        print(f"已保存降维后的数据集到: {output_file}")
    
    return reduced_df


def main():
    """主函数"""
    input_file = "/Users/wukai/Desktop/project/w_git/data/fea/final/normalized_dataset.csv"
    output_file = "/Users/wukai/Desktop/project/w_git/data/fea/final/reduced_dataset_90pct.csv"
    plot_dir = "/Users/wukai/Desktop/project/w_git/data/fea/final/pca_plots_90pct"
    
    reduce_features(
        input_file=input_file,
        output_file=output_file,
        variance_threshold=0.9,  # 设置方差解释率阈值为90%
        min_features_for_pca=10,
        max_components=None,  # 不限制主成分数量
        plot_dir=plot_dir
    )


if __name__ == "__main__":
    main()
