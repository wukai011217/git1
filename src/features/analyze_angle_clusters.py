#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
将O-Ce-O键角分为三类区间进行统计分析
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from scipy import stats

def identify_angle_clusters(df_path, n_clusters=3):
    """
    使用K-means聚类将角度分为n_clusters类
    
    Parameters
    ----------
    df_path : str
        O-Ce-O键角数据文件路径
    n_clusters : int, optional
        聚类数量，默认为3
        
    Returns
    -------
    DataFrame
        添加了聚类标签的数据框
    """
    print(f"读取数据: {df_path}")
    df = pd.read_csv(df_path)
    
    # 使用角度值进行聚类
    X = df['O_Ce_O_angle'].values.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['angle_cluster'] = kmeans.fit_predict(X)
    
    # 按角度值排序聚类标签
    cluster_centers = kmeans.cluster_centers_.flatten()
    cluster_order = np.argsort(cluster_centers)
    cluster_map = {old: new for new, old in enumerate(cluster_order)}
    df['angle_cluster'] = df['angle_cluster'].map(cluster_map)
    
    # 为每个聚类添加描述性标签
    cluster_labels = ['小角度', '中角度', '大角度']
    df['angle_type'] = df['angle_cluster'].map({i: label for i, label in enumerate(cluster_labels[:n_clusters])})
    
    # 计算每个聚类的角度范围
    cluster_ranges = {}
    for i in range(n_clusters):
        cluster_data = df[df['angle_cluster'] == i]['O_Ce_O_angle']
        cluster_ranges[i] = (cluster_data.min(), cluster_data.max())
    
    print("角度聚类结果:")
    for i in range(n_clusters):
        print(f"聚类 {i} ({cluster_labels[i]}): 中心值 = {cluster_centers[cluster_order[i]]:.2f}°, 范围 = [{cluster_ranges[i][0]:.2f}°, {cluster_ranges[i][1]:.2f}°]")
    
    return df, cluster_ranges

def analyze_angle_clusters(df, output_dir):
    """
    分析角度聚类结果
    
    Parameters
    ----------
    df : DataFrame
        带有聚类标签的数据框
    output_dir : str
        输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 按聚类和特征组统计
    cluster_stats = df.groupby(['angle_type', 'feature_group']).agg({
        'O_Ce_O_angle': ['count', 'mean', 'std', 'min', 'max'],
        'O1_Ce_distance': ['mean', 'std'],
        'O2_Ce_distance': ['mean', 'std'],
        'O1_O2_distance': ['mean', 'std']
    }).round(2)
    
    print("\n按角度类型和特征组的统计:")
    print(cluster_stats)
    
    # 保存统计结果
    cluster_stats.to_csv(os.path.join(output_dir, 'angle_clusters_stats.csv'))
    
    # 2. 计算每个特征组中各角度类型的比例
    angle_type_counts = df.groupby(['feature_group', 'angle_type']).size().unstack(fill_value=0)
    angle_type_props = angle_type_counts.div(angle_type_counts.sum(axis=1), axis=0) * 100
    
    print("\n各特征组中角度类型的比例 (%):")
    print(angle_type_props.round(2))
    
    # 保存比例结果
    angle_type_props.to_csv(os.path.join(output_dir, 'angle_type_proportions.csv'))
    
    # 3. 绘制聚类散点图
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='O1_Ce_distance', y='O_Ce_O_angle', hue='angle_type', 
                    style='feature_group', data=df, alpha=0.7)
    plt.xlabel('O1-Ce Distance (Å)')
    plt.ylabel('O-Ce-O Angle (degrees)')
    plt.title('O-Ce-O Angles Clustered by Angle Type')
    plt.legend(title='Angle Type')
    plt.savefig(os.path.join(output_dir, 'angle_clusters_scatter.png'), dpi=300)
    plt.close()
    
    # 4. 绘制堆叠柱状图，显示各特征组中角度类型的比例
    angle_type_props.plot(kind='bar', stacked=True, figsize=(10, 6), 
                          colormap='viridis')
    plt.xlabel('Feature Group')
    plt.ylabel('Proportion (%)')
    plt.title('Proportion of Angle Types by Feature Group')
    plt.legend(title='Angle Type')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angle_type_proportions.png'), dpi=300)
    plt.close()
    
    # 5. 绘制箱线图，按角度类型和特征组分组
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='angle_type', y='O_Ce_O_angle', hue='feature_group', data=df)
    plt.xlabel('Angle Type')
    plt.ylabel('O-Ce-O Angle (degrees)')
    plt.title('O-Ce-O Angles by Cluster and Feature Group')
    plt.legend(title='Feature Group')
    plt.savefig(os.path.join(output_dir, 'angle_clusters_boxplot.png'), dpi=300)
    plt.close()
    
    # 6. 绘制小提琴图，按角度类型和特征组分组
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='angle_type', y='O_Ce_O_angle', hue='feature_group', data=df,
                  split=True, inner='quartile')
    plt.xlabel('Angle Type')
    plt.ylabel('O-Ce-O Angle (degrees)')
    plt.title('O-Ce-O Angles Distribution by Cluster and Feature Group')
    plt.legend(title='Feature Group')
    plt.savefig(os.path.join(output_dir, 'angle_clusters_violin.png'), dpi=300)
    plt.close()
    
    # 7. 热图：特征组与角度类型的交叉统计
    pivot_df = pd.pivot_table(
        df, 
        values='O_Ce_O_angle',
        index='feature_group',
        columns='angle_type',
        aggfunc='count'
    ).fillna(0)
    
    # 计算百分比
    pivot_pct = pivot_df.div(pivot_df.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_pct, annot=True, cmap='YlGnBu', fmt='.1f')
    plt.title('Percentage of Angle Types by Feature Group')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angle_type_heatmap.png'), dpi=300)
    plt.close()
    
    # 8. 统计显著性检验
    print("\n统计显著性检验:")
    
    # 8.1 卡方检验：特征组与角度类型的关联性
    contingency_table = pd.crosstab(df['feature_group'], df['angle_type'])
    chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
    print(f"特征组与角度类型的卡方检验: chi2 = {chi2:.4f}, p = {p:.4f}")
    print(f"结论: {'存在显著关联' if p < 0.05 else '无显著关联'} (p < 0.05)")
    
    # 8.2 每种角度类型内，特征组间的ANOVA检验
    for angle_type in df['angle_type'].unique():
        angle_data = df[df['angle_type'] == angle_type]
        groups = [angle_data[angle_data['feature_group'] == g]['O_Ce_O_angle'] 
                 for g in angle_data['feature_group'].unique()]
        if len(groups) > 1 and all(len(g) > 0 for g in groups):
            f_val, p_val = stats.f_oneway(*groups)
            print(f"\n角度类型 '{angle_type}' 内特征组间的ANOVA检验:")
            print(f"F值: {f_val:.4f}, p值: {p_val:.4f}")
            print(f"结论: {'存在显著差异' if p_val < 0.05 else '无显著差异'} (p < 0.05)")
    
    return angle_type_props

def main():
    """主函数"""
    # 输入输出路径
    input_file = '/Users/wukai/Desktop/project/wjob/results/figures/o_ce_o_angles/o_ce_o_angles.csv'
    output_dir = '/Users/wukai/Desktop/project/wjob/results/figures/angle_clusters'
    
    # 确保输入文件存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        return
    
    # 识别角度聚类
    df, cluster_ranges = identify_angle_clusters(input_file)
    
    # 分析角度聚类
    angle_type_props = analyze_angle_clusters(df, output_dir)
    
    # 保存带有聚类标签的数据
    df.to_csv(os.path.join(output_dir, 'o_ce_o_angles_clustered.csv'), index=False)
    
    print(f"\n分析完成! 结果已保存到: {output_dir}")
    
    return df, angle_type_props

if __name__ == "__main__":
    main()
