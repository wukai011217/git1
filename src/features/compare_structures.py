#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对比三类结构（high/medium/low feature）的O-Ce几何关系
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from angle_analysis import (
    read_contcar, 
    extract_structure_info, 
    analyze_angles,
    process_contcar_files
)

def compare_feature_groups(all_results, output_dir=None):
    """
    对比三类结构的O-Ce几何关系
    
    Parameters
    ----------
    all_results : Dict
        所有结构的分析结果
    output_dir : str, optional
        输出目录
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 将结果转换为DataFrame
    bond_df = pd.DataFrame(all_results['bond_angles'])
    dihedral_df = pd.DataFrame(all_results['dihedral_angles']) if all_results['dihedral_angles'] else pd.DataFrame()
    
    # 设置更好的可视化样式
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 1. 按特征组对比O-Ce与z轴的角度分布
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='feature_group', y='angle_with_z', data=bond_df, palette='Set2')
    plt.xlabel('特征组')
    plt.ylabel('O-Ce与z轴的角度 (度)')
    plt.title('不同特征组的O-Ce与z轴角度分布对比')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_group_angle_z_boxplot.png'), dpi=300)
    plt.show()
    
    # 2. 按特征组对比O-Ce与平面的角度分布
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='feature_group', y='angle_with_plane', data=bond_df, palette='Set2')
    plt.xlabel('特征组')
    plt.ylabel('O-Ce与平面的角度 (度)')
    plt.title('不同特征组的O-Ce与平面角度分布对比')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_group_angle_plane_boxplot.png'), dpi=300)
    plt.show()
    
    # 3. 按特征组对比O-Ce距离分布
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='feature_group', y='O_Ce_distance', data=bond_df, palette='Set2')
    plt.xlabel('特征组')
    plt.ylabel('O-Ce距离 (埃)')
    plt.title('不同特征组的O-Ce距离分布对比')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_group_distance_boxplot.png'), dpi=300)
    plt.show()
    
    # 4. 按特征组对比O-Ce-O二面角分布
    if not dihedral_df.empty and 'feature_group' in dihedral_df.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='feature_group', y='dihedral', data=dihedral_df, palette='Set2')
        plt.xlabel('特征组')
        plt.ylabel('O-Ce-O二面角 (度)')
        plt.title('不同特征组的O-Ce-O二面角分布对比')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_group_dihedral_boxplot.png'), dpi=300)
        plt.show()
    
    # 5. 小提琴图：更详细地展示分布
    plt.figure(figsize=(14, 10))
    sns.violinplot(x='feature_group', y='angle_with_z', data=bond_df, palette='Set2', inner='quartile')
    plt.xlabel('特征组')
    plt.ylabel('O-Ce与z轴的角度 (度)')
    plt.title('不同特征组的O-Ce与z轴角度分布对比 (小提琴图)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_group_angle_z_violin.png'), dpi=300)
    plt.show()
    
    # 6. 散点图：角度与距离的关系，按特征组着色
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='O_Ce_distance', y='angle_with_z', hue='feature_group', data=bond_df, palette='Set2', alpha=0.7)
    plt.xlabel('O-Ce距离 (埃)')
    plt.ylabel('O-Ce与z轴的角度 (度)')
    plt.title('O-Ce距离与角度的关系 (按特征组)')
    plt.legend(title='特征组')
    if output_dir:
        plt.savefig(os.path.join(output_dir, 'feature_group_angle_distance_scatter.png'), dpi=300)
    plt.show()
    
    # 7. 热图：特征组与掺杂元素的交叉统计
    if 'dopant' in bond_df.columns:
        # 计算每个掺杂元素-特征组组合的平均角度
        pivot_df = bond_df.pivot_table(
            values='angle_with_z', 
            index='dopant', 
            columns='feature_group', 
            aggfunc='mean'
        ).fillna(0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title('不同掺杂元素和特征组的O-Ce与z轴平均角度')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'dopant_feature_group_heatmap.png'), dpi=300)
        plt.show()
    
    # 8. 统计分析
    # 按特征组统计
    group_stats = bond_df.groupby('feature_group').agg({
        'angle_with_z': ['count', 'mean', 'std', 'min', 'max'],
        'angle_with_plane': ['mean', 'std', 'min', 'max'],
        'O_Ce_distance': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    print("\n按特征组统计O-Ce几何关系:")
    print(group_stats)
    
    # 保存统计结果
    if output_dir:
        group_stats.to_csv(os.path.join(output_dir, 'feature_group_detailed_stats.csv'))
    
    # 9. 按掺杂元素和特征组的交叉统计
    if 'dopant' in bond_df.columns:
        dopant_group_stats = bond_df.groupby(['dopant', 'feature_group']).agg({
            'angle_with_z': ['count', 'mean', 'std'],
            'O_Ce_distance': ['mean', 'std']
        }).round(2)
        
        print("\n按掺杂元素和特征组的交叉统计:")
        print(dopant_group_stats)
        
        # 保存交叉统计结果
        if output_dir:
            dopant_group_stats.to_csv(os.path.join(output_dir, 'dopant_feature_group_stats.csv'))
    
    # 10. 按空位类型和特征组的交叉统计
    if 'vacancy_type' in bond_df.columns:
        vacancy_group_stats = bond_df.groupby(['vacancy_type', 'feature_group']).agg({
            'angle_with_z': ['count', 'mean', 'std'],
            'O_Ce_distance': ['mean', 'std']
        }).round(2)
        
        print("\n按空位类型和特征组的交叉统计:")
        print(vacancy_group_stats)
        
        # 保存交叉统计结果
        if output_dir:
            vacancy_group_stats.to_csv(os.path.join(output_dir, 'vacancy_feature_group_stats.csv'))
    
    # 11. 统计显著性检验
    try:
        from scipy import stats
        
        # 进行ANOVA检验，比较三组之间的差异
        groups = [bond_df[bond_df['feature_group'] == g]['angle_with_z'] for g in bond_df['feature_group'].unique()]
        f_val, p_val = stats.f_oneway(*groups)
        
        print("\n特征组间角度差异的ANOVA检验:")
        print(f"F值: {f_val:.4f}, p值: {p_val:.4f}")
        print(f"结论: {'存在显著差异' if p_val < 0.05 else '无显著差异'} (p < 0.05)")
        
        # 两两比较
        group_names = bond_df['feature_group'].unique()
        print("\n特征组两两比较 (t检验):")
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i+1:]:
                t_val, p_val = stats.ttest_ind(
                    bond_df[bond_df['feature_group'] == g1]['angle_with_z'],
                    bond_df[bond_df['feature_group'] == g2]['angle_with_z'],
                    equal_var=False  # 不假设方差相等
                )
                print(f"{g1} vs {g2}: t = {t_val:.4f}, p = {p_val:.4f}, {'显著差异' if p_val < 0.05 else '无显著差异'}")
    except ImportError:
        print("未安装scipy，跳过统计显著性检验")
    
    return {
        'bond_df': bond_df,
        'dihedral_df': dihedral_df,
        'group_stats': group_stats
    }

def main():
    """主函数"""
    print("开始对比三类结构的O-Ce几何关系...")
    
    # 定义文件路径
    high_feature_paths = [
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Os/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Ir/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Sec-2/Tc/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Sec-1/Os/M-2H/ads/CONTCAR'
    ]

    medium_feature_paths = [
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Fir-1/Tl/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Fir-1/Zr/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Sec-2/Sb/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Al/M-2H/ads/CONTCAR'
    ]

    low_feature_paths = [
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Sec-1/Zn/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Sec-1/Cd/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Cd/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Mg/M-2H/ads/CONTCAR'
    ]
    
    # 所有文件路径
    all_paths = high_feature_paths + medium_feature_paths + low_feature_paths
    
    # 创建输出目录
    output_dir = '/Users/wukai/Desktop/project/wjob/results/figures/structure_comparison'
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有文件
    print("\n1. 处理所有CONTCAR文件")
    all_results = process_contcar_files(all_paths)
    
    # 查看处理结果
    print(f"\n成功处理 {len(all_results['processed_files'])} 个文件")
    print(f"失败处理 {len(all_results['failed_files'])} 个文件")
    
    # 将结果转换为DataFrame
    bond_df = pd.DataFrame(all_results['bond_angles'])
    dihedral_df = pd.DataFrame(all_results['dihedral_angles']) if all_results['dihedral_angles'] else pd.DataFrame()
    
    # 添加特征组信息
    for idx, path in enumerate(all_paths):
        if path in high_feature_paths:
            group = 'high'
        elif path in medium_feature_paths:
            group = 'medium'
        else:
            group = 'low'
            
        bond_df.loc[bond_df['file'] == path, 'feature_group'] = group
        if not dihedral_df.empty:
            dihedral_df.loc[dihedral_df['file'] == path, 'feature_group'] = group
    
    # 更新结果
    all_results['bond_angles'] = bond_df.to_dict('records')
    if not dihedral_df.empty:
        all_results['dihedral_angles'] = dihedral_df.to_dict('records')
    
    # 对比三类结构
    print("\n2. 对比三类结构的O-Ce几何关系")
    comparison_results = compare_feature_groups(all_results, output_dir)
    
    # 保存结果
    print("\n3. 保存分析结果")
    
    # 保存键角结果
    bond_csv_path = os.path.join(output_dir, 'o_ce_geometry_comparison.csv')
    bond_df.to_csv(bond_csv_path, index=False)
    print(f"O-Ce几何关系结果已保存到: {bond_csv_path}")
    
    # 保存二面角结果
    if not dihedral_df.empty:
        dihedral_csv_path = os.path.join(output_dir, 'o_ce_o_dihedrals_comparison.csv')
        dihedral_df.to_csv(dihedral_csv_path, index=False)
        print(f"O-Ce-O二面角结果已保存到: {dihedral_csv_path}")
    
    print("\n分析完成！所有结果已保存到:", output_dir)
    
    return comparison_results

if __name__ == "__main__":
    main()
