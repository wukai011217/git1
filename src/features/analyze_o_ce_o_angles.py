#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
使用新的分析逻辑分析高、标准、低三类结构的O-Ce-O键角
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

def plot_o_ce_o_angles(results, output_dir=None, group_by='feature_group'):
    """
    绘制O-Ce-O键角分布图
    
    Parameters
    ----------
    results : Dict
        分析结果
    output_dir : str, optional
        输出目录
    group_by : str, optional
        分组字段
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 将结果转换为DataFrame
    o_ce_o_df = pd.DataFrame(results['o_ce_o_angles'])
    
    if o_ce_o_df.empty:
        print("没有找到O-Ce-O键角数据")
        return
    
    # 设置更好的可视化样式
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    
    # 1. 按特征组对比O-Ce-O键角分布
    plt.figure(figsize=(12, 8))
    sns.boxplot(x=group_by, y='O_Ce_O_angle', data=o_ce_o_df)
    plt.xlabel(group_by.replace('_', ' ').title())
    plt.ylabel('O-Ce-O Angle (degrees)')
    plt.title(f'Comparison of O-Ce-O Angles by {group_by.replace("_", " ").title()}')
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'o_ce_o_angle_boxplot_by_{group_by}.png'), dpi=300)
    plt.close()
    
    # 2. 小提琴图：更详细地展示分布
    plt.figure(figsize=(14, 10))
    sns.violinplot(x=group_by, y='O_Ce_O_angle', data=o_ce_o_df, inner='quartile')
    plt.xlabel(group_by.replace('_', ' ').title())
    plt.ylabel('O-Ce-O Angle (degrees)')
    plt.title(f'Comparison of O-Ce-O Angles by {group_by.replace("_", " ").title()} (Violin Plot)')
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'o_ce_o_angle_violin_by_{group_by}.png'), dpi=300)
    plt.close()
    
    # 3. 散点图：角度与距离的关系
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x='O1_Ce_distance', y='O_Ce_O_angle', hue=group_by, data=o_ce_o_df, alpha=0.7)
    plt.xlabel('O1-Ce Distance (Å)')
    plt.ylabel('O-Ce-O Angle (degrees)')
    plt.title('Relationship between O-Ce Distance and O-Ce-O Angle')
    plt.legend(title=group_by.replace('_', ' ').title())
    if output_dir:
        plt.savefig(os.path.join(output_dir, f'o_ce_o_angle_distance_scatter_by_{group_by}.png'), dpi=300)
    plt.close()
    
    # 4. 热图：特征组与掺杂元素的交叉统计
    if 'dopant' in o_ce_o_df.columns and group_by != 'dopant':
        # 计算每个掺杂元素-特征组组合的平均角度
        pivot_df = o_ce_o_df.pivot_table(
            values='O_Ce_O_angle', 
            index='dopant', 
            columns=group_by, 
            aggfunc='mean'
        ).fillna(0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.1f')
        plt.title(f'Average O-Ce-O Angle by Dopant and {group_by.replace("_", " ").title()}')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'dopant_{group_by}_heatmap.png'), dpi=300)
        plt.close()
    
    # 5. 统计分析
    # 按特征组统计
    group_stats = o_ce_o_df.groupby(group_by).agg({
        'O_Ce_O_angle': ['count', 'mean', 'std', 'min', 'max'],
        'O1_Ce_distance': ['mean', 'std', 'min', 'max'],
        'O2_Ce_distance': ['mean', 'std', 'min', 'max'],
        'O1_O2_distance': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    print(f"\n按{group_by}统计O-Ce-O键角:")
    print(group_stats)
    
    # 保存统计结果
    if output_dir:
        group_stats.to_csv(os.path.join(output_dir, f'{group_by}_detailed_stats.csv'))
    
    # 6. 统计显著性检验
    try:
        from scipy import stats
        
        # 进行ANOVA检验，比较三组之间的差异
        groups = [o_ce_o_df[o_ce_o_df[group_by] == g]['O_Ce_O_angle'] for g in o_ce_o_df[group_by].unique()]
        f_val, p_val = stats.f_oneway(*groups)
        
        print(f"\n{group_by}间O-Ce-O键角差异的ANOVA检验:")
        print(f"F值: {f_val:.4f}, p值: {p_val:.4f}")
        print(f"结论: {'存在显著差异' if p_val < 0.05 else '无显著差异'} (p < 0.05)")
        
        # 两两比较
        group_names = o_ce_o_df[group_by].unique()
        print(f"\n{group_by}两两比较 (t检验):")
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i+1:]:
                t_val, p_val = stats.ttest_ind(
                    o_ce_o_df[o_ce_o_df[group_by] == g1]['O_Ce_O_angle'],
                    o_ce_o_df[o_ce_o_df[group_by] == g2]['O_Ce_O_angle'],
                    equal_var=False  # 不假设方差相等
                )
                print(f"{g1} vs {g2}: t = {t_val:.4f}, p = {p_val:.4f}, {'显著差异' if p_val < 0.05 else '无显著差异'}")
    except ImportError:
        print("未安装scipy，跳过统计显著性检验")
    
    return o_ce_o_df


def analyze_structures():
    """分析高、标准、低三类结构的O-Ce-O键角"""
    print("开始分析高、标准、低三类结构的O-Ce-O键角...")
    
    # 定义文件路径
    high_feature_paths = [
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Os/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Ir/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Sec-2/Tc/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Sec-1/Os/M-2H/ads/CONTCAR'
    ]

    # 标准CeO2参杂结构
    standard_feature_paths = [
        '/Users/wukai/Desktop/project/wjob/data/raw/test/cont/CeO2/Doped-111/Ov0/None/di/M-2H/ads1/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/test/cont/CeO2/Doped-111/Ov0/None/di/M-2H/ads/CONTCAR'
    ]

    low_feature_paths = [
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Sec-1/Zn/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov1/Sec-1/Cd/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Cd/M-2H/ads/CONTCAR',
        '/Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Mg/M-2H/ads/CONTCAR'
    ]
    
    # 所有文件路径
    all_paths = high_feature_paths + standard_feature_paths + low_feature_paths
    
    # 创建输出目录
    output_dir = '/Users/wukai/Desktop/project/wjob/results/figures/o_ce_o_angles'
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理所有文件
    all_results = {
        'bond_angles': [],
        'dihedral_angles': [],
        'o_ce_o_angles': [],
        'file_paths': all_paths,
        'processed_files': [],
        'failed_files': []
    }
    
    print("\n1. 处理所有CONTCAR文件")
    for file_path in all_paths:
        print(f"处理文件: {file_path}")
        try:
            # 读取CONTCAR文件
            structure_data = read_contcar(file_path)
            if structure_data is None:
                all_results['failed_files'].append(file_path)
                continue
            
            # 分析键角和二面角
            angles_data = analyze_angles(structure_data)
            
            # 添加文件路径信息
            for bond_angle in angles_data['bond_angles']:
                bond_angle['file'] = file_path
            for dihedral_angle in angles_data['dihedral_angles']:
                dihedral_angle['file'] = file_path
            for o_ce_o_angle in angles_data['o_ce_o_angles']:
                o_ce_o_angle['file'] = file_path
            
            # 合并结果
            all_results['bond_angles'].extend(angles_data['bond_angles'])
            all_results['dihedral_angles'].extend(angles_data['dihedral_angles'])
            all_results['o_ce_o_angles'].extend(angles_data['o_ce_o_angles'])
            all_results['processed_files'].append(file_path)
        except Exception as e:
            print(f"处理文件时出错: {e}")
            all_results['failed_files'].append(file_path)
    
    print(f"处理完成: {len(all_results['processed_files'])}个文件成功, {len(all_results['failed_files'])}个文件失败")
    
    # 将结果转换为DataFrame
    bond_df = pd.DataFrame(all_results['bond_angles'])
    dihedral_df = pd.DataFrame(all_results['dihedral_angles']) if all_results['dihedral_angles'] else pd.DataFrame()
    o_ce_o_df = pd.DataFrame(all_results['o_ce_o_angles']) if all_results['o_ce_o_angles'] else pd.DataFrame()
    
    # 添加特征组信息
    for idx, path in enumerate(all_paths):
        if path in high_feature_paths:
            group = 'high'
        elif path in standard_feature_paths:
            group = 'standard'
        else:
            group = 'low'
            
        bond_df.loc[bond_df['file'] == path, 'feature_group'] = group
        if not dihedral_df.empty:
            dihedral_df.loc[dihedral_df['file'] == path, 'feature_group'] = group
        if not o_ce_o_df.empty:
            o_ce_o_df.loc[o_ce_o_df['file'] == path, 'feature_group'] = group
    
    # 更新结果
    all_results['bond_angles'] = bond_df.to_dict('records')
    if not dihedral_df.empty:
        all_results['dihedral_angles'] = dihedral_df.to_dict('records')
    if not o_ce_o_df.empty:
        all_results['o_ce_o_angles'] = o_ce_o_df.to_dict('records')
    
    # 分析O-Ce-O键角
    print("\n2. 分析O-Ce-O键角")
    if not o_ce_o_df.empty:
        o_ce_o_df = plot_o_ce_o_angles(all_results, output_dir, 'feature_group')
        
        # 保存O-Ce-O键角结果
        o_ce_o_csv_path = os.path.join(output_dir, 'o_ce_o_angles.csv')
        o_ce_o_df.to_csv(o_ce_o_csv_path, index=False)
        print(f"O-Ce-O键角结果已保存到: {o_ce_o_csv_path}")
    else:
        print("没有找到O-Ce-O键角数据")
    
    # 保存键角和二面角结果
    bond_csv_path = os.path.join(output_dir, 'bond_angles.csv')
    bond_df.to_csv(bond_csv_path, index=False)
    print(f"键角结果已保存到: {bond_csv_path}")
    
    if not dihedral_df.empty:
        dihedral_csv_path = os.path.join(output_dir, 'dihedral_angles.csv')
        dihedral_df.to_csv(dihedral_csv_path, index=False)
        print(f"二面角结果已保存到: {dihedral_csv_path}")
    
    print("\n分析完成！所有结果已保存到:", output_dir)
    
    return all_results


if __name__ == "__main__":
    analyze_structures()
