#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
以H为筛选中心的O与Ce几何关系分析脚本
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
    process_contcar_files,
    plot_angle_distributions
)

def main():
    """主函数"""
    print("开始分析以H为筛选中心的O与Ce几何关系...")
    
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
    output_dir = '/Users/wukai/Desktop/project/wjob/results/figures/angle_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理示例文件
    print("\n1. 处理示例文件进行验证")
    example_file = high_feature_paths[0]
    print(f"处理示例文件: {example_file}")
    
    # 读取CONTCAR文件
    structure_data = read_contcar(example_file)
    
    # 查看结构信息
    print("\n结构信息:")
    print(structure_data['structure_info'])
    
    # 查看原子类型和数量
    atom_types = set(structure_data['atom_symbols'])
    atom_counts = {atom: structure_data['atom_symbols'].count(atom) for atom in atom_types}
    print("\n原子类型和数量:")
    print(atom_counts)
    
    # 分析O与Ce的几何关系
    results = analyze_angles(structure_data)
    
    # 查看键角结果
    print(f"\n找到 {len(results['bond_angles'])} 个O-Ce几何关系")
    if results['bond_angles']:
        bond_df = pd.DataFrame(results['bond_angles'])
        print("\n键角统计:")
        print(bond_df[['angle_with_z', 'angle_with_plane', 'O_Ce_distance']].describe())
        
        # 显示前几个键角
        print("\n前5个O-Ce几何关系:")
        print(bond_df[['O_idx', 'Ce_idx', 'angle_with_z', 'angle_with_plane', 'O_Ce_distance']].head())
    
    # 查看二面角结果
    print(f"\n找到 {len(results['dihedral_angles'])} 个O-Ce-O二面角")
    if results['dihedral_angles']:
        dihedral_df = pd.DataFrame(results['dihedral_angles'])
        print("\n二面角统计:")
        print(dihedral_df['dihedral'].describe())
        
        # 显示前几个二面角
        print("\n前5个O-Ce-O二面角:")
        print(dihedral_df[['O1_idx', 'Ce_idx', 'O2_idx', 'dihedral']].head())
    
    # 处理所有文件
    print("\n2. 处理所有CONTCAR文件")
    all_results = process_contcar_files(all_paths)
    
    # 查看处理结果
    print(f"\n成功处理 {len(all_results['processed_files'])} 个文件")
    print(f"失败处理 {len(all_results['failed_files'])} 个文件")
    print(f"\n共找到 {len(all_results['bond_angles'])} 个O-Ce几何关系")
    print(f"共找到 {len(all_results['dihedral_angles'])} 个O-Ce-O二面角")
    
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
    
    # 按特征组统计
    print("\n3. 按特征组统计O-Ce几何关系")
    feature_group_stats = bond_df.groupby('feature_group')['angle_with_z'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
    print(feature_group_stats)
    
    # 按空位类型统计
    if 'vacancy_type' in bond_df.columns:
        print("\n按空位类型统计O-Ce几何关系:")
        vacancy_stats = bond_df.groupby('vacancy_type')['angle_with_z'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        print(vacancy_stats)
    
    # 按掉换元素统计
    if 'dopant' in bond_df.columns:
        print("\n按掉换元素统计O-Ce几何关系:")
        dopant_stats = bond_df.groupby('dopant')['angle_with_z'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        print(dopant_stats)
    
    # 绘制图表
    print("\n4. 绘制O-Ce几何关系图表")
    
    # 设置更好的可视化样式
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    
    # 绘制总体分布图
    print("绘制总体分布图...")
    plot_angle_distributions(all_results, output_dir)
    
    # 按特征组分组绘图
    print("按特征组绘制分布图...")
    plot_angle_distributions(all_results, os.path.join(output_dir, 'by_feature_group'), group_by='feature_group')
    
    # 按空位类型分组绘图
    print("按空位类型绘制分布图...")
    plot_angle_distributions(all_results, os.path.join(output_dir, 'by_vacancy'), group_by='vacancy_type')
    
    # 按掉换元素分组绘图
    print("按掉换元素绘制分布图...")
    plot_angle_distributions(all_results, os.path.join(output_dir, 'by_dopant'), group_by='dopant')
    
    # 保存结果
    print("\n5. 保存分析结果")
    
    # 保存键角结果
    bond_csv_path = os.path.join(output_dir, 'o_ce_geometry.csv')
    bond_df.to_csv(bond_csv_path, index=False)
    print(f"O-Ce几何关系结果已保存到: {bond_csv_path}")
    
    # 保存二面角结果
    if not dihedral_df.empty:
        dihedral_csv_path = os.path.join(output_dir, 'o_ce_o_dihedrals.csv')
        dihedral_df.to_csv(dihedral_csv_path, index=False)
        print(f"O-Ce-O二面角结果已保存到: {dihedral_csv_path}")
    
    # 保存统计结果
    feature_group_stats.to_csv(os.path.join(output_dir, 'o_ce_feature_group_stats.csv'), index=False)
    print(f"特征组统计结果已保存到: {os.path.join(output_dir, 'o_ce_feature_group_stats.csv')}")
    
    if 'vacancy_type' in bond_df.columns:
        vacancy_stats.to_csv(os.path.join(output_dir, 'o_ce_vacancy_stats.csv'), index=False)
        print(f"空位类型统计结果已保存到: {os.path.join(output_dir, 'o_ce_vacancy_stats.csv')}")
    
    if 'dopant' in bond_df.columns:
        dopant_stats.to_csv(os.path.join(output_dir, 'o_ce_dopant_stats.csv'), index=False)
        print(f"掉换元素统计结果已保存到: {os.path.join(output_dir, 'o_ce_dopant_stats.csv')}")
    
    print("\n分析完成！所有结果已保存到:", output_dir)
    
    # 返回结果供其他脚本使用
    return {
        'bond_df': bond_df,
        'dihedral_df': dihedral_df,
        'all_results': all_results,
        'output_dir': output_dir
    }

if __name__ == "__main__":
    main()
