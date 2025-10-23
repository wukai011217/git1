#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分析CeO2掺杂结构与标准结构的畸变程度
计算O和Ce元素位置的偏离（畸变）大小
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from angle_analysis import read_contcar, extract_structure_info

def calculate_rmsd(coords1: np.ndarray, coords2: np.ndarray, atom_indices: List[int] = None) -> float:
    """
    计算两组坐标之间的均方根位移(RMSD)
    
    Parameters
    ----------
    coords1 : np.ndarray
        第一组原子坐标
    coords2 : np.ndarray
        第二组原子坐标
    atom_indices : List[int], optional
        要计算的原子索引，默认为None（计算所有原子）
    
    Returns
    -------
    float
        RMSD值（埃）
    """
    if atom_indices is None:
        # 使用所有原子
        selected_coords1 = coords1
        selected_coords2 = coords2
    else:
        # 只使用指定的原子
        selected_coords1 = coords1[atom_indices]
        selected_coords2 = coords2[atom_indices]
    
    # 计算均方根位移
    squared_diff = np.sum((selected_coords1 - selected_coords2) ** 2, axis=1)
    rmsd = np.sqrt(np.mean(squared_diff))
    return rmsd

def calculate_max_displacement(coords1: np.ndarray, coords2: np.ndarray, atom_indices: List[int] = None) -> Tuple[float, int]:
    """
    计算两组坐标之间的最大原子位移
    
    Parameters
    ----------
    coords1 : np.ndarray
        第一组原子坐标
    coords2 : np.ndarray
        第二组原子坐标
    atom_indices : List[int], optional
        要计算的原子索引，默认为None（计算所有原子）
    
    Returns
    -------
    Tuple[float, int]
        最大位移值（埃）和对应的原子索引
    """
    if atom_indices is None:
        # 使用所有原子
        selected_coords1 = coords1
        selected_coords2 = coords2
        indices = list(range(len(coords1)))
    else:
        # 只使用指定的原子
        selected_coords1 = coords1[atom_indices]
        selected_coords2 = coords2[atom_indices]
        indices = atom_indices
    
    # 计算每个原子的位移
    displacements = np.sqrt(np.sum((selected_coords1 - selected_coords2) ** 2, axis=1))
    
    # 找到最大位移
    max_disp_idx = np.argmax(displacements)
    max_disp = displacements[max_disp_idx]
    
    return max_disp, indices[max_disp_idx]

def align_structures(ref_coords: np.ndarray, target_coords: np.ndarray, 
                    ref_atom_indices: List[int] = None, target_atom_indices: List[int] = None) -> np.ndarray:
    """
    通过最小二乘法将目标结构对齐到参考结构
    
    Parameters
    ----------
    ref_coords : np.ndarray
        参考结构的原子坐标
    target_coords : np.ndarray
        目标结构的原子坐标
    ref_atom_indices : List[int], optional
        参考结构中用于对齐的原子索引，默认为None（使用所有原子）
    target_atom_indices : List[int], optional
        目标结构中用于对齐的原子索引，默认为None（使用所有原子）
    
    Returns
    -------
    np.ndarray
        对齐后的目标结构坐标
    """
    try:
        # 如果没有指定索引，使用所有原子
        if ref_atom_indices is None:
            ref_atom_indices = list(range(len(ref_coords)))
        if target_atom_indices is None:
            target_atom_indices = list(range(len(target_coords)))
        
        # 确保两组坐标具有相同数量的原子
        if len(ref_atom_indices) != len(target_atom_indices):
            print("警告：对齐的原子数量不同，使用简单平移对齐")
            # 简单平移对齐
            ref_centroid = np.mean(ref_coords, axis=0)
            target_centroid = np.mean(target_coords, axis=0)
            return target_coords - target_centroid + ref_centroid
        
        # 提取用于对齐的坐标
        ref_align = ref_coords[ref_atom_indices]
        target_align = target_coords[target_atom_indices]
        
        # 计算质心
        ref_centroid = np.mean(ref_align, axis=0)
        target_centroid = np.mean(target_align, axis=0)
        
        # 将坐标平移到质心
        ref_centered = ref_align - ref_centroid
        target_centered = target_align - target_centroid
        
        # 计算协方差矩阵
        covariance = np.dot(target_centered.T, ref_centered)
        
        # 使用SVD求解最优旋转矩阵
        U, S, Vt = np.linalg.svd(covariance)
        rotation_matrix = np.dot(U, Vt)
        
        # 如果行列式为负，需要翻转一个轴以保证是旋转而非反射
        if np.linalg.det(rotation_matrix) < 0:
            U[:, -1] = -U[:, -1]
            rotation_matrix = np.dot(U, Vt)
        
        # 应用旋转和平移
        aligned_coords = np.dot(target_coords - target_centroid, rotation_matrix.T) + ref_centroid
        
        return aligned_coords
    except Exception as e:
        print(f"结构对齐出错: {e}")
        print("使用简单平移对齐")
        # 简单平移对齐
        ref_centroid = np.mean(ref_coords, axis=0)
        target_centroid = np.mean(target_coords, axis=0)
        return target_coords - target_centroid + ref_centroid

def analyze_distortion(standard_structure: Dict, target_structure: Dict) -> Dict:
    """
    分析目标结构相对于标准结构的畸变程度
    
    Parameters
    ----------
    standard_structure : Dict
        标准结构数据
    target_structure : Dict
        目标结构数据
    
    Returns
    -------
    Dict
        畸变分析结果
    """
    # 提取坐标和原子类型
    std_coords = standard_structure['coords']
    std_symbols = standard_structure['atom_symbols']
    target_coords = target_structure['coords']
    target_symbols = target_structure['atom_symbols']
    
    # 找出O和Ce原子的索引
    std_o_indices = [i for i, symbol in enumerate(std_symbols) if symbol == 'O']
    std_ce_indices = [i for i, symbol in enumerate(std_symbols) if symbol == 'Ce']
    target_o_indices = [i for i, symbol in enumerate(target_symbols) if symbol == 'O']
    target_ce_indices = [i for i, symbol in enumerate(target_symbols) if symbol == 'Ce']
    
    # 找出掺杂原子（非O、Ce、H的原子）
    target_dopant_indices = [i for i, symbol in enumerate(target_symbols) 
                          if symbol not in ['O', 'Ce', 'H']]
    target_dopant_symbols = [target_symbols[i] for i in target_dopant_indices]
    
    # 打印结构信息
    print(f"标准结构: {len(std_coords)}个原子, {len(std_o_indices)}个O, {len(std_ce_indices)}个Ce")
    print(f"目标结构: {len(target_coords)}个原子, {len(target_o_indices)}个O, {len(target_ce_indices)}个Ce")
    
    if target_dopant_indices:
        print(f"目标结构中的掺杂元素: {', '.join(set(target_dopant_symbols))}")
    
    # 检查原子数量是否一致（仅作为信息，不再作为错误条件）
    if len(std_coords) != len(target_coords):
        print(f"注意：标准结构与目标结构的原子总数不同，这可能是由于掺杂或氧空位导致的")
    
    if len(std_o_indices) != len(target_o_indices):
        print(f"注意：O原子数量不同，可能存在氧空位")
    
    if len(std_ce_indices) != len(target_ce_indices):
        print(f"注意：Ce原子数量不同，可能存在Ce被掺杂元素替代的情况")
    
    # 确定用于对齐的原子索引
    # 使用Ce原子作为主要对齐参考，但需要处理Ce数量不同的情况
    if len(std_ce_indices) > 0 and len(target_ce_indices) > 0:
        # 如果两个结构都有Ce原子，使用Ce原子对齐
        # 但只使用较少数量的Ce原子（取两者的最小值）
        min_ce_count = min(len(std_ce_indices), len(target_ce_indices))
        align_std_indices = std_ce_indices[:min_ce_count]
        align_target_indices = target_ce_indices[:min_ce_count]
        print(f"使用{min_ce_count}个Ce原子进行结构对齐")
    elif len(std_o_indices) > 0 and len(target_o_indices) > 0:
        # 如果没有Ce原子，使用O原子对齐
        min_o_count = min(len(std_o_indices), len(target_o_indices))
        align_std_indices = std_o_indices[:min_o_count]
        align_target_indices = target_o_indices[:min_o_count]
        print(f"使用{min_o_count}个O原子进行结构对齐")
    else:
        # 如果既没有Ce也没有O，使用所有原子对齐
        print("无法找到Ce或O原子，使用所有原子进行简单对齐")
        align_std_indices = None
        align_target_indices = None
    
    # 对齐结构
    aligned_target_coords = align_structures(std_coords, target_coords, align_std_indices, align_target_indices)
    
    # 计算各种原子的RMSD和位移
    results = {}
    
    # 处理O原子
    if len(std_o_indices) > 0 and len(target_o_indices) > 0:
        # 计算O原子的RMSD（使用两个结构中都存在的O原子）
        min_o_count = min(len(std_o_indices), len(target_o_indices))
        o_std_indices = std_o_indices[:min_o_count]
        o_target_indices = target_o_indices[:min_o_count]
        
        # 计算O原子的RMSD
        o_rmsd = calculate_rmsd(std_coords[o_std_indices], aligned_target_coords[o_target_indices])
        results['o_rmsd'] = float(o_rmsd)
        
        # 计算O原子的最大位移和平均位移
        o_displacements = np.sqrt(np.sum((std_coords[o_std_indices] - aligned_target_coords[o_target_indices]) ** 2, axis=1))
        o_max_disp = np.max(o_displacements)
        o_max_idx = o_std_indices[np.argmax(o_displacements)]
        o_avg_disp = np.mean(o_displacements)
        
        results['o_max_displacement'] = float(o_max_disp)
        results['o_max_displacement_idx'] = int(o_max_idx)
        results['o_avg_displacement'] = float(o_avg_disp)
        results['o_displacements'] = o_displacements.tolist()
    else:
        results['o_rmsd'] = None
        results['o_max_displacement'] = None
        results['o_max_displacement_idx'] = None
        results['o_avg_displacement'] = None
        results['o_displacements'] = []
    
    # 处理Ce原子
    if len(std_ce_indices) > 0 and len(target_ce_indices) > 0:
        # 计算Ce原子的RMSD（使用两个结构中都存在的Ce原子）
        min_ce_count = min(len(std_ce_indices), len(target_ce_indices))
        ce_std_indices = std_ce_indices[:min_ce_count]
        ce_target_indices = target_ce_indices[:min_ce_count]
        
        # 计算Ce原子的RMSD
        ce_rmsd = calculate_rmsd(std_coords[ce_std_indices], aligned_target_coords[ce_target_indices])
        results['ce_rmsd'] = float(ce_rmsd)
        
        # 计算Ce原子的最大位移和平均位移
        ce_displacements = np.sqrt(np.sum((std_coords[ce_std_indices] - aligned_target_coords[ce_target_indices]) ** 2, axis=1))
        ce_max_disp = np.max(ce_displacements)
        ce_max_idx = ce_std_indices[np.argmax(ce_displacements)]
        ce_avg_disp = np.mean(ce_displacements)
        
        results['ce_max_displacement'] = float(ce_max_disp)
        results['ce_max_displacement_idx'] = int(ce_max_idx)
        results['ce_avg_displacement'] = float(ce_avg_disp)
        results['ce_displacements'] = ce_displacements.tolist()
    else:
        results['ce_rmsd'] = None
        results['ce_max_displacement'] = None
        results['ce_max_displacement_idx'] = None
        results['ce_avg_displacement'] = None
        results['ce_displacements'] = []
    
    # 计算总体RMSD（使用所有可比较的原子）
    # 找出两个结构中相同类型的原子
    comparable_atoms = []
    for atom_type in set(std_symbols) & set(target_symbols):
        std_indices = [i for i, symbol in enumerate(std_symbols) if symbol == atom_type]
        target_indices = [i for i, symbol in enumerate(target_symbols) if symbol == atom_type]
        min_count = min(len(std_indices), len(target_indices))
        if min_count > 0:
            comparable_atoms.append((std_indices[:min_count], target_indices[:min_count]))
    
    # 合并所有可比较的原子索引
    all_std_indices = [idx for indices_pair in comparable_atoms for idx in indices_pair[0]]
    all_target_indices = [idx for indices_pair in comparable_atoms for idx in indices_pair[1]]
    
    if all_std_indices and all_target_indices:
        total_rmsd = calculate_rmsd(std_coords[all_std_indices], aligned_target_coords[all_target_indices])
        results['total_rmsd'] = float(total_rmsd)
    else:
        results['total_rmsd'] = None
    
    # 添加结构信息
    results['structure_info'] = target_structure.get('structure_info', {})
    results['std_atom_count'] = len(std_coords)
    results['target_atom_count'] = len(target_coords)
    results['std_o_count'] = len(std_o_indices)
    results['std_ce_count'] = len(std_ce_indices)
    results['target_o_count'] = len(target_o_indices)
    results['target_ce_count'] = len(target_ce_indices)
    results['target_dopant_symbols'] = target_dopant_symbols
    
    return results

def analyze_multiple_structures(standard_path: str, target_paths: List[str], 
                               output_dir: str = None) -> Dict:
    """
    分析多个结构相对于标准结构的畸变程度
    
    Parameters
    ----------
    standard_path : str
        标准结构文件路径
    target_paths : List[str]
        目标结构文件路径列表
    output_dir : str, optional
        输出目录，默认为None
    
    Returns
    -------
    Dict
        所有结构的畸变分析结果
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 读取标准结构
    print(f"读取标准结构: {standard_path}")
    standard_structure = read_contcar(standard_path)
    if standard_structure is None:
        print(f"无法读取标准结构: {standard_path}")
        return {'error': f"无法读取标准结构: {standard_path}"}
    
    # 分析所有目标结构
    results = []
    failed_files = []
    
    for target_path in target_paths:
        print(f"分析结构: {target_path}")
        try:
            # 读取目标结构
            target_structure = read_contcar(target_path)
            if target_structure is None:
                print(f"无法读取目标结构: {target_path}")
                failed_files.append(target_path)
                continue
            
            # 分析畸变
            distortion_result = analyze_distortion(standard_structure, target_structure)
            
            # 如果有错误，记录并继续
            if 'error' in distortion_result:
                print(f"分析错误: {distortion_result['error']}")
                failed_files.append(target_path)
                continue
            
            # 添加文件路径信息
            distortion_result['file'] = target_path
            
            # 添加到结果列表
            results.append(distortion_result)
            
        except Exception as e:
            print(f"处理文件时出错: {target_path}, 错误: {e}")
            failed_files.append(target_path)
    
    # 将结果转换为DataFrame
    if results:
        results_df = pd.DataFrame(results)
        
        # 添加特征组信息
        for idx, path in enumerate(target_paths):
            if path in results_df['file'].values:
                if 'high_feature_paths' in globals() and path in high_feature_paths:
                    group = 'high'
                elif 'medium_feature_paths' in globals() and path in medium_feature_paths:
                    group = 'medium'
                elif 'low_feature_paths' in globals() and path in low_feature_paths:
                    group = 'low'
                else:
                    group = 'unknown'
                
                results_df.loc[results_df['file'] == path, 'feature_group'] = group
        
        # 保存结果
        if output_dir:
            csv_path = os.path.join(output_dir, 'structure_distortion_analysis.csv')
            results_df.to_csv(csv_path, index=False)
            print(f"分析结果已保存到: {csv_path}")
    
    return {
        'results': results,
        'failed_files': failed_files,
        'standard_structure': standard_path
    }

def visualize_distortion_results(results: Dict, output_dir: str = None) -> None:
    """
    可视化畸变分析结果
    
    Parameters
    ----------
    results : Dict
        畸变分析结果
    output_dir : str, optional
        输出目录，默认为None
    """
    if not results.get('results'):
        print("没有可视化的结果")
        return
    
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 将结果转换为DataFrame
    df = pd.DataFrame(results['results'])
    
    # 设置可视化样式
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    
    # 1. 按特征组对比总体RMSD
    if 'feature_group' in df.columns:
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='feature_group', y='total_rmsd', data=df, palette='Set2')
        plt.xlabel('Feature Group')
        plt.ylabel('Total RMSD (Å)')
        plt.title('Comparison of Total RMSD by Feature Group')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_group_total_rmsd_boxplot.png'), dpi=300)
        plt.close()
        
        # 2. 按特征组对比O原子RMSD
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='feature_group', y='o_rmsd', data=df, palette='Set2')
        plt.xlabel('Feature Group')
        plt.ylabel('O Atoms RMSD (Å)')
        plt.title('Comparison of O Atoms RMSD by Feature Group')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_group_o_rmsd_boxplot.png'), dpi=300)
        plt.close()
        
        # 3. 按特征组对比Ce原子RMSD
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='feature_group', y='ce_rmsd', data=df, palette='Set2')
        plt.xlabel('Feature Group')
        plt.ylabel('Ce Atoms RMSD (Å)')
        plt.title('Comparison of Ce Atoms RMSD by Feature Group')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_group_ce_rmsd_boxplot.png'), dpi=300)
        plt.close()
        
        # 4. 按特征组对比O原子最大位移
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='feature_group', y='o_max_displacement', data=df, palette='Set2')
        plt.xlabel('Feature Group')
        plt.ylabel('O Max Displacement (Å)')
        plt.title('Comparison of O Max Displacement by Feature Group')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_group_o_max_disp_boxplot.png'), dpi=300)
        plt.close()
        
        # 5. 按特征组对比Ce原子最大位移
        plt.figure(figsize=(12, 8))
        sns.boxplot(x='feature_group', y='ce_max_displacement', data=df, palette='Set2')
        plt.xlabel('Feature Group')
        plt.ylabel('Ce Max Displacement (Å)')
        plt.title('Comparison of Ce Max Displacement by Feature Group')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_group_ce_max_disp_boxplot.png'), dpi=300)
        plt.close()
        
        # 6. 小提琴图：更详细地展示O和Ce的RMSD分布
        plt.figure(figsize=(14, 10))
        df_melt = pd.melt(df, id_vars=['feature_group'], value_vars=['o_rmsd', 'ce_rmsd'],
                         var_name='Atom Type', value_name='RMSD')
        df_melt['Atom Type'] = df_melt['Atom Type'].map({'o_rmsd': 'O', 'ce_rmsd': 'Ce'})
        
        sns.violinplot(x='feature_group', y='RMSD', hue='Atom Type', data=df_melt, 
                      palette='Set2', split=True, inner='quartile')
        plt.xlabel('Feature Group')
        plt.ylabel('RMSD (Å)')
        plt.title('Distribution of O and Ce RMSD by Feature Group')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'feature_group_o_ce_rmsd_violin.png'), dpi=300)
        plt.close()
        
        # 7. 散点图：O和Ce的RMSD关系
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x='o_rmsd', y='ce_rmsd', hue='feature_group', data=df, palette='Set2', alpha=0.7)
        plt.xlabel('O Atoms RMSD (Å)')
        plt.ylabel('Ce Atoms RMSD (Å)')
        plt.title('Relationship between O and Ce RMSD by Feature Group')
        plt.legend(title='Feature Group')
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'o_ce_rmsd_scatter.png'), dpi=300)
        plt.close()
    
    # 8. 热图：掺杂元素与RMSD的关系
    if 'structure_info' in df.columns and df['structure_info'].apply(lambda x: 'dopant' in x).any():
        # 提取掺杂元素
        df['dopant'] = df['structure_info'].apply(lambda x: x.get('dopant', 'unknown'))
        
        # 计算每个掺杂元素的平均RMSD
        pivot_df = df.pivot_table(
            values=['total_rmsd', 'o_rmsd', 'ce_rmsd'], 
            index='dopant',
            aggfunc='mean'
        ).fillna(0)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(pivot_df, annot=True, cmap='YlGnBu', fmt='.3f')
        plt.title('Average RMSD by Dopant Element')
        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'dopant_rmsd_heatmap.png'), dpi=300)
        plt.close()
    
    # 9. 统计分析
    # 按特征组统计
    if 'feature_group' in df.columns:
        group_stats = df.groupby('feature_group').agg({
            'total_rmsd': ['count', 'mean', 'std', 'min', 'max'],
            'o_rmsd': ['mean', 'std', 'min', 'max'],
            'ce_rmsd': ['mean', 'std', 'min', 'max'],
            'o_max_displacement': ['mean', 'std', 'min', 'max'],
            'ce_max_displacement': ['mean', 'std', 'min', 'max']
        }).round(4)
        
        print("\n按特征组统计畸变程度:")
        print(group_stats)
        
        # 保存统计结果
        if output_dir:
            group_stats.to_csv(os.path.join(output_dir, 'feature_group_distortion_stats.csv'))
    
    # 10. 统计显著性检验
    try:
        from scipy import stats
        
        if 'feature_group' in df.columns:
            # 进行ANOVA检验，比较三组之间的差异
            groups = [df[df['feature_group'] == g]['total_rmsd'] for g in df['feature_group'].unique()]
            f_val, p_val = stats.f_oneway(*groups)
            
            print("\n特征组间总体RMSD差异的ANOVA检验:")
            print(f"F值: {f_val:.4f}, p值: {p_val:.4f}")
            print(f"结论: {'存在显著差异' if p_val < 0.05 else '无显著差异'} (p < 0.05)")
            
            # 两两比较
            group_names = df['feature_group'].unique()
            print("\n特征组两两比较 (t检验):")
            for i, g1 in enumerate(group_names):
                for g2 in group_names[i+1:]:
                    t_val, p_val = stats.ttest_ind(
                        df[df['feature_group'] == g1]['total_rmsd'],
                        df[df['feature_group'] == g2]['total_rmsd'],
                        equal_var=False  # 不假设方差相等
                    )
                    print(f"{g1} vs {g2}: t = {t_val:.4f}, p = {p_val:.4f}, {'显著差异' if p_val < 0.05 else '无显著差异'}")
    except ImportError:
        print("未安装scipy，跳过统计显著性检验")

# 定义全局变量
high_feature_paths = []
medium_feature_paths = []
low_feature_paths = []

def main():
    """主函数"""
    print("开始分析CeO2掺杂结构与标准结构的畸变程度...")
    
    # 定义文件路径
    global high_feature_paths, medium_feature_paths, low_feature_paths
    
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
    
    # 标准结构
    standard_path = '/Users/wukai/Desktop/project/wjob/data/raw/test/cont/CeO2/Doped-111/Ov0/None/di/M-2H/ads1/CONTCAR'
    
    # 所有目标结构
    all_target_paths = high_feature_paths + medium_feature_paths + low_feature_paths
    
    # 创建输出目录
    output_dir = '/Users/wukai/Desktop/project/wjob/results/figures/structure_distortion'
    os.makedirs(output_dir, exist_ok=True)
    
    # 分析所有结构
    print("\n1. 分析所有结构的畸变程度")
    analysis_results = analyze_multiple_structures(standard_path, all_target_paths, output_dir)
    
    # 查看处理结果
    print(f"\n成功处理 {len(analysis_results.get('results', []))} 个文件")
    print(f"失败处理 {len(analysis_results.get('failed_files', []))} 个文件")
    
    # 可视化结果
    print("\n2. 可视化畸变分析结果")
    visualize_distortion_results(analysis_results, output_dir)
    
    print("\n分析完成！所有结果已保存到:", output_dir)
    
    return analysis_results

if __name__ == "__main__":
    main()
