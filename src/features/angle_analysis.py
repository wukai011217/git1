#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
计算VASP CONTCAR文件中以H为中心的O与Ce的键角与二面角统计
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.pyplot as plt
import re


def read_contcar(file_path: str) -> Dict:
    """
    读取CONTCAR文件并解析原子坐标
    
    Parameters
    ----------
    file_path : str
        CONTCAR文件路径
    
    Returns
    -------
    Dict
        包含原子类型、数量和坐标的字典
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"文件不存在: {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None
    
    # 第一行是注释
    comment = lines[0].strip()
    
    # 第二行是缩放因子
    scale = float(lines[1].strip())
    
    # 第3-5行是晶格向量
    lattice = np.zeros((3, 3))
    for i in range(3):
        lattice[i] = np.array([float(x) for x in lines[i+2].split()])
    
    # 第6行是原子类型
    atom_types = lines[5].split()
    
    # 第7行是每种原子的数量
    atom_counts = [int(x) for x in lines[6].split()]
    
    # 检查是否有Selective dynamics行
    line_idx = 7
    if lines[line_idx].strip().lower().startswith('s'):
        line_idx += 1
    
    # 检查是Direct还是Cartesian坐标
    coord_type = lines[line_idx].strip()
    is_direct = coord_type.lower().startswith('d')
    line_idx += 1
    
    # 读取原子坐标
    total_atoms = sum(atom_counts)
    coords = np.zeros((total_atoms, 3))
    
    for i in range(total_atoms):
        if line_idx + i < len(lines):
            parts = lines[line_idx + i].split()
            if len(parts) >= 3:
                coords[i] = np.array([float(x) for x in parts[:3]])
    
    # 如果是分数坐标，转换为笛卡尔坐标
    if is_direct:
        coords = np.dot(coords, lattice)
    
    # 构建原子类型列表
    atom_symbols = []
    for t, c in zip(atom_types, atom_counts):
        atom_symbols.extend([t] * c)
    
    # 提取结构信息
    structure_info = extract_structure_info(file_path)
    
    return {
        'comment': comment,
        'scale': scale,
        'lattice': lattice,
        'atom_types': atom_types,
        'atom_counts': atom_counts,
        'coords': coords,
        'atom_symbols': atom_symbols,
        'is_direct': is_direct,
        'structure_info': structure_info
    }


def extract_structure_info(file_path: str) -> Dict[str, str]:
    """
    从文件路径中提取结构信息
    
    Parameters
    ----------
    file_path : str
        CONTCAR文件路径
    
    Returns
    -------
    Dict[str, str]
        结构信息字典
    """
    # 示例路径: /Users/wukai/Desktop/project/wjob/data/raw/final/cont/CeO2/Doped-111/Ov0/None/Ag/M-2H/ads/CONTCAR
    pattern = r".*cont/CeO2/Doped-111/(Ov\d+)/([^/]+)/([^/]+)/M-2H/ads/CONTCAR"
    match = re.search(pattern, file_path)
    
    info = {
        'path': file_path,
        'filename': os.path.basename(file_path),
        'oxygen_vacancy': 'unknown',
        'vacancy_position': 'unknown',
        'dopant': 'unknown'
    }
    
    if match:
        info['oxygen_vacancy'] = match.group(1)  # Ov0 or Ov1
        info['vacancy_position'] = match.group(2)  # None, Fir-1, Sec-1, etc.
        info['dopant'] = match.group(3)  # Ag, Os, etc.
        
        # 根据记忆中的映射关系转换
        if info['oxygen_vacancy'] == 'Ov1':
            if info['vacancy_position'] == 'Fir-1':
                info['vacancy_type'] = 'Ov-surf1'
            elif info['vacancy_position'] == 'Sec-2':
                info['vacancy_type'] = 'Ov-sub2'
            elif info['vacancy_position'] == 'Fir-2':
                info['vacancy_type'] = 'Ov-surf2'
            elif info['vacancy_position'] == 'Sec-1':
                info['vacancy_type'] = 'Ov-sub1'
        elif info['oxygen_vacancy'] == 'Ov0' and info['vacancy_position'] == 'None':
            info['vacancy_type'] = 'pristine'
    
    return info


def calculate_angle(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    计算两个向量之间的角度（度）
    
    Parameters
    ----------
    v1, v2 : np.ndarray
        两个3D向量
    
    Returns
    -------
    float
        角度（度）
    """
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    # 处理零向量情况
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    
    cos_angle = dot / (norm1 * norm2)
    # 处理数值误差，确保cos_angle在[-1, 1]范围内
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180 / np.pi
    return angle


def calculate_dihedral(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> float:
    """
    计算四个点定义的二面角
    
    Parameters
    ----------
    p1, p2, p3, p4 : np.ndarray
        四个3D点的坐标
    
    Returns
    -------
    float
        二面角（度）
    """
    try:
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        # 计算法向量
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        # 检查法向量是否为零向量
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-10 or n2_norm < 1e-10:
            return 0.0
        
        # 归一化
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # 计算二面角
        b2_norm = np.linalg.norm(b2)
        if b2_norm < 1e-10:
            return 0.0
            
        m1 = np.cross(n1, b2 / b2_norm)
        x = np.dot(n1, n2)
        y = np.dot(m1, n2)
        
        # 处理数值误差
        angle = np.arctan2(y, x) * 180 / np.pi
        return angle
    except Exception as e:
        print(f"计算二面角时出错: {e}")
        return 0.0


def find_nearest_atoms(coords: np.ndarray, atom_symbols: List[str], 
                       center_idx: int, target_symbol: str, n: int = 1,
                       max_distance: float = 3.0) -> List[int]:
    """
    找到距离中心原子最近的n个目标类型原子，并限制最大距离
    
    Parameters
    ----------
    coords : np.ndarray
        所有原子的坐标
    atom_symbols : List[str]
        所有原子的元素符号
    center_idx : int
        中心原子的索引
    target_symbol : str
        目标原子类型
    n : int, optional
        要找的最近原子数量，默认为1
    max_distance : float, optional
        最大距离限制（埃），默认为3.0
    
    Returns
    -------
    List[int]
        最近的n个目标原子的索引
    """
    center_coord = coords[center_idx]
    distances = []
    
    for i, (coord, symbol) in enumerate(zip(coords, atom_symbols)):
        if symbol == target_symbol and i != center_idx:
            dist = np.linalg.norm(coord - center_coord)
            if dist <= max_distance:  # 只考虑在最大距离内的原子
                distances.append((i, dist))
    
    # 按距离排序
    distances.sort(key=lambda x: x[1])
    
    # 返回最近的n个原子的索引，但不超过找到的原子数量
    return [idx for idx, _ in distances[:min(n, len(distances))]]


def analyze_angles(structure_data: Dict) -> Dict:
    """
    以H为筛选中心，分析结构中附近的O与Ce原子之间的键角关系
    
    新的分析逻辑：
    1. 以H为筛选中心，找到所有H原子
    2. 对每个H原子，寻找所有距离≤6Å的O原子
    3. 对每个H原子，寻找所有距离≤6Å的Ce原子
    4. 计算O-Ce-O键角，特别是每个Ce最近一层的O之间的键角（距离<3Å）
    
    Parameters
    ----------
    structure_data : Dict
        从CONTCAR文件读取的结构数据
    
    Returns
    -------
    Dict
        包含键角和二面角信息的字典
    """
    if structure_data is None:
        return {'bond_angles': [], 'dihedral_angles': [], 'o_ce_o_angles': []}
    
    coords = structure_data['coords']
    atom_symbols = structure_data['atom_symbols']
    structure_info = structure_data.get('structure_info', {})
    
    # 找到所有H原子的索引
    h_indices = [i for i, symbol in enumerate(atom_symbols) if symbol == 'H']
    
    results = {
        'bond_angles': [],
        'dihedral_angles': [],
        'o_ce_o_angles': [],  # 新增：O-Ce-O键角
        'structure_info': structure_info
    }
    
    for h_idx in h_indices:
        # 找到6Å内的所有O原子
        nearby_o = find_nearest_atoms(coords, atom_symbols, h_idx, 'O', n=100, max_distance=6.0)
        
        # 找到6Å内的所有Ce原子
        nearby_ce = find_nearest_atoms(coords, atom_symbols, h_idx, 'Ce', n=100, max_distance=6.0)
        
        # 如果找到了O和Ce原子，计算所有可能的O-Ce键角
        for o_idx in nearby_o:
            for ce_idx in nearby_ce:
                # 计算O-Ce键角（以H为筛选中心）
                v1 = coords[o_idx] - coords[ce_idx]  # O到Ce的矢量
                
                # 计算O-Ce之间的距离
                o_ce_distance = np.linalg.norm(v1)
                
                # 计算H到O和Ce的距离（用于记录）
                h_o_distance = np.linalg.norm(coords[o_idx] - coords[h_idx])
                h_ce_distance = np.linalg.norm(coords[ce_idx] - coords[h_idx])
                
                # 计算O-Ce与参考坐标轴的角度
                # 使用z轴作为参考轴
                z_axis = np.array([0, 0, 1])
                angle_with_z = calculate_angle(v1, z_axis)
                
                # 计算O-Ce与平面的角度（与平面法向的角度）
                xy_plane_normal = np.array([0, 0, 1])
                angle_with_plane = 90 - calculate_angle(v1, xy_plane_normal)
                
                results['bond_angles'].append({
                    'H_idx': int(h_idx),
                    'O_idx': int(o_idx),
                    'Ce_idx': int(ce_idx),
                    'O_Ce_distance': float(o_ce_distance),
                    'angle_with_z': float(angle_with_z),
                    'angle_with_plane': float(angle_with_plane),
                    'H_O_distance': float(h_o_distance),
                    'H_Ce_distance': float(h_ce_distance),
                    'dopant': structure_info.get('dopant', 'unknown'),
                    'vacancy_type': structure_info.get('vacancy_type', 'unknown')
                })
        
        # 计算O-Ce-O键角，特别是每个Ce最近一层的O之间的键角
        for ce_idx in nearby_ce:
            # 找到距离Ce原子<3Å的所有O原子（最近一层）
            ce_coord = coords[ce_idx]
            nearest_o_to_ce = []
            
            for o_idx in nearby_o:
                o_coord = coords[o_idx]
                o_ce_distance = np.linalg.norm(o_coord - ce_coord)
                if o_ce_distance < 3.0:  # 最近一层的O原子
                    nearest_o_to_ce.append((o_idx, o_ce_distance))
            
            # 计算最近一层O原子之间的O-Ce-O键角
            for i, (o1_idx, o1_ce_dist) in enumerate(nearest_o_to_ce):
                for j, (o2_idx, o2_ce_dist) in enumerate(nearest_o_to_ce):
                    if i != j:  # 确保使用两个不同的O原子
                        # 计算O1-Ce-O2键角
                        v1 = coords[o1_idx] - coords[ce_idx]  # O1到Ce的矢量
                        v2 = coords[o2_idx] - coords[ce_idx]  # O2到Ce的矢量
                        o_ce_o_angle = calculate_angle(v1, v2)
                        
                        # 计算O1-O2距离
                        o1_o2_distance = np.linalg.norm(coords[o1_idx] - coords[o2_idx])
                        
                        results['o_ce_o_angles'].append({
                            'H_idx': int(h_idx),  # H仅用于筛选
                            'Ce_idx': int(ce_idx),
                            'O1_idx': int(o1_idx),
                            'O2_idx': int(o2_idx),
                            'O_Ce_O_angle': float(o_ce_o_angle),
                            'O1_Ce_distance': float(o1_ce_dist),
                            'O2_Ce_distance': float(o2_ce_dist),
                            'O1_O2_distance': float(o1_o2_distance),
                            'dopant': structure_info.get('dopant', 'unknown'),
                            'vacancy_type': structure_info.get('vacancy_type', 'unknown')
                        })
        
        # 计算二面角（O-Ce-O）
        if len(nearby_o) >= 2 and len(nearby_ce) >= 1:
            for ce_idx in nearby_ce:
                # 找到距离Ce原子<3Å的所有O原子（最近一层）
                ce_coord = coords[ce_idx]
                nearest_o_to_ce = []
                
                for o_idx in nearby_o:
                    o_coord = coords[o_idx]
                    o_ce_distance = np.linalg.norm(o_coord - ce_coord)
                    if o_ce_distance < 3.0:  # 最近一层的O原子
                        nearest_o_to_ce.append((o_idx, o_ce_distance))
                
                # 只考虑最近一层的O原子计算二面角
                for i, (o1_idx, _) in enumerate(nearest_o_to_ce):
                    for j, (o2_idx, _) in enumerate(nearest_o_to_ce):
                        if i != j:  # 确保使用两个不同的O原子
                            # 计算O1-Ce-O2二面角
                            dihedral = calculate_dihedral(
                                coords[o1_idx], coords[ce_idx], 
                                coords[ce_idx], coords[o2_idx]
                            )
                            
                            # 计算距离
                            o1_ce_distance = np.linalg.norm(coords[o1_idx] - coords[ce_idx])
                            o2_ce_distance = np.linalg.norm(coords[o2_idx] - coords[ce_idx])
                            o1_o2_distance = np.linalg.norm(coords[o1_idx] - coords[o2_idx])
                            
                            results['dihedral_angles'].append({
                                'H_idx': int(h_idx),  # H仅用于筛选
                                'O1_idx': int(o1_idx),
                                'Ce_idx': int(ce_idx),
                                'O2_idx': int(o2_idx),
                                'dihedral': float(dihedral),
                                'O1_Ce_distance': float(o1_ce_distance),
                                'O2_Ce_distance': float(o2_ce_distance),
                                'O1_O2_distance': float(o1_o2_distance),
                                'dopant': structure_info.get('dopant', 'unknown'),
                                'vacancy_type': structure_info.get('vacancy_type', 'unknown')
                            })
    
    return results


def process_contcar_files(file_paths: List[str]) -> Dict:
    """
    处理多个CONTCAR文件并统计键角和二面角
    
    Parameters
    ----------
    file_paths : List[str]
        CONTCAR文件路径列表
    
    Returns
    -------
    Dict
        包含所有文件的键角和二面角统计信息
    """
    all_results = {
        'bond_angles': [],
        'dihedral_angles': [],
        'file_paths': file_paths,
        'processed_files': [],
        'failed_files': []
    }
    
    for file_path in file_paths:
        try:
            print(f"处理文件: {file_path}")
            structure_data = read_contcar(file_path)
            if structure_data is None:
                all_results['failed_files'].append({'path': file_path, 'error': '无法读取文件'})
                continue
                
            results = analyze_angles(structure_data)
            
            # 提取结构信息
            structure_info = structure_data.get('structure_info', {})
            
            # 添加文件路径信息
            for angle_data in results['bond_angles']:
                angle_data['file'] = file_path
            for dihedral_data in results['dihedral_angles']:
                dihedral_data['file'] = file_path
            
            all_results['bond_angles'].extend(results['bond_angles'])
            all_results['dihedral_angles'].extend(results['dihedral_angles'])
            all_results['processed_files'].append({
                'path': file_path,
                'structure_info': structure_info,
                'bond_angles_count': len(results['bond_angles']),
                'dihedral_angles_count': len(results['dihedral_angles'])
            })
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
            all_results['failed_files'].append({'path': file_path, 'error': str(e)})
    
    print(f"处理完成: {len(all_results['processed_files'])}个文件成功, {len(all_results['failed_files'])}个文件失败")
    return all_results


def plot_angle_distributions(results: Dict, output_dir: str = None, group_by: str = None):
    """
    绘制键角和二面角分布图
    
    Parameters
    ----------
    results : Dict
        键角和二面角统计结果
    output_dir : str, optional
        输出目录，默认为None（不保存）
    group_by : str, optional
        按特定字段分组，如'dopant'或'vacancy_type'
    """
    # 创建输出目录
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有数据，直接返回
    if not results['bond_angles'] and not results['dihedral_angles']:
        print("没有数据可供绘图")
        return
    
    # 将数据转换为DataFrame便于处理
    bond_df = pd.DataFrame(results['bond_angles']) if results['bond_angles'] else None
    dihedral_df = pd.DataFrame(results['dihedral_angles']) if results['dihedral_angles'] else None
    
    # 按组绘制图表
    if group_by and bond_df is not None and group_by in bond_df.columns:
        groups = bond_df[group_by].unique()
        
        # 绘制O-Ce与z轴的角度分布（按组）
        plt.figure(figsize=(12, 8))
        for group in groups:
            group_data = bond_df[bond_df[group_by] == group]
            plt.hist(group_data['angle_with_z'], bins=20, alpha=0.5, label=f'{group_by}={group}')
        
        plt.xlabel('O-Ce与z轴的角度 (度)')
        plt.ylabel('频次')
        plt.title(f'O-Ce与z轴的角度分布 (按{group_by}分组)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'angle_with_z_by_{group_by}.png'), dpi=300)
        plt.show()
        
        # 绘制O-Ce与平面的角度分布（按组）
        plt.figure(figsize=(12, 8))
        for group in groups:
            group_data = bond_df[bond_df[group_by] == group]
            plt.hist(group_data['angle_with_plane'], bins=20, alpha=0.5, label=f'{group_by}={group}')
        
        plt.xlabel('O-Ce与平面的角度 (度)')
        plt.ylabel('频次')
        plt.title(f'O-Ce与平面的角度分布 (按{group_by}分组)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, f'angle_with_plane_by_{group_by}.png'), dpi=300)
        plt.show()
        
        # 绘制二面角分布（按组）
        if dihedral_df is not None and group_by in dihedral_df.columns:
            plt.figure(figsize=(12, 8))
            for group in dihedral_df[group_by].unique():
                group_data = dihedral_df[dihedral_df[group_by] == group]
                plt.hist(group_data['dihedral'], bins=20, alpha=0.5, label=f'{group_by}={group}')
            
            plt.xlabel('O-Ce-O二面角 (度)')
            plt.ylabel('频次')
            plt.title(f'O-Ce-O二面角分布 (按{group_by}分组)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            if output_dir:
                plt.savefig(os.path.join(output_dir, f'dihedral_angles_by_{group_by}.png'), dpi=300)
            plt.show()
    
    # 绘制总体分布
    # 绘制O-Ce与z轴的角度分布
    if bond_df is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(bond_df['angle_with_z'], bins=20, alpha=0.7, color='blue')
        plt.xlabel('O-Ce与z轴的角度 (度)')
        plt.ylabel('频次')
        plt.title('O-Ce与z轴的角度分布')
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'angle_with_z_distribution.png'), dpi=300)
        plt.show()
        
        # 绘制O-Ce与平面的角度分布
        plt.figure(figsize=(10, 6))
        plt.hist(bond_df['angle_with_plane'], bins=20, alpha=0.7, color='blue')
        plt.xlabel('O-Ce与平面的角度 (度)')
        plt.ylabel('频次')
        plt.title('O-Ce与平面的角度分布')
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'angle_with_plane_distribution.png'), dpi=300)
        plt.show()
        
        # 绘制角度与距离的关系
        plt.figure(figsize=(10, 6))
        plt.scatter(bond_df['O_Ce_distance'], bond_df['angle_with_z'], alpha=0.5)
        plt.xlabel('O-Ce距离 (埃)')
        plt.ylabel('O-Ce与z轴的角度 (度)')
        plt.title('O-Ce距离与角度的关系')
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'angle_vs_OCe_distance.png'), dpi=300)
        plt.show()
        
        # 绘制H原子距离与角度的关系（H仅作为筛选中心）
        if 'H_O_distance' in bond_df.columns and 'H_Ce_distance' in bond_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(bond_df['H_O_distance'], bond_df['angle_with_z'], alpha=0.5)
            plt.xlabel('H-O距离 (埃)')
            plt.ylabel('O-Ce与z轴的角度 (度)')
            plt.title('H-O距离与O-Ce角度的关系')
            plt.grid(True, alpha=0.3)
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'angle_vs_HO_distance.png'), dpi=300)
            plt.show()
    
    # 绘制二面角分布
    if dihedral_df is not None:
        plt.figure(figsize=(10, 6))
        plt.hist(dihedral_df['dihedral'], bins=20, alpha=0.7, color='green')
        plt.xlabel('O-Ce-O二面角 (度)')
        plt.ylabel('频次')
        plt.title('O-Ce-O二面角分布')
        plt.grid(True, alpha=0.3)
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'dihedral_angles_distribution.png'), dpi=300)
        plt.show()
        
        # 绘制二面角与距离的关系
        if 'O1_Ce_distance' in dihedral_df.columns and 'O2_Ce_distance' in dihedral_df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(dihedral_df['O1_O2_distance'], dihedral_df['dihedral'], alpha=0.5)
            plt.xlabel('O1-O2距离 (埃)')
            plt.ylabel('O-Ce-O二面角 (度)')
            plt.title('O1-O2距离与O-Ce-O二面角的关系')
            plt.grid(True, alpha=0.3)
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'dihedral_vs_O1O2_distance.png'), dpi=300)
            plt.show()


def main():
    """主函数"""
    # 真实的文件路径
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
    
    # 处理所有文件
    print("处理所有CONTCAR文件...")
    all_results = process_contcar_files(all_paths)
    
    # 分组处理结果
    if all_results['bond_angles'] and all_results['dihedral_angles']:
        # 将结果转换为DataFrame
        bond_df = pd.DataFrame(all_results['bond_angles'])
        dihedral_df = pd.DataFrame(all_results['dihedral_angles'])
        
        # 添加特征组信息
        for idx, path in enumerate(all_paths):
            if path in high_feature_paths:
                group = 'high'
            elif path in medium_feature_paths:
                group = 'medium'
            else:
                group = 'low'
                
            bond_df.loc[bond_df['file'] == path, 'feature_group'] = group
            dihedral_df.loc[dihedral_df['file'] == path, 'feature_group'] = group
        
        # 绘制总体分布图
        print("绘制总体键角和二面角分布...")
        plot_angle_distributions(all_results, output_dir)
        
        # 按空位类型分组绘图
        print("按空位类型绘制键角和二面角分布...")
        plot_angle_distributions(all_results, os.path.join(output_dir, 'by_vacancy'), group_by='vacancy_type')
        
        # 按掉换元素分组绘图
        print("按掉换元素绘制键角和二面角分布...")
        plot_angle_distributions(all_results, os.path.join(output_dir, 'by_dopant'), group_by='dopant')
        
        # 按特征组分组绘图
        print("按特征组绘制键角和二面角分布...")
        plot_angle_distributions(all_results, os.path.join(output_dir, 'by_feature_group'), group_by='feature_group')
        
        # 保存结果为CSV
        print("保存结果到CSV文件...")
        bond_df.to_csv(os.path.join(output_dir, 'bond_angles.csv'), index=False)
        dihedral_df.to_csv(os.path.join(output_dir, 'dihedral_angles.csv'), index=False)
        
        # 计算统计信息
        print("计算统计信息...")
        # 按特征组统计
        feature_group_stats = bond_df.groupby('feature_group')['angle'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
        feature_group_stats.to_csv(os.path.join(output_dir, 'bond_angles_feature_group_stats.csv'), index=False)
        
        # 按空位类型统计
        if 'vacancy_type' in bond_df.columns:
            vacancy_stats = bond_df.groupby('vacancy_type')['angle'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            vacancy_stats.to_csv(os.path.join(output_dir, 'bond_angles_vacancy_stats.csv'), index=False)
        
        # 按掉换元素统计
        if 'dopant' in bond_df.columns:
            dopant_stats = bond_df.groupby('dopant')['angle'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
            dopant_stats.to_csv(os.path.join(output_dir, 'bond_angles_dopant_stats.csv'), index=False)
        
        print("分析完成！")
        print(f"结果已保存到: {output_dir}")
        
        # 返回结果供其他脚本使用
        return {
            'bond_df': bond_df,
            'dihedral_df': dihedral_df,
            'all_results': all_results,
            'output_dir': output_dir
        }
    else:
        print("没有有效的结果数据")
        return None


if __name__ == "__main__":
    main()
