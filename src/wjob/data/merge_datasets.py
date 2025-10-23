#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
合并SOAP特征、吸附能和元素性质数据

此模块将多个数据源合并为一个统一的数据集，用于后续机器学习分析：
1. SOAP特征数据（从fea目录中的各个元素和结构）
2. 氢吸附能数据（h2_adsorption_energy.csv）
3. 元素基础性质数据（ele_propety.csv）

每个样本代表一种元素在五种结构（pristine和四种氧空位）上的组合。
"""

import os
import glob
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from wjob.config import DEFAULT_STRUCTURE_NAME


def find_soap_files(base_dir: str) -> Dict[str, Dict[str, Dict[str, Tuple[str, str]]]]:
    """
    在指定目录下查找所有SOAP特征文件，按元素和结构类型组织
    
    参数:
        base_dir: 基础目录，通常是data/fea/4.24/cont
        
    返回:
        嵌套字典，格式为 {元素: {结构类型: {中心原子: (特征文件路径, 元数据文件路径)}}}
    """
    result = {}
    
    # 遍历所有可能的元素目录
    for element_path in glob.glob(f"{base_dir}/CeO2/Doped-111/*/*/*/M*/ads"):
        # 解析路径获取元素和结构类型
        parts = element_path.split('/')
        
        # 找到元素名称（通常是倒数第三个部分）
        element = parts[-3]
        
        # 找到结构类型（Ov类型、位置和M类型）
        ov_type = parts[-5]  # Ov0 或 Ov1
        position = parts[-4]  # None, Fir-1, Fir-2, Sec-1, Sec-2
        m_type = parts[-2]    # M, M-H, M-2H

        # 构建结构标识符
        if ov_type == "Ov0" and position == "None":
            base_structure = "pristine"
        else:
            # 使用配置中的映射
            base_structure = DEFAULT_STRUCTURE_NAME.get(position, position)

        # 将M类型添加到结构标识符中
        structure_type = f"{base_structure}_{m_type}"
        
        # 查找该目录下的所有SOAP特征文件
        for feature_file in glob.glob(f"{element_path}/soap_*.npy"):
            # 从文件名获取中心原子类型
            feature_filename = os.path.basename(feature_file)
            centre_atom = feature_filename.replace('soap_', '').replace('.npy', '')
            
            # 对应的元数据文件
            meta_file = feature_file.replace('.npy', '_meta.csv')
            
            # 检查文件是否存在
            if os.path.exists(feature_file) and os.path.exists(meta_file):
                # 将元素添加到结果字典
                if element not in result:
                    result[element] = {}
                
                # 将结构类型添加到元素字典
                if structure_type not in result[element]:
                    result[element][structure_type] = {}
                
                # 将中心原子和文件路径添加到结构类型字典
                result[element][structure_type][centre_atom] = (feature_file, meta_file)
    
    return result


def load_and_combine_features(
    soap_files: Dict[str, Dict[str, Dict[str, Tuple[str, str]]]],
    filter_elements: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    加载并合并所有SOAP特征数据
    
    参数:
        soap_files: 由find_soap_files返回的文件映射
        filter_elements: 可选，要排除的元素列表（如Ce）
        
    返回:
        合并后的特征DataFrame，每行代表一个元素在一种结构上的特征
    """
    # 创建一个字典，用于按元素和基础结构类型组织数据
    element_data = {}
    
    # 遍历每个元素
    for element, structures in soap_files.items():
        if element in filter_elements:
            continue
        # 跳过没有任何结构的元素
        if not structures:  
            continue
        
        # 遍历每种结构
        for structure_type, centre_atoms in structures.items():
            # 跳过没有任何中心原子的结构
            if not centre_atoms:
                continue
            
            # 解析结构类型，分离基础结构和M类型
            parts = structure_type.split('_')
            if len(parts) != 2:
                continue
                
            base_structure, m_type = parts
            
            # 创建元素-基础结构的键
            element_structure_key = f"{element}_{base_structure}"
            
            # 如果这个元素-基础结构组合还没有在字典中，创建它
            if element_structure_key not in element_data:
                element_data[element_structure_key] = {
                    "element": element,
                    "structure_type": base_structure
                }
            
            # 获取该元素-基础结构的行数据
            row_data = element_data[element_structure_key]
            
            # 处理每种中心原子的特征
            for centre_atom, (feature_file, meta_file) in centre_atoms.items():
                # 加载元数据
                meta_df = pd.read_csv(meta_file)
                
                # # 过滤掉指定元素的特征（如Ce）
                # if filter_elements:
                #     meta_df = meta_df[~((meta_df['elem1'].isin(filter_elements)) | 
                #                        (meta_df['elem2'].isin(filter_elements)))]
                
                # 为每个特征创建名称
                feature_names = []
                feature_indices = {}  # 用于跟踪特征索引
                
                for idx, row in meta_df.iterrows():
                    # 获取元素名称，将金属元素替换为'M'
                    elem1 = row['elem1']
                    elem2 = row['elem2']
                    
                    # 如果元素1或元素2与当前金属元素相同，则替换为'M'
                    if elem1 == element:
                        elem1 = 'M'
                    if elem2 == element:
                        elem2 = 'M'
                    if centre_atom == element:
                        centre_atom_name = "M"
                    else:
                        centre_atom_name = centre_atom
                    
                    # 标准化元素顺序，确保Ce总是在其他元素之后
                    # 如果其中一个元素是Ce，确保它总是放在第二位
                    if elem1 == 'Ce' and elem2 != 'Ce':
                        elem1, elem2 = elem2, elem1
                        # 交换n1和n2的值，保持特征的一致性
                        n1, n2 = row['n2'], row['n1']
                    else:
                        n1, n2 = row['n1'], row['n2']
                    
                    # 创建特征名称，包含中心原子信息和M类型
                    feature_name = f"{m_type}_{centre_atom_name}_{elem1}_{elem2}_n1{n1}_n2{n2}_l{row['l']}"
                    
                    # 将特征名称添加到列表中
                    feature_names.append(feature_name)
                    
                    # 记录特征索引
                    feature_indices[idx] = len(feature_names) - 1
                
                # 加载特征数据
                features = np.load(feature_file)
                
                # 处理多个原子的情况（如多个H原子）
                if len(features.shape) > 1 and features.shape[0] > 1:
                    # 为每个原子创建单独的特征名称和特征值
                    atom_features = []
                    atom_feature_names = []
                    
                    # 遍历每个原子
                    for atom_idx in range(features.shape[0]):
                        # 为每个原子复制一份特征名称，添加原子索引后缀
                        atom_specific_names = [f"{name}_atom{atom_idx}" for name in feature_names]
                        atom_feature_names.extend(atom_specific_names)
                        
                        # 获取当前原子的特征值
                        atom_feature_values = features[atom_idx]
                        atom_features.append(atom_feature_values)
                    
                    # 更新特征名称列表和特征索引映射
                    feature_names = atom_feature_names
                    
                    # 将所有原子的特征值合并为一个数组
                    features = np.concatenate(atom_features)
                    
                    # 更新特征索引映射
                    feature_indices = {}
                    for i in range(len(feature_names)):
                        feature_indices[i] = i
                elif len(features.shape) > 1:
                    features = features[0]  # 只有一个原子
                
                # # 如果需要过滤特定元素
                # if filter_elements:
                #     # 获取要保留的特征索引
                #     keep_indices = meta_df.index.tolist()
                #     features = features[keep_indices]
                
                # 将特征添加到行数据中
                for idx, feature_value in enumerate(features):
                    if idx in feature_indices:
                        feature_idx = feature_indices[idx]
                        if feature_idx < len(feature_names):
                            column_name = feature_names[feature_idx]
                            row_data[column_name] = feature_value
            
    # 将字典转换为列表
    combined_data = list(element_data.values())
    
    # 转换为DataFrame
    return pd.DataFrame(combined_data)


def merge_with_adsorption_energy(
    features_df: pd.DataFrame, 
    energy_file: str
) -> pd.DataFrame:
    """
    将特征数据与氢吸附能数据合并
    
    参数:
        features_df: 特征DataFrame
        energy_file: 吸附能数据文件路径
        
    返回:
        合并后的DataFrame
    """
    # 加载吸附能数据
    energy_df = pd.read_csv(energy_file)
    
    # 创建结果DataFrame的副本
    result_df = features_df.copy()
    
    # 创建映射字典，将M_system映射到结构类型
    structure_mapping = {}
    for ov_type, position in [("Ov1", "Fir-1"), ("Ov1", "Sec-2"), ("Ov1", "Fir-2"), 
                           ("Ov1", "Sec-1"), ("Ov0", "None")]:
        key = f"{ov_type}/{position}"
        if position == "None":
            structure_mapping[key] = "pristine"
        else:
            structure_mapping[key] = DEFAULT_STRUCTURE_NAME.get(position, position)
    
    # 为每一行添加吸附能数据
    for i, row in result_df.iterrows():
        element = row['element']
        structure_type = row['structure_type']
        
        # 找到对应的M_system键
        m_system_key = None
        for key, value in structure_mapping.items():
            if value == structure_type:
                m_system_key = key
                break
        
        if m_system_key is None:
            continue
        
        # 查找该元素和结构的吸附能数据
        energy_row = energy_df[(energy_df['element'] == element) & 
                              (energy_df['M_system'] == m_system_key)]
        
        if not energy_row.empty:
            # 添加吸附能数据
            result_df.loc[i, 'H2_adsorption_energy'] = energy_row['H2_adsorption_energy'].values[0]
            
            # # 添加M系统和M2H系统能量
            # result_df.loc[i, 'M_energy'] = energy_row['M_energy'].values[0]
            # result_df.loc[i, 'M2H_energy'] = energy_row['M2H_energy'].values[0]
    
    return result_df


def merge_with_element_properties(
    features_df: pd.DataFrame, 
    properties_file: str
) -> pd.DataFrame:
    """
    将特征数据与元素性质数据合并
    
    参数:
        features_df: 特征DataFrame
        properties_file: 元素性质数据文件路径
        
    返回:
        合并后的DataFrame
    """
    # 加载元素性质数据
    properties_df = pd.read_csv(properties_file)
    
    # 基于元素列合并数据（每个元素-结构组合保留元素的基础性质）
    return pd.merge(features_df, properties_df, on='element', how='left')


def main(
    base_dir: str = "/Users/wukai/Desktop/project/w_git/data/fea/final/cont",
    energy_file: str = "/Users/wukai/Desktop/project/w_git/data/fea/final/h2_adsorption_energy.csv",
    properties_file: str = "/Users/wukai/Desktop/project/w_git/data/ele_propety.csv",
    output_file: str = "/Users/wukai/Desktop/project/w_git/data/fea/final/merged_dataset.csv",
    filter_elements: Optional[List[str]] = None
):
    """
    主函数：执行完整的数据合并流程
    
    参数:
        base_dir: SOAP特征文件的基础目录
        energy_file: 氢吸附能数据文件路径
        properties_file: 元素性质数据文件路径
        output_file: 输出文件路径
        filter_elements: 要过滤的元素列表
    """
    print(f"开始合并数据集...")
    
    # 查找所有SOAP特征文件
    print(f"查找SOAP特征文件...")
    soap_files = find_soap_files(base_dir)
    print(f"找到 {len(soap_files)} 种元素的SOAP特征")
    
    # 加载并合并特征
    print(f"加载并合并SOAP特征...")
    if filter_elements:
        print(f"过滤元素: {', '.join(filter_elements)}")
    
    features_df = load_and_combine_features(soap_files, filter_elements)
    print(f"合并后的特征数据形状: {features_df.shape}")
    
    # 合并吸附能数据
    print(f"合并氢吸附能数据...")
    merged_df = merge_with_adsorption_energy(features_df, energy_file)
    
    # 合并元素性质数据
    print(f"合并元素基础性质数据...")
    final_df = merge_with_element_properties(merged_df, properties_file)
    print(f"最终数据集形状: {final_df.shape}")
    
    # 保存结果
    print(f"保存合并后的数据集到 {output_file}")
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    final_df.to_csv(output_file, index=False)
    
    print(f"数据合并完成!")
    return final_df


if __name__ == "__main__":
    # 默认过滤掉Ce元素的特征
    main(filter_elements=["Ce"])
