#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据预处理模块

实现特征归一化、删除空值和全为0的特征等操作。
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def remove_constant_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    删除全为0或其他常数的特征
    
    参数:
        df: 输入数据集
        
    返回:
        删除常数特征后的数据集
    """
    # 保存非特征列（如element、structure_type等）
    non_feature_cols = ['element', 'structure_type', 'H2_adsorption_energy', 'M_energy', 'M2H_energy']
    non_feature_cols.extend([col for col in df.columns if col in [
        'Electronegativity', 'First_Ionization_Energy', 'Second_Ionization_Energy',
        'Atomic_Radius', 'Covalent_Radius', 'Period', 'Group', 'Valence_Electrons', 'Electron_Shells'
    ]])
    
    # 获取特征列
    feature_cols = [col for col in df.columns if col not in non_feature_cols]
    
    # 计算每个特征的标准差
    std_series = df[feature_cols].std()
    
    # 找出标准差为0的列（常数列）
    constant_cols = std_series[std_series == 0].index.tolist()
    print(f"删除了 {len(constant_cols)} 个常数特征")
    
    # 删除常数列
    df_filtered = df.drop(columns=constant_cols)
    
    return df_filtered


def remove_nan_features(df: pd.DataFrame, threshold: float = 0.1) -> pd.DataFrame:
    """
    删除空值过多的特征
    
    参数:
        df: 输入数据集
        threshold: 空值比例阈值，超过该比例的列将被删除
        
    返回:
        删除空值过多的特征后的数据集
    """
    # 计算每列的空值比例
    nan_ratio = df.isna().mean()
    
    # 找出空值比例超过阈值的列
    cols_to_drop = nan_ratio[nan_ratio > threshold].index.tolist()
    print(f"删除了 {len(cols_to_drop)} 个空值过多的特征")
    
    # 删除这些列
    df_filtered = df.drop(columns=cols_to_drop)
    
    # 对剩下的空值进行填充
    # 区分数值列和非数值列
    numeric_cols = df_filtered.select_dtypes(include=['number']).columns
    
    # 对数值列使用均值填充
    if not numeric_cols.empty:
        df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(df_filtered[numeric_cols].mean())
    
    # 对非数值列使用众数填充
    non_numeric_cols = df_filtered.select_dtypes(exclude=['number']).columns
    for col in non_numeric_cols:
        if df_filtered[col].isna().any():
            most_frequent = df_filtered[col].mode()[0]
            df_filtered[col] = df_filtered[col].fillna(most_frequent)
    
    return df_filtered


def normalize_features(df: pd.DataFrame, method: str = 'standard') -> pd.DataFrame:
    """
    对特征进行归一化
    
    参数:
        df: 输入数据集
        method: 归一化方法，'standard'表示标准化，'minmax'表示最小-最大归一化
        
    返回:
        归一化后的数据集
    """
    # 保存非特征列（如element、structure_type等）和目标变量
    non_feature_cols = ['element', 'structure_type']
    target_cols = ['H2_adsorption_energy']  # 预测目标值，不进行归一化
    
    # 获取数值特征列，只对数值列进行归一化
    numeric_cols = df.select_dtypes(include=['number']).columns
    feature_cols = [col for col in numeric_cols if col not in non_feature_cols and col not in target_cols]
    
    print(f"将对 {len(feature_cols)} 个数值特征进行归一化")
    print(f"以下列不会被归一化: {', '.join(non_feature_cols + target_cols)}")
    
    # 创建归一化器
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"不支持的归一化方法: {method}")
    
    # 对数值特征进行归一化
    df_normalized = df.copy()
    if feature_cols:
        df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    return df_normalized


def preprocess_dataset(input_file: str, output_file: str = None, method: str = 'standard') -> pd.DataFrame:
    """
    对数据集进行完整的预处理流程
    
    参数:
        input_file: 输入数据集文件路径
        output_file: 输出数据集文件路径，如果为None则不保存
        method: 归一化方法
        
    返回:
        预处理后的数据集
    """
    print(f"加载数据集: {input_file}")
    df = pd.read_csv(input_file)
    print(f"原始数据集大小: {df.shape}")
    
    # 删除常数特征
    df = remove_constant_features(df)
    print(f"删除常数特征后的数据集大小: {df.shape}")
    
    # 删除空值过多的特征
    df = remove_nan_features(df)
    print(f"删除空值过多的特征后的数据集大小: {df.shape}")
    
    # 归一化特征
    df = normalize_features(df, method)
    print(f"归一化后的数据集大小: {df.shape}")
    
    # 保存处理后的数据集
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"已保存预处理后的数据集到: {output_file}")
    
    return df


def main():
    """主函数"""
    input_file = "/Users/wukai/Desktop/project/w_git/data/fea/final/merged_dataset.csv"
    output_file = "/Users/wukai/Desktop/project/w_git/data/fea/final/normalized_dataset.csv"
    
    # 默认使用标准化归一化
    preprocess_dataset(input_file, output_file, method='standard')


if __name__ == "__main__":
    main()