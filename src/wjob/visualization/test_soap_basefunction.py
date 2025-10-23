#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SOAP基函数可视化测试脚本

运行各种SOAP基本组件的可视化示例。
"""

import os
import sys
import subprocess
import matplotlib.pyplot as plt
import numpy as np

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
base_script = os.path.join(script_dir, 'soap_basefunction.py')

def test_radial_functions():
    """测试径向基函数可视化"""
    print("测试径向基函数可视化...")
    
    # GTO基函数示例
    cmd = f"python {base_script} radial --type gto --n-max 4 --r-cut 6.0 --alpha 0.5"
    print(f"运行: {cmd}")
    subprocess.run(cmd, shell=True)
    
    # 多项式基函数示例
    cmd = f"python {base_script} radial --type poly --n-max 4 --r-cut 6.0"
    print(f"运行: {cmd}")
    subprocess.run(cmd, shell=True)
    
def test_angular_functions():
    """测试球谐函数可视化"""
    print("测试球谐函数可视化...")
    
    # 几个不同的l和m组合
    l_m_pairs = [(1, 0), (2, 0), (2, 1), (3, 2)]
    
    for l, m in l_m_pairs:
        cmd = f"python {base_script} angular -l {l} -m {m}"
        print(f"运行: {cmd}")
        subprocess.run(cmd, shell=True)

def test_structure_visualization(structure_file):
    """测试原子结构可视化
    
    Args:
        structure_file: 原子结构文件路径
    """
    if not os.path.exists(structure_file):
        print(f"错误: 结构文件 '{structure_file}' 不存在")
        return
    
    print("测试原子结构可视化...")
    
    # 没有中心原子的可视化
    cmd = f"python {base_script} atoms {structure_file}"
    print(f"运行: {cmd}")
    subprocess.run(cmd, shell=True)
    
    # 有中心原子的可视化（假设0是一个有效的索引）
    cmd = f"python {base_script} atoms {structure_file} --center 0"
    print(f"运行: {cmd}")
    subprocess.run(cmd, shell=True)

def test_density_visualization(structure_file):
    """测试密度分布可视化
    
    Args:
        structure_file: 原子结构文件路径
    """
    if not os.path.exists(structure_file):
        print(f"错误: 结构文件 '{structure_file}' 不存在")
        return
    
    print("测试原子密度可视化...")
    
    # 假设0是一个有效的中心原子索引
    cmd = f"python {base_script} density {structure_file} 0 --radius 4.0 --grid 40"
    print(f"运行: {cmd}")
    subprocess.run(cmd, shell=True)

def main():
    """主函数"""
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python test_soap_basefunction.py <结构文件路径> [测试模式]")
        print("测试模式: all, radial, angular, atoms, density (默认: all)")
        return
    
    structure_file = sys.argv[1]
    mode = "all" if len(sys.argv) < 3 else sys.argv[2]
    
    if mode in ["all", "radial"]:
        test_radial_functions()
        
    if mode in ["all", "angular"]:
        test_angular_functions()
        
    if mode in ["all", "atoms"]:
        test_structure_visualization(structure_file)
        
    if mode in ["all", "density"]:
        test_density_visualization(structure_file)

if __name__ == "__main__":
    main()
