#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
SOAP基函数可视化

该脚本用于可视化SOAP特征构造过程中的各个基本组件，包括：
1. 原子坐标与邻域定义
2. 局部原子密度分布
3. 径向基函数
4. 球谐函数（角向基函数）
5. SOAP特征构造流程

作者: 吴凯 (Kai Wu)
日期: 2025-06-19
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
from scipy.special import sph_harm
import argparse
import os
import sys

# 尝试导入用于3D可视化的库
try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    print("警告: PyVista未安装，3D可视化功能将不可用。")
    print("可通过'pip install pyvista'安装。")

# 尝试导入ASE用于原子结构处理
try:
    import ase
    import ase.io
    HAS_ASE = True
except ImportError:
    HAS_ASE = False
    print("警告: ASE未安装，原子结构相关功能将不可用。")
    print("可通过'pip install ase'安装。")


#------------------------------------------------------------------------------
# SOAP基函数定义
#------------------------------------------------------------------------------

def gto_radial_basis(r, n, alpha):
    """高斯型轨道(GTO)径向基函数。
    
    参数:
        r (numpy.ndarray): 半径点
        n (int): 主量子数
        alpha (float): 高斯函数的导出参数
        
    返回:
        numpy.ndarray: 径向基函数值
    """
    # 根据 SOAP 原始公式，GTO 形式为 r^n * exp(-alpha*r^2)
    return r**n * np.exp(-alpha * r**2)


def poly_radial_basis(r, n, r_cut):
    """多项式径向基函数。
    
    参数:
        r (numpy.ndarray): 半径点
        n (int): 主量子数
        r_cut (float): 截断半径
        
    返回:
        numpy.ndarray: 径向基函数值
    """
    # 基于常用的多项式形式定义
    x = r/r_cut
    # 当r超过r_cut时，函数为0
    mask = r > r_cut
    result = (1 - x)**n * np.exp(-(r/(r_cut/2))**2)
    result[mask] = 0.0
    return result


def visualize_radial_basis(basis_type='gto', n_max=3, r_cut=5.0, alpha=1.0, figsize=(10, 6)):
    """可视化径向基函数。
    
    参数:
        basis_type (str): 基函数类型，'gto'或'poly'
        n_max (int): 要可视化的最大主量子数
        r_cut (float): 截断半径
        alpha (float): GTO基函数的alpha参数
        figsize (tuple): 图形大小
    """
    r = np.linspace(0, r_cut*1.2, 1000)
    
    plt.figure(figsize=figsize)
    
    for n in range(1, n_max+1):
        if basis_type.lower() == 'gto':
            g_n = gto_radial_basis(r, n, alpha)
            plt.plot(r, g_n, label=f"GTO: n={n}, $\\alpha$={alpha}")
        elif basis_type.lower() == 'poly':
            g_n = poly_radial_basis(r, n, r_cut)
            plt.plot(r, g_n, label=f"多项式: n={n}, $r_{{cut}}$={r_cut}")
    
    # 添加竖线标记截断半径
    plt.axvline(x=r_cut, color='gray', linestyle='--', alpha=0.7)
    plt.text(r_cut*1.02, 0.5, f"$r_{{cut}}$={r_cut}", rotation=90, verticalalignment='center')
    
    plt.title(f"SOAP径向基函数 ({basis_type.upper()})")
    plt.xlabel("半径 (\u00c5)")
    plt.ylabel("基函数值")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    return plt


#------------------------------------------------------------------------------
# 球谐函数（角向基函数）
#------------------------------------------------------------------------------

def visualize_spherical_harmonic(l, m, cmap='RdBu', figsize=(10, 10)):
    """可视化特定角量子数l和磁量子数m的球谐函数。
    
    参数:
        l (int): 角量子数
        m (int): 磁量子数，范围为-l到l
        cmap (str): matplotlib颜色映射
        figsize (tuple): 图形大小
    """
    # 创建三维网格
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2*np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    
    # 计算球谐函数
    Y = sph_harm(m, l, theta, phi)
    Y_real = Y.real
    Y_abs = np.abs(Y)
    
    # 根据球谐函数值调整半径
    # 使用实部值为颜色，绝对值为半径
    r = 0.5 + 0.2 * (Y_abs / np.max(Y_abs)) # 归一化的半径
    
    # 转换为笛卡尔坐标
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    # 规范化实部值为颜色映射
    Y_norm = (Y_real - Y_real.min()) / (Y_real.max() - Y_real.min())
    
    # 创建图形
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制表面
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.get_cmap(cmap)(Y_norm), alpha=0.9, linewidth=0, antialiased=True)
    
    # 添加颜色条
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(Y_real)
    plt.colorbar(m, ax=ax, shrink=0.6, aspect=10, label='球谐函数实部值')
    
    # 设置图形属性
    ax.set_title(f"$Y_{{{l}}}^{{{m}}}(\\theta, \\phi)$")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # 设置等比例轴以确保张量形状正确
    ax.set_box_aspect([1,1,1])
    
    # 添加坐标轴原点说明
    ax.text(0, 0, -0.6, "(0,0,0)", color='black', fontsize=10)
    
    return plt


#------------------------------------------------------------------------------
# 密度分布与原子可视化
#------------------------------------------------------------------------------

def visualize_atoms(structure_file, center_atom_idx=None, radius=5.0):
    """可视化原子结构和可能的邻域。
    
    参数:
        structure_file (str): 原子结构文件路径（支持ASE能读取的格式）
        center_atom_idx (int): 中心原子索引
        radius (float): 邻域半径
    """
    if not HAS_ASE:
        print("需要ASE库来可视化原子结构")
        return None
    
    if not HAS_PYVISTA:
        print("需要PyVista库来可视化3D结构")
        return None
    
    # 加载结构
    atoms = ase.io.read(structure_file)
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # 设置颜色映射
    element_colors = {
        'H': [1, 1, 1],  # 白色
        'O': [1, 0, 0],  # 红色
        'Ce': [0.8, 0.8, 0.2],  # 金色
        'Ag': [0.75, 0.75, 0.75],  # 银色
        'Pt': [0.7, 0.7, 0.8],  # 铂色
    }
    # 为其他元素设置默认颜色
    default_color = [0.5, 0.5, 0.5]  # 灰色
    
    # 创建PyVista绘图对象
    p = pv.Plotter()
    
    # 绘制所有原子
    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        # 根据原子编号调整大小（氢小一些，其他大一些）
        size = 0.3 if sym == 'H' else 0.6
        
        # 设置颜色
        color = element_colors.get(sym, default_color)
        
        # 为中心原子设置特殊样式
        if center_atom_idx is not None and i == center_atom_idx:
            # 中心原子神异一些，加高透明度
            sphere = pv.Sphere(radius=size, center=pos)
            p.add_mesh(sphere, color=color, opacity=0.8)
            
            # 添加半径球来显示SOAP的拓扑范围
            boundary = pv.Sphere(radius=radius, center=pos)
            p.add_mesh(boundary, style='wireframe', color='cyan', opacity=0.5)
            
            # 添加文本注释
            p.add_point_labels([pos], [f"{sym}_{i} (center)"], font_size=12)
        else:
            # 其他原子
            sphere = pv.Sphere(radius=size, center=pos)
            p.add_mesh(sphere, color=color)
    
    # 添加坐标轴
    p.add_axes()
    p.add_bounding_box()
    
    # 设置视角
    if center_atom_idx is not None:
        # 将焦点设置为中心原子
        p.camera_position = 'xy'
        p.camera.focal_point = positions[center_atom_idx]
    
    return p


def visualize_atom_density(structure_file, center_atom_idx, radius=5.0, grid_size=30):
    """可视化局部原子密度分布。
    
    参数:
        structure_file (str): 原子结构文件路径
        center_atom_idx (int): 中心原子索引
        radius (float): 邻域半径
        grid_size (int): 网格大小
    """
    if not HAS_ASE or not HAS_PYVISTA:
        print("需要ASE和PyVista库来可视化密度分布")
        return None
    
    # 加载原子结构
    atoms = ase.io.read(structure_file)
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    
    # 获取中心原子信息
    center_pos = positions[center_atom_idx]
    center_sym = symbols[center_atom_idx]
    
    # 创建空间网格
    x = np.linspace(center_pos[0]-radius, center_pos[0]+radius, grid_size)
    y = np.linspace(center_pos[1]-radius, center_pos[1]+radius, grid_size)
    z = np.linspace(center_pos[2]-radius, center_pos[2]+radius, grid_size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # 初始化密度场
    density = np.zeros((grid_size, grid_size, grid_size))
    
    # 对于列表中的每个原子，计算高斯分布
    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        # 计算距离中心原子的距离
        dist = np.sqrt(np.sum((pos - center_pos)**2))
        
        # 只考虑半径内的原子
        if dist <= radius:
            # 计算每个网格点到当前原子的距离
            dx = X - pos[0]
            dy = Y - pos[1]
            dz = Z - pos[2]
            r_squared = dx**2 + dy**2 + dz**2
            
            # 高斯分布形式的密度贡献
            # 使用较小的sigma对氢原子，较大的对其他原子
            sigma = 0.3 if sym == 'H' else 0.6
            # 直接使用对应元素的原子序数作为密度幅度
            amp = 1.0 if sym == 'H' else 6.0 if sym == 'O' else 58.0 if sym == 'Ce' else 1.0
            
            # 高斯函数形式的密度叠加
            density += amp * np.exp(-r_squared / (2 * sigma**2))
    
    # 创建 PyVista 对象这里的 -grid_size/2 是为了将网格中心于移动到中心原子位置
    grid = pv.ImageData(
        dimensions=(grid_size, grid_size, grid_size),
        spacing=(x[1]-x[0], y[1]-y[0], z[1]-z[0]),
        origin=(x[0], y[0], z[0])
    )
    
    # 添加密度场数据
    grid.point_data["density"] = density.flatten()
    
    # 创建绘图对象
    p = pv.Plotter()
    
    # 添加等值面（使用几个不同的等密度水平）
    contours = grid.contour([0.5, 1.0, 2.0])
    p.add_mesh(contours, opacity=0.5, cmap='viridis')
    
    # 添加原子结构
    for i, (pos, sym) in enumerate(zip(positions, symbols)):
        # 计算距离中心原子的距离
        dist = np.sqrt(np.sum((pos - center_pos)**2))
        
        # 只显示半径内的原子
        if dist <= radius:
            size = 0.2 if sym == 'H' else 0.4
            
            # 设置颜色
            if i == center_atom_idx:
                color = 'yellow'  # 突出中心原子
                sphere = pv.Sphere(radius=size*1.2, center=pos)
                p.add_mesh(sphere, color=color, opacity=1.0)
            else:
                if sym == 'H':
                    color = 'white'
                elif sym == 'O':
                    color = 'red'
                elif sym == 'Ce':
                    color = 'gold'
                else:
                    color = 'gray'
                    
                sphere = pv.Sphere(radius=size, center=pos)
                p.add_mesh(sphere, color=color)
    
    # 显示半径范围
    boundary = pv.Sphere(radius=radius, center=center_pos)
    p.add_mesh(boundary, style='wireframe', color='cyan', opacity=0.3)
    
    # 添加坐标轴和边界框
    p.add_axes()
    p.add_bounding_box()
    
    # 设置标题
    p.add_text(f"Center: {center_sym}_{center_atom_idx}, Radius: {radius} Å", font_size=12)
    
    return p


#------------------------------------------------------------------------------
# 主程序
#------------------------------------------------------------------------------

def main():
    """主函数入口
    """
    parser = argparse.ArgumentParser(description='SOAP基函数可视化工具')
    subparsers = parser.add_subparsers(dest='mode', help='可视化模式')
    
    # 径向基函数可视化
    radial_parser = subparsers.add_parser('radial', help='径向基函数可视化')
    radial_parser.add_argument('--type', choices=['gto', 'poly'], default='gto', help='基函数类型')
    radial_parser.add_argument('--n-max', type=int, default=3, help='最大主量子数')
    radial_parser.add_argument('--r-cut', type=float, default=5.0, help='截断半径')
    radial_parser.add_argument('--alpha', type=float, default=1.0, help='GTO基函数的alpha参数')
    
    # 角向基函数（球谐函数）可视化
    angular_parser = subparsers.add_parser('angular', help='球谐函数可视化')
    angular_parser.add_argument('-l', type=int, default=2, help='角量子数')
    angular_parser.add_argument('-m', type=int, default=0, help='磁量子数')
    angular_parser.add_argument('--cmap', default='RdBu', help='颜色映射')
    
    # 原子可视化
    atoms_parser = subparsers.add_parser('atoms', help='原子结构可视化')
    atoms_parser.add_argument('structure_file', help='原子结构文件路径')
    atoms_parser.add_argument('--center', type=int, help='中心原子索引')
    atoms_parser.add_argument('--radius', type=float, default=5.0, help='邻域半径')
    
    # 原子密度可视化
    density_parser = subparsers.add_parser('density', help='原子密度可视化')
    density_parser.add_argument('structure_file', help='原子结构文件路径')
    density_parser.add_argument('center', type=int, help='中心原子索引')
    density_parser.add_argument('--radius', type=float, default=5.0, help='邻域半径')
    density_parser.add_argument('--grid', type=int, default=30, help='网格大小')
    
    args = parser.parse_args()
    
    if args.mode == 'radial':
        # 绘制径向基函数
        plt = visualize_radial_basis(args.type, args.n_max, args.r_cut, args.alpha)
        plt.show()
        
    elif args.mode == 'angular':
        # 检查m值是否有效
        if abs(args.m) > args.l:
            print(f"错误: |m| 必须小于或等于 l. 当前l={args.l}, 但m={args.m}")
            sys.exit(1)
        
        # 绘制球谐函数
        plt = visualize_spherical_harmonic(args.l, args.m, args.cmap)
        plt.show()
        
    elif args.mode == 'atoms':
        # 可视化原子结构
        plotter = visualize_atoms(args.structure_file, args.center, args.radius)
        if plotter:
            plotter.show()
        
    elif args.mode == 'density':
        # 可视化密度分布
        plotter = visualize_atom_density(args.structure_file, args.center, args.radius, args.grid)
        if plotter:
            plotter.show()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
