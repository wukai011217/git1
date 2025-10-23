# -*- coding: utf-8 -*-
"""
SOAP特征可视化模块

提供各种SOAP特征的可视化功能，包括:
1. 基本特征柱状图/热图
2. 角向分量球面分布
3. 完整的三维密度重建

使用方法:
    python -m wjob.visualization.soap_fea <结构文件> <元数据CSV> <特征CSV> <特征索引>
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path
from scipy.special import sph_harm

from ase.io import read
from dscribe.descriptors import SOAP

from wjob.config import SOAP_PARAMS


def visualize_angular_distribution(l, m=0, title=None, ax=None, cmap='seismic'):
    """可视化角量子数为l的球谐函数分布。
    
    参数:
        l (int): 角量子数
        m (int, optional): 磁量子数，默认为0
        title (str, optional): 图表标题
        ax (matplotlib.Axes, optional): 绘图轴对象
        cmap (str, optional): 颜色映射名称
    
    返回:
        matplotlib.Axes: 绘图轴对象
    """
    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
    
    # 生成球面网格
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta, phi = np.meshgrid(theta, phi)
    
    # 计算球谐函数
    Y = sph_harm(m, l, phi, theta).real
    
    # 归一化并创建径向变化
    Y_norm = (Y - Y.min()) / (Y.max() - Y.min())
    R = 1 + Y_norm * 0.3  # 半径调整使形状更明显
    
    # 转换为笛卡尔坐标
    X = R * np.sin(theta) * np.cos(phi)
    Y_cart = R * np.sin(theta) * np.sin(phi)
    Z = R * np.cos(theta)
    
    # 绘制3D表面
    color_values = cm.get_cmap(cmap)((Y - Y.min()) / (Y.max() - Y.min()))
    surf = ax.plot_surface(X, Y_cart, Z, facecolors=color_values,
                          rstride=2, cstride=2, linewidth=0.1, antialiased=True)
    
    if title is not None:
        ax.set_title(title)
    ax.set_axis_off()
    
    return ax


def reconstruct_soap_density(structure_file, meta_df, features, feature_idx, 
                             grid_size=50, isosurface_level=0.5):
    """重建指定的SOAP特征对应的三维密度分布。
    
    参数:
        structure_file (str): 原子结构文件路径
        meta_df (pd.DataFrame): 特征元数据DataFrame
        features (np.ndarray): 特征向量
        feature_idx (int): 要可视化的特征索引
        grid_size (int): 3D网格尺寸
        isosurface_level (float): 密度等值面绘制的百分比水平
    
    返回:
        tuple: (points, density, meta) 网格点, 密度值, 和特征元数据
    """
    try:
        import pyvista as pv
    except ImportError:
        print("请安装pyvista: pip install pyvista")
        raise
    
    # 获取特征元数据
    meta_row = meta_df.iloc[feature_idx]
    elem1, elem2 = meta_row['elem1'], meta_row['elem2']
    n1, n2 = int(meta_row['n1']), int(meta_row['n2'])
    l = int(meta_row['l'])
    print(f"可视化特征: ({elem1}, {elem2}, n1={n1}, n2={n2}, l={l})")
    
    # 读取结构
    atoms = read(structure_file)
    
    # 创建SOAP描述符
    unique_species = sorted(set(atoms.get_atomic_numbers()))
    soap = SOAP(species=unique_species, **SOAP_PARAMS)
    
    # 获取径向基函数
    if soap._rbf == "gto":
        # 对于GTO基函数需要特殊处理
        alphas, betas = soap._alphas, soap._betas
        print(f"使用GTO径向基函数 (需要展开α和β参数)")
        
        # 建立径向基函数中简单的近似
        def g_n(n, r):
            return np.exp(-(r**2) / (2*soap._sigma**2)) * (r**n)
    
    elif soap._rbf == "polynomial":
        # 获取多项式基函数
        rx, gss = soap.get_basis_poly(soap._r_cut, soap._n_max)
        g_n1 = np.polynomial.polynomial.Polynomial(gss[n1])
        g_n2 = np.polynomial.polynomial.Polynomial(gss[n2])
        
        print(f"使用多项式径向基函数")
        
        def g_n(n, r):
            if n == n1:
                return g_n1(r)
            elif n == n2:
                return g_n2(r)
            return np.zeros_like(r)
    
    # 设置3D网格
    r_cut = soap._r_cut
    grid = np.linspace(-r_cut, r_cut, grid_size)
    X, Y, Z = np.meshgrid(grid, grid, grid, indexing='ij')
    
    # 计算球坐标
    R = np.sqrt(X**2 + Y**2 + Z**2)
    THETA = np.arccos(np.clip(Z/(R + 1e-10), -1.0, 1.0))  # 避免除0错误
    PHI = np.arctan2(Y, X)
    
     # 计算l角量子数的所有m值的球谐函数
    # 为了计算SOAP特征的完整表示，需要加入所有m值的球谐函数贡献
    Y_l_all = np.zeros_like(PHI)
    for m in range(-l, l+1):
        # 对于每个m，计算并累加球谐函数的幅度平方(这在SOAP中被称为power spectrum)
        # SOAP特征是所有m值的power spectrum的组合
        Y_lm = sph_harm(m, l, PHI, THETA)
        Y_l_all += np.abs(Y_lm) ** 2
    
    # 归一化结果
    Y_l = np.sqrt(Y_l_all)
    
    # 构建径向函数与角向函数的乘积
    g1 = np.zeros_like(R)
    g2 = np.zeros_like(R)
    
    # 仅在半径范围内计算(避免过大计算量)
    mask = R <= r_cut
    g1[mask] = g_n(n1, R[mask])
    g2[mask] = g_n(n2, R[mask])
    
    # 将径向和角向部分组合成完整的密度
    density = g1 * Y_l * g2 * Y_l
    density[~mask] = 0.0  # 截断半径之外的值设为0
    
    # 将数据转换为pyvista格式
    points = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T
    grid = pv.StructuredGrid(X, Y, Z)
    grid["density"] = density.flatten()
    
    # 设置合理的等值面水平为最大值的百分比
    density_max = np.max(np.abs(density))
    iso_val = density_max * isosurface_level
    
    # 创建等值面
    contour = grid.contour([iso_val])
    
    # 创建一个pyvista窗口进行显示
    p = pv.Plotter()
    p.add_mesh(contour, opacity=0.7, color='lightblue')
    
    # 添加坐标轴
    p.show_axes()
    
    # 添加原点参考
    sphere = pv.Sphere(radius=0.1, center=(0,0,0))
    p.add_mesh(sphere, color='red')
    
    # 添加标题
    title = f"SOAP Feature: ({elem1}-{elem2}, n1={n1}, n2={n2}, l={l})"
    p.add_text(title, position='upper_edge', font_size=14)
    
    print("渲染3D密度等值面...")
    p.show()
    
    # 同时返回数据以便进一步处理
    meta_info = {
        'elem1': elem1, 'elem2': elem2,
        'n1': n1, 'n2': n2, 'l': l,
        'feature_value': features[0, feature_idx] if features.shape[0] > 0 else None
    }
    
    return points, density, meta_info


def visualize_soap_with_structure(structure_file, meta_df, features, feature_idx, grid_size=50, iso_level=0.5, atom_index=None):
    """将SOAP特征可视化与原子结构结合展示
    
    参数:
        structure_file: CONTCAR文件路径
        meta_df: 特征元数据DataFrame
        features: 特征值数组
        feature_idx: 要可视化的特征索引
        grid_size: 网格大小
        iso_level: 等值面相对阈值
        atom_index: 指定聚焦的中心原子索引（可选）
    """
    from ase.io import read
    from wjob.config import SOAP_PARAMS
    import numpy as np
    import pyvista as pv
    
    # 加载结构
    atoms = read(structure_file)
    
    # 获取特征信息
    feature_info = meta_df.iloc[feature_idx]
    elem1 = feature_info['elem1']
    elem2 = feature_info['elem2']
    n1 = feature_info['n1']
    n2 = feature_info['n2']
    l = feature_info['l']
    
    print(f"可视化特征与结构: ({elem1}, {elem2}, n1={n1}, n2={n2}, l={l})")
    
    # 获取原子的密度场数据
    points, density, meta_info = reconstruct_soap_density(structure_file, meta_df, features, feature_idx, 
                                                      grid_size, iso_level)
    
    # 更新网格大小以匹配密度数组尺寸
    adjusted_size = int(np.cbrt(density.size))
    print(f"网格尺寸: {adjusted_size}, 密度数组大小: {density.size}")
    
    # 创建PyVista场景
    p = pv.Plotter()
    
    # 添加原子结构
    atom_radii = {"Ag": 1.65, "O": 0.6, "H": 0.25, "Ce": 1.8, "C": 0.7}  # 原子半径（埃）
    atom_colors = {"Ag": "silver", "O": "red", "H": "white", "Ce": "blue", "C": "black"}
    
    # 确定中心原子位置（如果指定）
    center_pos = None
    if atom_index is not None and 0 <= atom_index < len(atoms):
        center_pos = atoms[atom_index].position
        center_atom = atoms[atom_index]
        print(f"聚焦的中心原子: {center_atom.symbol}_{atom_index} 位置: {center_pos}")
    
    # 添加所有原子
    for i, atom in enumerate(atoms):
        symbol = atom.symbol
        pos = atom.position
        radius = atom_radii.get(symbol, 1.0)
        color = atom_colors.get(symbol, "gray")
        
        # 确定此原子是否在关注区内
        in_focus = True
        if center_pos is not None:
            dist = np.linalg.norm(pos - center_pos)
            in_focus = dist <= SOAP_PARAMS["r_cut"]
        
        # 如果原子类型与特征相关或在关注区内，突出显示
        if (symbol == elem1 or symbol == elem2) and in_focus:
            # 增大这些原子的显示半径并添加标签
            sphere = pv.Sphere(radius=radius*1.2, center=pos)
            p.add_mesh(sphere, color=color, opacity=0.8)
            p.add_point_labels([pos], [f"{symbol}_{i}"], point_size=0, font_size=10)
        elif in_focus:
            # 普通显示其他在关注区内的原子
            sphere = pv.Sphere(radius=radius*0.9, center=pos)
            p.add_mesh(sphere, color=color, opacity=0.6)
        else:
            # 淡化显示区域外的原子
            sphere = pv.Sphere(radius=radius*0.7, center=pos)
            p.add_mesh(sphere, color=color, opacity=0.3)
    
    # 根据密度场数据创建结构化网格，使其中心与中心原子对齐
    if center_pos is not None:
        # 如果有中心原子，将网格中心与其对齐
        grid_center = center_pos
    else:
        # 如果没有指定中心原子，使用默认中心（原点）
        grid_center = np.array([0, 0, 0])
    
    # 计算网格原点（左下角），使中心点位于网格中央
    grid_size_real = adjusted_size * 1.0  # 使用较小的网格间距，增加分辨率
    grid_origin = grid_center - grid_size_real/2
    
    grid = pv.ImageData(
        dimensions=(adjusted_size, adjusted_size, adjusted_size),
        origin=(grid_origin[0], grid_origin[1], grid_origin[2]),
        spacing=(grid_size_real/adjusted_size, grid_size_real/adjusted_size, grid_size_real/adjusted_size)
    )
    # 确保密度数组大小与网格点数匹配
    if density.size == adjusted_size**3:
        grid.point_data["density"] = density.flatten()
    else:
        print(f"尝试调整密度数组尺寸以匹配网格点数: {grid.n_points}")
        # 重新采样或调整密度数据以适应网格点数
        reshaped = density.reshape(adjusted_size, adjusted_size, adjusted_size)
        grid.point_data["density"] = reshaped.flatten()
    
    # 添加等值面，使用半透明效果
    density_max = np.max(density)
    iso_value = iso_level * density_max if density_max > 0 else 0.01
    iso_surface = grid.contour([iso_value])
    p.add_mesh(iso_surface, opacity=0.6, color='lightblue')
    
    # 添加SOAP参数标注
    r_cut_text = f"SOAP r_cut: {SOAP_PARAMS['r_cut']} Å"
    feature_text = f"SOAP特征: ({elem1}, {elem2}, n1={n1}, n2={n2}, l={l})"
    p.add_text(feature_text + "\n" + r_cut_text, position='upper_edge', font_size=12)
    
    # 如果指定了中心原子，添加截断球
    if center_pos is not None:
        # 添加截断球体表示SOAP切断半径
        cutoff_sphere = pv.Sphere(radius=SOAP_PARAMS["r_cut"], center=center_pos)
        p.add_mesh(cutoff_sphere, opacity=0.1, color='gray', style='wireframe')
        
        # 添加中心原子到指定元素原子的连线，显示方向性
        center_atom = atoms[atom_index]
        center_symbol = center_atom.symbol
        
        # 找到特征中涉及的两个元素
        target_elem = elem1 if center_symbol == elem2 else elem2
        
        # 添加从中心原子到目标元素原子的连线
        vectors = []
        for i, atom in enumerate(atoms):
            if atom.symbol == target_elem:
                dist = np.linalg.norm(atom.position - center_pos)
                if dist <= SOAP_PARAMS["r_cut"] and dist > 0.1:  # 排除非常近的原子
                    # 创建线段
                    line = pv.Line(center_pos, atom.position)
                    # 连线颜色和宽度可调整
                    line_opacity = 0.8 if dist < SOAP_PARAMS["r_cut"]/2 else 0.4
                    p.add_mesh(line, color='gold', line_width=6, opacity=line_opacity)
                    
                    # 添加距离标注
                    mid_point = (center_pos + atom.position) / 2
                    dist_text = f"{dist:.2f}Å"
                    p.add_point_labels([mid_point], [dist_text], point_size=0, font_size=8)
                    
                    # 记录方向向量用于后续分析
                    vec = atom.position - center_pos
                    vec_norm = vec / np.linalg.norm(vec)
                    vectors.append(vec_norm)
        
        # 如果有多个方向向量，计算平均方向
        if vectors:
            avg_vec = np.mean(vectors, axis=0)
            avg_vec = avg_vec / np.linalg.norm(avg_vec)
            
            # 添加平均方向向量的线段（更粗显示）
            end_point = center_pos + avg_vec * (SOAP_PARAMS["r_cut"] * 0.8)
            avg_line = pv.Line(center_pos, end_point)
            p.add_mesh(avg_line, color='magenta', line_width=5, opacity=0.9)
            p.add_point_labels([end_point], ["主方向"], point_size=0, font_size=12)
        
        # 添加指示特征方向的文本说明
        msg = f"方向性: {center_symbol}→{target_elem} (l={l})"
        p.add_text(msg, position='upper_left', font_size=12, color='white')
    
    # 添加坐标轴
    p.add_axes()
    p.show_bounds()
    
    # 显示图形
    print("渲染原子结构与SOAP特征等值面...")
    p.show()
    
    return


def visualize_feature_histogram(features, meta_df, feature_idx=None, n_bins=30):
    """可视化SOAP特征的直方图分布。
    
    参数:
        features (np.ndarray): 特征数据
        meta_df (pd.DataFrame): 特征元数据
        feature_idx (int, optional): 指定的特征索引
        n_bins (int): 直方图的箱数
    """
    if feature_idx is None:
        # 如果没有指定特征，则显示所有特征的平均值分布
        plt.figure(figsize=(12, 6))
        values = features.mean(axis=0)
        plt.bar(range(len(values)), values)
        plt.xlabel('特征索引')
        plt.ylabel('平均特征值')
        plt.title('所有SOAP特征的平均值分布')
    else:
        # 显示指定特征的直方图
        plt.figure(figsize=(8, 6))
        meta_row = meta_df.iloc[feature_idx]
        title = f"特征 ({meta_row['elem1']}, {meta_row['elem2']}, n1={meta_row['n1']}, n2={meta_row['n2']}, l={meta_row['l']})"
        
        values = features[:, feature_idx]
        plt.hist(values, bins=n_bins)
        plt.xlabel('特征值')
        plt.ylabel('计数')
        plt.title(title)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='SOAP特征可视化工具')
    parser.add_argument('structure_file', help='原子结构文件路径 (CONTCAR或其他ASE支持的格式)')
    parser.add_argument('meta_file', help='特征元数据CSV文件')
    parser.add_argument('feature_file', help='特征CSV文件')
    parser.add_argument('feature_idx', type=int, help='要可视化的特征索引')
    parser.add_argument('--grid', type=int, default=50, help='3D网格尺寸')
    parser.add_argument('--iso', type=float, default=0.5, help='等值面水平(最大值的百分比)')
    parser.add_argument('--mode', choices=['3d', 'angular', 'hist', 'structure'], default='3d',
                        help='可视化模式: 3d=完整密度, angular=仅角向, hist=特征直方图, structure=结构与特征')
    parser.add_argument('--atom-index', type=int, help='聚焦的中心原子索引')
    
    args = parser.parse_args()
    
    # 读取特征和元数据
    meta_df = pd.read_csv(args.meta_file)
    
    if args.feature_file.endswith('.npy'):
        features = np.load(args.feature_file)
    else:
        features = np.loadtxt(args.feature_file)
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
    
    # 确认特征索引有效
    if args.feature_idx < 0 or args.feature_idx >= meta_df.shape[0]:
        print(f"无效特征索引: {args.feature_idx}. 有效范围: 0-{meta_df.shape[0]-1}")
        return
    
    # 获取特征元数据
    meta_row = meta_df.iloc[args.feature_idx]
    print(f"\n可视化SOAP特征: ({meta_row['elem1']}, {meta_row['elem2']}, n1={meta_row['n1']}, n2={meta_row['n2']}, l={meta_row['l']})")
    
    if args.mode == '3d':
        # 3D密度可视化
        reconstruct_soap_density(args.structure_file, meta_df, features, 
                                args.feature_idx, args.grid, args.iso)
    
    elif args.mode == 'angular':
        # 仅角向分布可视化
        l = int(meta_row['l'])
        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, projection='3d')
        visualize_angular_distribution(l, title=f"角向分布 (l={l})", ax=ax)
        plt.tight_layout()
        plt.show()
    
    elif args.mode == 'hist':
        # 特征直方图
        visualize_feature_histogram(features, meta_df, args.feature_idx)
    
    elif args.mode == 'structure':
        # 结构与特征可视化
        visualize_soap_with_structure(args.structure_file, meta_df, features, args.feature_idx,
                                      grid_size=args.grid, iso_level=args.iso, atom_index=args.atom_index)


if __name__ == '__main__':
    import sys
    sys.argv = ['soap_fea.py', 
                '/Users/wukai/Desktop/project/wjob/tests/cont/Ag/M-2H/ads/CONTCAR', 
                '/Users/wukai/Desktop/project/wjob/tests/cont/Ag/M-2H/ads/soap_Ag_meta.csv', 
                '/Users/wukai/Desktop/project/wjob/tests/cont/Ag/M-2H/ads/soap_Ag.npy',
                '311', 
                '--grid', '100', 
                '--iso', '0.5', 
                '--mode', 'structure', 
                '--atom-index', 
                '81'
                ] 
    main()
