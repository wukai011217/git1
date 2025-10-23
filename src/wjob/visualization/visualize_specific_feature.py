#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
可视化特定SOAP特征: M-2H_H_O_M_n10_n20_l4_atom0

该脚本展示该特征的物理意义和空间分布
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import sph_harm
import os
import sys

# 设置matplotlib参数以优化显示
plt.rcParams['figure.figsize'] = (12, 10)
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

def gto_radial_basis(r, n, alpha=1.0):
    """高斯型轨道径向基函数
    """
    return r**n * np.exp(-alpha * r**2)

def visualize_spherical_harmonic_multiple(l, m_vals=None, cmap='viridis', subplot_layout=None):
    """可视化特定l值下的多个m值的球谐函数
    
    参数:
        l (int): 角量子数
        m_vals (list): 要可视化的磁量子数列表，默认为None表示所有m值
        cmap (str): 颜色映射
        subplot_layout (tuple): 子图布局，如(行数,列数)
    """
    if m_vals is None:
        # 如果未指定m值，则使用所有从-l到l的值
        m_vals = list(range(-l, l+1))
    
    # 确定子图布局
    if subplot_layout is None:
        n_plots = len(m_vals)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        subplot_layout = (rows, cols)
    
    # 创建网格
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2*np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)
    
    # 创建带有自定义布局的图形
    fig = plt.figure(figsize=(subplot_layout[1]*4, subplot_layout[0]*4))
    
    for i, m in enumerate(m_vals):
        # 跳过无效索引
        if i >= subplot_layout[0] * subplot_layout[1]:
            break
            
        # 计算球谐函数
        Y = sph_harm(m, l, theta, phi)
        Y_real = Y.real
        Y_abs = np.abs(Y)
        
        # 调整半径并转换为笛卡尔坐标
        r = 0.5 + 0.5 * (Y_abs / np.max(Y_abs))
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        
        # 归一化实部用于颜色映射
        Y_norm = (Y_real - Y_real.min()) / (Y_real.max() - Y_real.min() + 1e-10)
        
        # 创建子图
        ax = fig.add_subplot(subplot_layout[0], subplot_layout[1], i+1, projection='3d')
        
        # 绘制表面
        surf = ax.plot_surface(x, y, z, facecolors=plt.cm.get_cmap(cmap)(Y_norm), 
                              alpha=0.9, linewidth=0, antialiased=True)
        
        # 设置图形属性
        ax.set_title(f"$Y_{{{l}}}^{{{m}}}(\\theta, \\phi)$")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        # 设置等比例轴
        ax.set_box_aspect([1,1,1])
        
    plt.tight_layout()
    plt.suptitle(f"SOAP特征 M-2H_H_O_M_n10_n20_l4_atom0 的角向部分 (l={l})", fontsize=16, y=1.02)
    return fig

def plot_soap_radial_angular():
    """展示SOAP特征的径向和角向部分组合"""
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # 径向部分 - n1=0, n2=0
    r = np.linspace(0, 6, 1000)
    g_n1 = gto_radial_basis(r, n=0, alpha=0.5)  # n1=0
    g_n2 = gto_radial_basis(r, n=0, alpha=0.5)  # n2=0
    
    axs[0].plot(r, g_n1, 'b-', label='$g_0(r)$ (n1=0)')
    axs[0].plot(r, g_n2, 'r--', label='$g_0(r)$ (n2=0)')
    axs[0].set_title("SOAP特征径向部分: n1=0, n2=0")
    axs[0].set_xlabel("半径 r (Å)")
    axs[0].set_ylabel("基函数值")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend()
    
    # 径向和角向部分结合示意图
    theta = np.linspace(0, 2*np.pi, 100)
    r_vals = np.linspace(0, 3, 4)
    
    # l=4的一种特征形状 (m=0的示例)
    for r_val in r_vals:
        radial = np.exp(-r_val)  # 简化的径向衰减
        # l=4, m=0的简化形状
        angular = 0.5 + 0.3*np.cos(8*theta/2)
        x = r_val * angular * np.cos(theta)
        y = r_val * angular * np.sin(theta)
        
        axs[1].plot(x, y, label=f'r={r_val:.1f}Å')
    
    axs[1].set_title("SOAP特征径向与角向结合示意图 (l=4)")
    axs[1].set_xlabel("X (Å)")
    axs[1].set_ylabel("Y (Å)")
    axs[1].set_aspect('equal')
    axs[1].grid(True, alpha=0.3)
    axs[1].legend()
    
    plt.tight_layout()
    plt.suptitle("SOAP特征 M-2H_H_O_M_n10_n20_l4_atom0 的径向与角向组成", fontsize=16, y=1.05)
    return fig

def visualize_feature_in_structure():
    """使用示意图展示在原子结构中的特征意义
    
    在缺氢的CeO2表面展示H-O-M的相互作用
    """
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    
    # 绘制Ce原子
    ce_x, ce_y = [1, 3, 5], [1, 3, 1]
    ax.scatter(ce_x, ce_y, s=300, c='gold', label='Ce', edgecolors='black')
    
    # 绘制O原子
    o_x, o_y = [1, 3, 5, 2, 4, 4, 2], [3, 1, 3, 2, 2, 4, 4]
    ax.scatter(o_x, o_y, s=150, c='red', label='O', edgecolors='black')
    
    # 绘制H原子 (中心原子)
    h_x, h_y = [3], [2.5]
    ax.scatter(h_x, h_y, s=100, c='cyan', label='H (center)', edgecolors='black')
    
    # 绘制SOAP邻域范围
    circle = plt.Circle((h_x[0], h_y[0]), 2.0, color='blue', fill=False, linestyle='--', alpha=0.5)
    ax.add_artist(circle)
    
    # 为特征相互作用添加线
    for i in range(len(o_x)):
        # 计算H与氧的距离
        dist_h_o = np.sqrt((h_x[0] - o_x[i])**2 + (h_y[0] - o_y[i])**2)
        if dist_h_o <= 2.0:  # 只连接近邻的氧
            ax.plot([h_x[0], o_x[i]], [h_y[0], o_y[i]], 'r-', alpha=0.5)
            
            # 对于每个氧，连接到近邻的Ce
            for j in range(len(ce_x)):
                dist_o_ce = np.sqrt((o_x[i] - ce_x[j])**2 + (o_y[i] - ce_y[j])**2)
                if dist_o_ce <= 2.0:
                    # H-O-Ce三体相互作用
                    ax.plot([o_x[i], ce_x[j]], [o_y[i], ce_y[j]], 'g--', alpha=0.5)
    
    # 标注H-O-Ce三体相互作用
    ax.annotate('H-O-Ce三体相互作用\n(l=4描述复杂空间排布)', 
                xy=(2.5, 2), 
                xytext=(1, 5),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    # 标注多个O原子共同贡献的环境
    ax.annotate('多个O与Ce原子\n形成的复杂局部环境', 
                xy=(3.3, 3), 
                xytext=(6, 4),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8))
    
    ax.set_title("SOAP特征 M-2H_H_O_M_n10_n20_l4_atom0 在原子结构中的意义")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)
    ax.legend(loc='upper right')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    return fig

def main():
    """主函数"""
    print("可视化SOAP特征: M-2H_H_O_M_n10_n20_l4_atom0")
    
    # 1. 可视化l=4的球谐函数 (不同m值)
    fig1 = visualize_spherical_harmonic_multiple(l=4, m_vals=[0, 1, 2, 3, 4])
    plt.savefig('soap_feature_l4_spherical_harmonics.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 2. 可视化径向和角向部分组合
    fig2 = plot_soap_radial_angular()
    plt.savefig('soap_feature_radial_angular_combination.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 3. 在原子结构中展示特征的物理意义
    fig3 = visualize_feature_in_structure()
    plt.savefig('soap_feature_in_structure.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("已保存可视化结果为PNG文件")

if __name__ == "__main__":
    main()
