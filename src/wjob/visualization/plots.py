import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import sph_harm, gamma
from ase.io import read
import pyvista as pv
from dscribe.descriptors import SOAP
from wjob.config import SOAP_PARAMS

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 使用支持中文的字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def gto_radial(n, r, r_cut, n_max):
    """
    Gaussian Type Orbital (GTO) 径向基函数.
    """
    alpha_n = (n_max ** 2) / (r_cut ** 2)
    norm_factor = np.sqrt(2 * alpha_n ** (n + 1.5) / gamma(n + 1.5))
    return norm_factor * r ** n * np.exp(-alpha_n * r ** 2)

def angular_part(l, theta, phi):
    """
    所有m分量叠加 (SOAP定义中对所有m求和).
    """
    Y_sum = np.zeros(theta.shape, dtype=np.complex128)
    for m in range(-l, l + 1):
        Y_sum += sph_harm(m, l, phi, theta)
    return Y_sum.real

def reconstruct_soap_density_precise(structure_file, meta_df, feature_idx, grid_points=60, iso_level=0.5):
    """
    严格重建SOAP特征空间密度，可视化等值曲面。
    """
    # 读取特征元数据
    meta_row = meta_df.iloc[feature_idx]
    n, l = int(meta_row['n1']), int(meta_row['l'])
    elem1, elem2 = meta_row['elem1'], meta_row['elem2']

    # SOAP参数
    r_cut = SOAP_PARAMS["r_cut"]
    n_max = SOAP_PARAMS["n_max"]

    print(f"重建SOAP密度: 元素对 ({elem1}-{elem2}), n={n}, l={l}, r_cut={r_cut} Å")

    # 构造空间网格（真实长度单位: Å）
    spacing = 2 * r_cut / (grid_points - 1)
    grid_lin = np.linspace(-r_cut, r_cut, grid_points)
    X, Y, Z = np.meshgrid(grid_lin, grid_lin, grid_lin, indexing="ij")

    # 计算球坐标
    R = np.sqrt(X**2 + Y**2 + Z**2)
    THETA = np.arccos(np.clip(Z / (R + 1e-10), -1, 1))
    PHI = np.arctan2(Y, X)

    # 计算密度场 (径向×角向)
    mask = R <= r_cut
    density = np.zeros(R.shape)
    radial_vals = np.zeros(R.shape)
    radial_vals[mask] = gto_radial(n, R[mask], r_cut, n_max)
    angular_vals = angular_part(l, THETA, PHI)

    density[mask] = radial_vals[mask] * angular_vals[mask]
    density[~mask] = 0.0

    # 使用Pyvista进行可视化
    grid = pv.ImageData()
    grid.dimensions = np.array(density.shape) + 1
    grid.origin = (-r_cut, -r_cut, -r_cut)
    grid.spacing = (spacing, spacing, spacing)

    grid.cell_data["density"] = density.flatten(order="F")

    # 创建等值面 (iso-surface)
    iso_value = np.max(np.abs(density)) * iso_level
    contours = grid.contour([iso_value])

    # 准备原子结构 (中心原子默认0索引)
    atoms = read(structure_file)
    positions = atoms.positions
    symbols = atoms.get_chemical_symbols()
    atom_radii = {"H":0.25, "C":0.7, "O":0.6, "Ce":1.8, "Ag":1.65}  #尺度可再调整
    atom_colors = {"H":"white","C":"gray","O":"red","Ce":"blue","Ag":"silver"}

    p = pv.Plotter()

    # 显示原子结构
    for i, (sym, pos) in enumerate(zip(symbols, positions)):
        radius = atom_radii.get(sym, 0.7)
        color = atom_colors.get(sym, 'gray')
        sphere = pv.Sphere(radius=radius, center=pos)
        p.add_mesh(sphere, color=color, opacity=0.5)

    # 添加密度等值曲面
    p.add_mesh(contours, opacity=0.7, color='cyan', name='SOAP Density')

    # 中心位置添加截断半径球面（便于参考）
    cutoff_sphere = pv.Sphere(radius=r_cut, center=positions[0])
    p.add_mesh(cutoff_sphere, style='wireframe', color='black', opacity=0.2, name='Cutoff radius')

    # 添加坐标轴和标记
    p.add_axes()
    title_text = f'SOAP Density: ({elem1}, {elem2}), n={n}, l={l} | r_cut={r_cut} Å'
    p.add_text(title_text, font_size=12, position='upper_edge')

    # 可视化
    p.show()

def plot_metrics_comparison(r2_train, r2_test, rmse_train, rmse_test, save_path=None, tick_fontsize=20):
    """
    绘制训练集和测试集的R2和RMSE折线图
    
    参数:
        r2_train (list): 训练集R2值列表
        r2_test (list): 测试集R2值列表
        rmse_train (list): 训练集RMSE值列表
        rmse_test (list): 测试集RMSE值列表
        save_path (str, optional): 图片保存路径，如果为None则不保存
        tick_fontsize (int, optional): 坐标轴刻度标签的字体大小，默认为20
    """
    # 创建图形和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # 生成x轴数据
    epochs = np.arange(1, len(r2_train) + 1)
    
    # 绘制R2曲线
    ax1.plot(epochs, r2_train, 'o-', label='训练集', color='#1f77b4', linewidth=3, markersize=18)
    ax1.plot(epochs, r2_test, 's-', label='测试集', color='#ff7f0e', linewidth=3, markersize=18)
    ax1.set_xlabel('训练轮次', fontsize=28)
    ax1.set_ylabel('R²', fontsize=28)
    ax1.set_title('训练集和测试集R²对比', fontsize=28, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=24, loc='lower right')
    
    # 设置刻度标签字体大小
    ax1.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax1.tick_params(axis='both', which='minor', labelsize=tick_fontsize-2)
    
    # 绘制RMSE曲线
    ax2.plot(epochs, rmse_train, 'o-', label='训练集', color='#1f77b4', linewidth=3, markersize=18)
    ax2.plot(epochs, rmse_test, 's-', label='测试集', color='#ff7f0e', linewidth=3, markersize=18)
    ax2.set_xlabel('训练轮次', fontsize=28)
    ax2.set_ylabel('RMSE (eV)', fontsize=28)
    ax2.set_title('训练集和测试集RMSE对比', fontsize=28, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=24, loc='upper right')
    
    # 设置刻度标签字体大小
    ax2.tick_params(axis='both', which='major', labelsize=tick_fontsize)
    ax2.tick_params(axis='both', which='minor', labelsize=tick_fontsize-2)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'图片已保存至: {save_path}')
    
    plt.show()
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='严格定义SOAP密度场可视化')
    parser.add_argument('structure', help='结构文件 (CONTCAR或其他ASE支持格式)')
    parser.add_argument('meta_csv', help='特征元数据CSV文件')
    parser.add_argument('feature_idx', type=int, help='特征索引 (CSV行索引)')
    parser.add_argument('--grid_points', type=int, default=60, help='网格点数')
    parser.add_argument('--iso_level', type=float, default=0.5, help='等值曲面水平(相对最大值百分比)')

    args = parser.parse_args()

    meta_df = pd.read_csv(args.meta_csv)

    reconstruct_soap_density_precise(
        args.structure,
        meta_df,
        args.feature_idx,
        grid_points=args.grid_points,
        iso_level=args.iso_level
    )
