"""
绘图工具模块。

提供各种绘图相关的工具函数，特别是用于设置绘图样式和中文字体支持。
"""
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, Tuple, List, Union, Dict, Any


def set_plot_style(style: str = 'whitegrid', 
                  font_size: int = 12, 
                  figure_size: Tuple[int, int] = (10, 6)):
    """
    设置全局绘图样式，确保中文显示正常。
    
    Args:
        style (str): seaborn绘图风格，可选值包括'whitegrid', 'darkgrid', 'white', 'dark', 'ticks'
        font_size (int): 字体大小
        figure_size (Tuple[int, int]): 默认图形尺寸
    """
    try:
        import seaborn as sns
        sns.set_style(style)
    except ImportError:
        print("未安装seaborn库，使用matplotlib默认样式")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 设置字体大小
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.titlesize'] = font_size + 2
    plt.rcParams['axes.labelsize'] = font_size
    plt.rcParams['xtick.labelsize'] = font_size - 1
    plt.rcParams['ytick.labelsize'] = font_size - 1
    plt.rcParams['legend.fontsize'] = font_size - 1
    
    # 设置默认图形尺寸
    plt.rcParams['figure.figsize'] = figure_size
    
    # 设置DPI
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300


def create_figure(width: int = 10, 
                 height: int = 6, 
                 style: str = 'whitegrid',
                 font_size: int = 12) -> Tuple[plt.Figure, plt.Axes]:
    """
    创建一个带有设定样式的新图形。
    
    Args:
        width (int): 图形宽度
        height (int): 图形高度
        style (str): 绘图样式
        font_size (int): 字体大小
        
    Returns:
        Tuple[plt.Figure, plt.Axes]: 图形和轴对象
    """
    # 设置样式
    set_plot_style(style, font_size, (width, height))
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(width, height))
    
    return fig, ax


def save_figure(fig: plt.Figure, 
               file_path: str, 
               dpi: int = 300, 
               bbox_inches: str = 'tight',
               create_dir: bool = True) -> bool:
    """
    保存图形到文件。
    
    Args:
        fig (plt.Figure): 要保存的图形
        file_path (str): 文件保存路径
        dpi (int): 分辨率
        bbox_inches (str): 边界框设置
        create_dir (bool): 如果目录不存在是否创建
        
    Returns:
        bool: 是否成功保存
    """
    # 确保目录存在
    if create_dir:
        directory = os.path.dirname(file_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    try:
        fig.savefig(file_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f"图形已保存到: {file_path}")
        return True
    except Exception as e:
        print(f"保存图形时出错: {e}")
        return False


def plot_with_style(x: Union[List, np.ndarray], 
                   y: Union[List, np.ndarray], 
                   title: str = "", 
                   xlabel: str = "", 
                   ylabel: str = "",
                   style: str = 'whitegrid',
                   save_path: Optional[str] = None) -> plt.Figure:
    """
    使用预设样式创建折线图。
    
    Args:
        x: x轴数据
        y: y轴数据
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        style: 绘图样式
        save_path: 保存路径，为None则不保存
        
    Returns:
        plt.Figure: 创建的图形对象
    """
    fig, ax = create_figure(style=style)
    
    # 绘制数据
    ax.plot(x, y)
    
    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图形
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def scatter_with_style(x: Union[List, np.ndarray], 
                      y: Union[List, np.ndarray], 
                      title: str = "", 
                      xlabel: str = "", 
                      ylabel: str = "",
                      color: str = 'blue',
                      alpha: float = 0.7,
                      style: str = 'whitegrid',
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    使用预设样式创建散点图。
    
    Args:
        x: x轴数据
        y: y轴数据
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        color: 点的颜色
        alpha: 透明度
        style: 绘图样式
        save_path: 保存路径，为None则不保存
        
    Returns:
        plt.Figure: 创建的图形对象
    """
    fig, ax = create_figure(style=style)
    
    # 绘制数据
    ax.scatter(x, y, c=color, alpha=alpha)
    
    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图形
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def bar_with_style(x: Union[List, np.ndarray], 
                  y: Union[List, np.ndarray], 
                  title: str = "", 
                  xlabel: str = "", 
                  ylabel: str = "",
                  color: str = 'skyblue',
                  style: str = 'whitegrid',
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    使用预设样式创建条形图。
    
    Args:
        x: x轴数据
        y: y轴数据
        title: 图表标题
        xlabel: x轴标签
        ylabel: y轴标签
        color: 条形颜色
        style: 绘图样式
        save_path: 保存路径，为None则不保存
        
    Returns:
        plt.Figure: 创建的图形对象
    """
    fig, ax = create_figure(style=style)
    
    # 绘制数据
    ax.bar(x, y, color=color)
    
    # 设置标题和标签
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图形
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def setup_chinese_fonts():
    """
    仅设置中文字体支持，不修改其他绘图样式。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    
# 导入时自动设置中文字体
setup_chinese_fonts()
