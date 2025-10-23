"""
SHAP Feature Importance Analysis Tools

This module provides functions for analyzing feature importance using SHAP (SHapley Additive exPlanations).
All visualization functions use English labels and support the latest SHAP API with Explanation objects.
"""

import shap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Union, Optional, Tuple, Any


def rename_features(feature_names: List[str], rename_map: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Rename features according to a mapping dictionary or apply default renaming rules.
    
    Args:
        feature_names: List of original feature names
        rename_map: Optional dictionary mapping original names to new names
        
    Returns:
        List of renamed features
    """
    # Default renaming rules if no map is provided
    if rename_map is None:
        rename_map = {
            # Examples of renaming patterns
            "M-2H_H_O_Ce_atom0_PC": "2H/H_OCe/0",
            "M-2H_H_O_O_atom0_PC": "2H/H_OO/0",
            # Add more patterns as needed
        }
    
    renamed_features = []
    for name in feature_names:
        # Check for exact matches first
        if name in rename_map:
            renamed_features.append(rename_map[name])
            continue
            
        # Check for partial matches (patterns)
        renamed = name
        for pattern, replacement in rename_map.items():
            if pattern in name:
                # Extract number after PC if it exists
                if "PC" in pattern and "PC" in name:
                    pc_num = name.split("PC")[-1]
                    renamed = replacement + pc_num
                    break
                else:
                    renamed = name.replace(pattern, replacement)
                    break
        
        renamed_features.append(renamed)
    
    return renamed_features


def shap_feature_importance(
    model: Any, 
    X: pd.DataFrame, 
    max_display: int = 10,
    feature_rename_map: Optional[Dict[str, str]] = None,
    plot_title: str = None,  # 默认为None，不显示标题
    save_path: Optional[str] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a bar plot of SHAP feature importance.
    
    Args:
        model: Trained model with feature_importances_ attribute or compatible with SHAP
        X: Feature dataset (preferably as pandas DataFrame with column names)
        max_display: Maximum number of top features to display
        feature_rename_map: Dictionary mapping original feature names to display names
        plot_title: Title for the plot (None for no title)
        save_path: Optional path to save the figure
        
    Returns:
        Figure and axes objects
    """
    # 设置全局字体参数 - 使用Arial字体并加大加粗
    plt.rcParams['font.family'] = 'Arial'  # 设置字体为Arial
    plt.rcParams['font.weight'] = 'bold'  # 全局设置字体为粗体
    plt.rcParams['axes.titleweight'] = 'bold'  # 坐标轴标题加粗
    plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
    plt.rcParams['font.size'] = 20  # 全局设置字体大小
    plt.rcParams['axes.linewidth'] = 1.5  # 增加坐标轴线宽
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 添加边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # Create SHAP explainer based on model type
    if hasattr(model, 'predict'):
        # For most sklearn models
        explainer = shap.Explainer(model, X)
    else:
        # Fallback for other model types
        explainer = shap.Explainer(model)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    # Get mean absolute SHAP values for each feature
    mean_abs_shap = np.abs(shap_values.values).mean(0)
    
    # Get feature names (either from DataFrame or create generic names)
    if hasattr(X, 'columns'):
        feature_names = list(X.columns)
    else:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    # Create a DataFrame for sorting
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    })
    
    # Sort by importance and take top features
    feature_importance = feature_importance.sort_values('Importance', ascending=False).head(max_display)
    
    # Rename features if mapping is provided
    if feature_rename_map:
        feature_importance['Feature'] = rename_features(
            feature_importance['Feature'].tolist(), 
            feature_rename_map
        )
    
    # Plot
    bars = ax.barh(
        y=feature_importance['Feature'],
        width=feature_importance['Importance'],
        color='#1E88E5',
        edgecolor='black',
        linewidth=1.5,
        alpha=0.8
    )
    
    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width * 1.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.3f}',
                va='center', fontsize=18, fontweight='bold')
    
    # Customize plot
    if plot_title:
        ax.set_title(plot_title, fontsize=24, fontweight='bold')
    ax.set_xlabel('Mean |SHAP Value| (Impact on Prediction)', fontsize=22, fontweight='bold')
    ax.set_ylabel('Features', fontsize=22, fontweight='bold')
    
    # 调整坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=20)
    
    # 不添加网格线（按要求移除）
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax


def shap_beeswarm_plot(
    model: Any, 
    X: pd.DataFrame, 
    max_display: int = 10,
    feature_rename_map: Optional[Dict[str, str]] = None,
    plot_title: str = None,  # 默认为None，不显示标题
    cmap: str = "coolwarm",
    save_path: Optional[str] = None
) -> None:
    """
    Create a SHAP beeswarm plot showing feature impacts.
    
    Args:
        model: Trained model compatible with SHAP
        X: Feature dataset
        max_display: Maximum number of top features to display
        feature_rename_map: Dictionary mapping original feature names to display names
        plot_title: Title for the plot (None for no title)
        cmap: Colormap for the plot
        save_path: Optional path to save the figure
    """
    # 设置全局字体参数 - 使用Arial字体并加大加粗
    plt.rcParams['font.family'] = 'Arial'  # 设置字体为Arial
    plt.rcParams['font.weight'] = 'bold'  # 全局设置字体为粗体
    plt.rcParams['axes.titleweight'] = 'bold'  # 坐标轴标题加粗
    plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
    plt.rcParams['font.size'] = 20  # 全局设置字体大小
    plt.rcParams['axes.linewidth'] = 1.5  # 增加坐标轴线宽
    
    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    
    # Calculate SHAP values
    shap_values = explainer(X)
    
    # 选取前10个最重要的特征
    # 计算每个特征的平均绝对SHAP值
    mean_abs_shap = np.abs(shap_values.values).mean(0)
    
    # 获取特征名称
    if hasattr(X, 'columns'):
        feature_names = list(X.columns)
    else:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    
    # 创建DataFrame用于排序
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    })
    
    # 按重要性排序并取前几个特征
    top_features = feature_importance.sort_values('Importance', ascending=False).head(max_display)['Feature'].tolist()
    
    # 创建一个新的数据集，只包含前几个重要的特征
    X_top = X[top_features].copy()
    
    # 重新计算SHAP值，只使用前几个特征
    explainer_top = shap.Explainer(model, X_top)
    shap_values_top = explainer_top(X_top)
    
    # If feature renaming is needed, we need to modify the explanation object
    if feature_rename_map:
        # Rename features
        renamed_features = rename_features(top_features, feature_rename_map)
        
        # Update feature names in the explanation object if possible
        if hasattr(shap_values_top, 'feature_names'):
            shap_values_top.feature_names = renamed_features
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 添加边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # 调整坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=22)
    
    # Create beeswarm plot - 注意：shap.plots.beeswarm会创建自己的图形，需要特殊处理
    plt.figure(figsize=(10, 8))  # 创建一个新的图形用于SHAP绘图
    
    # 创建beeswarm plot，只绘制前10个特征
    shap.plots.beeswarm(
        shap_values_top,
        max_display=max_display,  # 这里已经只有前10个特征了
        show=False,
        color=cmap
    )
    
    # 获取当前图形并应用样式
    current_fig = plt.gcf()
    current_ax = plt.gca()
    
    # 设置标题（如果提供）
    if plot_title:
        current_ax.set_title(plot_title, fontsize=24, fontweight='bold')
    else:
        # 移除标题空间
        current_ax.set_title('')
    
    # 设置坐标轴标签字体
    if hasattr(current_ax, 'get_xlabel') and callable(current_ax.get_xlabel):
        current_ax.set_xlabel(current_ax.get_xlabel(), fontsize=22, fontweight='bold')
    if hasattr(current_ax, 'get_ylabel') and callable(current_ax.get_ylabel):
        current_ax.set_ylabel(current_ax.get_ylabel(), fontsize=22, fontweight='bold')
    
    # 调整坐标轴刻度字体
    current_ax.tick_params(axis='both', which='major', labelsize=20)
    
    # 添加边框
    for spine in current_ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # 不添加网格线（按要求移除）
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def shap_waterfall_plot(
    model: Any, 
    X: pd.DataFrame,
    sample_idx: int = 0,
    feature_rename_map: Optional[Dict[str, str]] = None,
    plot_title: str = None,  # 默认为None，不显示标题
    max_display: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Create a SHAP waterfall plot for a single prediction.
    
    Args:
        model: Trained model compatible with SHAP
        X: Feature dataset
        sample_idx: Index of the sample to explain
        feature_rename_map: Dictionary mapping original feature names to display names
        plot_title: Title for the plot (None for no title)
        max_display: Maximum number of features to display
        save_path: Optional path to save the figure
    """
    # 设置全局字体参数 - 使用Arial字体并加大加粗
    plt.rcParams['font.family'] = 'Arial'  # 设置字体为Arial
    plt.rcParams['font.weight'] = 'bold'  # 全局设置字体为粗体
    plt.rcParams['axes.titleweight'] = 'bold'  # 坐标轴标题加粗
    plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
    plt.rcParams['font.size'] = 20  # 全局设置字体大小
    plt.rcParams['axes.linewidth'] = 1.5  # 增加坐标轴线宽
    
    # Create SHAP explainer
    explainer = shap.Explainer(model, X)
    
    # Calculate SHAP values for the specific sample
    shap_values = explainer(X.iloc[[sample_idx]])
    
    # If feature renaming is needed, we need to modify the explanation object
    if feature_rename_map:
        # Get original feature names
        if hasattr(X, 'columns'):
            feature_names = list(X.columns)
        else:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Rename features
        renamed_features = rename_features(feature_names, feature_rename_map)
        
        # Update feature names in the explanation object if possible
        if hasattr(shap_values, 'feature_names'):
            shap_values.feature_names = renamed_features
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create waterfall plot
    shap.plots.waterfall(
        shap_values[0],
        max_display=max_display,
        show=False
        # 注意：某些SHAP版本不支持show_sum参数
    )
    
    # 获取当前图形并应用样式
    current_fig = plt.gcf()
    current_ax = plt.gca()
    
    # 设置标题（如果提供）
    if plot_title:
        current_ax.set_title(plot_title, fontsize=24, fontweight='bold')
    else:
        # 移除标题空间
        current_ax.set_title('')
    
    # 设置坐标轴标签字体
    if hasattr(current_ax, 'get_xlabel') and callable(current_ax.get_xlabel):
        current_ax.set_xlabel(current_ax.get_xlabel(), fontsize=22, fontweight='bold')
    if hasattr(current_ax, 'get_ylabel') and callable(current_ax.get_ylabel):
        current_ax.set_ylabel(current_ax.get_ylabel(), fontsize=22, fontweight='bold')
    
    # 调整坐标轴刻度字体
    current_ax.tick_params(axis='both', which='major', labelsize=20)
    
    # 添加边框
    for spine in current_ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')
    
    # 不添加网格线（按要求移除）
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        # Create directory if it doesn't exist
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_feature_rename_map() -> Dict[str, str]:
    """
    Create a default feature renaming map for SOAP features.
    
    Returns:
        Dictionary mapping original feature names to more readable names
    """
    rename_map = {
        # Basic patterns
        "M-2H_H_O_Ce_atom0_PC": "2H/H_OCe/0",
        "M-2H_H_O_O_atom0_PC": "2H/H_OO/0",
        "M-2H_H_Ce_O_atom0_PC": "2H/H_CeO/0",
        "M-2H_H_Ce_Ce_atom0_PC": "2H/H_CeCe/0",
        
        # Add more patterns as needed
        "M-2H_H_O_Ce_atom1_PC": "2H/H_OCe/1",
        "M-2H_H_O_O_atom1_PC": "2H/H_OO/1",
        "M-2H_H_Ce_O_atom1_PC": "2H/H_CeO/1",
        "M-2H_H_Ce_Ce_atom1_PC": "2H/H_CeCe/1",
    }
    
    return rename_map


def run_complete_shap_analysis(
    model: Any,
    X: pd.DataFrame,
    output_dir: Optional[str] = None,
    sample_idx: int = 0
) -> None:
    # Create output directory if specified and doesn't exist
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
    """
    Run a complete SHAP analysis with all plot types.
    
    Args:
        model: Trained model compatible with SHAP
        X: Feature dataset
        output_dir: Directory to save plots (None for no saving)
        sample_idx: Index of sample to use for waterfall plot
    """
    # Create feature rename map
    rename_map = create_feature_rename_map()
    
    # 1. Feature Importance Bar Plot
    print("Generating SHAP feature importance bar plot...")
    shap_feature_importance(
        model=model,
        X=X,
        max_display=10,
        feature_rename_map=rename_map,
        plot_title="Top 10 Features Impacting Adsorption Energy",
        save_path=f"{output_dir}/shap_importance_bar.png" if output_dir else None
    )
    
    # 2. Beeswarm Plot
    print("Generating SHAP beeswarm plot...")
    shap_beeswarm_plot(
        model=model,
        X=X,
        max_display=10,
        feature_rename_map=rename_map,
        plot_title="Feature Impact on Adsorption Energy Prediction",
        save_path=f"{output_dir}/shap_beeswarm.png" if output_dir else None
    )
    
    # 3. Waterfall Plot for a single prediction
    print("Generating SHAP waterfall plot for sample prediction...")
    shap_waterfall_plot(
        model=model,
        X=X,
        sample_idx=sample_idx,
        feature_rename_map=rename_map,
        plot_title="SHAP Waterfall Plot for Single Adsorption Energy Prediction",
        save_path=f"{output_dir}/shap_waterfall.png" if output_dir else None
    )
    
    print("SHAP analysis complete!")


# Example usage code (commented out)
"""
# Example usage:
from shap_analysis import run_complete_shap_analysis, shap_feature_importance

# Assuming model and X_train are already defined
run_complete_shap_analysis(
    model=model,
    X=X_train,
    output_dir="./plots"
)

# Or individual plots:
shap_feature_importance(
    model=model,
    X=X_train,
    max_display=10
)
"""
