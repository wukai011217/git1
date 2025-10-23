"""
使用SHAP解释机器学习模型的模块。

提供用于解释模型决策和特征重要性的功能。
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import shap
import joblib
import os
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from sklearn.base import BaseEstimator
from pathlib import Path
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 中文字体设置
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
# 设置全局字体参数 - 使用Arial字体并加大加粗
plt.rcParams['font.family'] = 'Arial'  # 设置字体为Arial
plt.rcParams['font.weight'] = 'bold'  # 全局设置字体为粗体
plt.rcParams['axes.titleweight'] = 'bold'  # 坐标轴标题加粗
plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
plt.rcParams['font.size'] = 20  # 全局设置字体大小
plt.rcParams['axes.linewidth'] = 1.5  # 增加坐标轴线宽
# 忽略警告
warnings.filterwarnings('ignore')

def load_model(model_path: str) -> BaseEstimator:
    """
    加载保存的模型。

    Args:
        model_path: 模型文件路径

    Returns:
        加载的模型对象
    """
    logger.info(f"正在加载模型: {model_path}")
    try:
        model = joblib.load(model_path)
        logger.info(f"模型加载成功: {type(model).__name__}")
        return model
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

def load_data(data_path: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    加载数据集。

    Args:
        data_path: 数据文件路径

    Returns:
        特征矩阵和目标变量
    """
    logger.info(f"正在加载数据: {data_path}")
    try:
        data = pd.read_csv(data_path)
        data = data.dropna()  # 删除缺失值

        # 分离特征和目标变量
        X = data.drop(['element', 'structure_type', 'H2_adsorption_energy'], axis=1)
        y = data['H2_adsorption_energy']

        logger.info(f"数据加载成功: {X.shape[0]} 行, {X.shape[1]} 列")
        return X, y
    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise

def explain_model(model: Any, 
                X: pd.DataFrame, 
                feature_names: Optional[List[str]] = None,
                n_samples: Optional[int] = None) -> shap.Explainer:
    """
    创建模型的SHAP解释器。

    Args:
        model: 训练好的模型
        X: 特征矩阵
        feature_names: 特征名称列表
        n_samples: 用于计算SHAP值的样本数，None表示使用所有样本

    Returns:
        SHAP解释器对象
    """
    try:
        # 选择要用于SHAP计算的样本
        if n_samples is not None and n_samples < X.shape[0]:
            X_sample = shap.sample(X, n_samples)
        else:
            X_sample = X
        
        # 根据模型类型选择合适的解释器
        if str(type(model)).find('xgboost') != -1:
            explainer = shap.TreeExplainer(model)
        elif str(type(model)).find('sklearn.ensemble') != -1:
            explainer = shap.TreeExplainer(model)
        elif str(type(model)).find('sklearn.linear_model') != -1:
            explainer = shap.LinearExplainer(model, X_sample)
        else:
            explainer = shap.KernelExplainer(model.predict, X_sample)
        
        logger.info("SHAP解释器创建成功")
        return explainer
    
    except Exception as e:
        logger.error(f"创建SHAP解释器时出错: {e}")
        return None


def calculate_shap_values(explainer: shap.Explainer, 
                        X: pd.DataFrame) -> np.ndarray:
    """
    计算SHAP值。
    
    Args:
        explainer: SHAP解释器
        X: 特征矩阵
        
    Returns:
        SHAP值数组
    """
    try:
        shap_values = explainer.shap_values(X)
        logger.info(f"已计算 {X.shape[0]} 个样本的SHAP值")
        return shap_values
    except Exception as e:
        logger.error(f"计算SHAP值时出错: {e}")
        return None


def plot_summary(shap_values: np.ndarray, 
               X: pd.DataFrame, 
               feature_names: Optional[List[str]] = None,
               max_display: int = 10,
               plot_type: str = "bar",
               save_path: Optional[str] = None) -> None:
    """
    绘制SHAP摘要图，展示特征重要性。
    
    Args:
        shap_values: SHAP值数组
        X: 特征矩阵
        feature_names: 特征名称列表
        max_display: 显示的最大特征数
        plot_type: 图表类型，"bar"或"dot"
        save_path: 保存路径，None表示不保存
    """
    try:
        if feature_names is not None:
            X_with_names = X.copy()
            X_with_names.columns = feature_names
        else:
            X_with_names = X
        
        plt.figure(figsize=(10, 8))
        if plot_type == "bar":
            shap.summary_plot(shap_values, X_with_names, plot_type="bar", max_display=max_display, show=False)
        else:
            shap.summary_plot(shap_values, X_with_names, max_display=max_display, show=False)
        
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"SHAP摘要图已保存到: {save_path}")
            
        plt.show()
        
    except Exception as e:
        logger.error(f"绘制SHAP摘要图时出错: {e}")


def plot_dependence(shap_values: np.ndarray, 
                  X: pd.DataFrame, 
                  feature_idx: int,
                  interaction_idx: Optional[int] = None,
                  feature_names: Optional[List[str]] = None,
                  save_path: Optional[str] = None) -> None:
    """
    绘制SHAP依赖图，展示特征与模型输出的关系。
    
    Args:
        shap_values: SHAP值数组
        X: 特征矩阵
        feature_idx: 主要特征索引
        interaction_idx: 交互特征索引，None表示不考虑交互
        feature_names: 特征名称列表
        save_path: 保存路径，None表示不保存
    """
    try:
        if feature_names is not None:
            X_with_names = X.copy()
            X_with_names.columns = feature_names
            feature = feature_names[feature_idx]
            interaction = None if interaction_idx is None else feature_names[interaction_idx]
        else:
            X_with_names = X
            feature = feature_idx
            interaction = interaction_idx
        
        plt.figure(figsize=(10, 7))
        shap.dependence_plot(
            feature, 
            shap_values, 
            X_with_names,
            interaction_index=interaction,
            show=False
        )
        
        plt.tight_layout()
        
        if save_path:
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"SHAP依赖图已保存到: {save_path}")
            
        plt.show()
        
    except Exception as e:
        logger.error(f"绘制SHAP依赖图时出错: {e}")


def get_feature_importance(shap_values: np.ndarray, 
                         feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    从SHAP值计算特征重要性。
    
    Args:
        shap_values: SHAP值数组
        feature_names: 特征名称列表
        
    Returns:
        包含特征重要性的DataFrame
    """
    # 计算每个特征的平均绝对SHAP值
    importance = np.mean(np.abs(shap_values), axis=0)
    
    # 创建特征名称列表
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(len(importance))]
    
    # 创建DataFrame并按重要性降序排序
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    return importance_df


def get_model_feature_importance(model: Any, feature_names: List[str], 
                                 top_n: int = 10, 
                                 figsize: Tuple[int, int] = (10, 8),
                                 title: str = '',
                                 save_path: Optional[str] = None) -> pd.DataFrame:
    """
    获取模型原生的特征重要性（适用于树模型）。
    
    Args:
        model: 训练好的模型
        feature_names: 特征名称列表
        top_n: 显示的特征数量
        figsize: 图表尺寸
        title: 图表标题
        save_path: 保存图表的路径，None不保存
    
    Returns:
        特征重要性DataFrame
    """
    try:
        # 检查模型是否有特征重要性属性
        if not hasattr(model, 'feature_importances_'):
            logger.warning(f"模型 {type(model).__name__} 没有feature_importances_属性")
            return pd.DataFrame()
            
        # 获取特征重要性
        importances = model.feature_importances_
        
        # 创建DataFrame
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # 按重要性排序
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # 绘图
        plt.figure(figsize=figsize)
        plt.title(title, fontsize=15)
        top_features = importance_df.head(top_n)
        plt.barh(top_features['Feature'], top_features['Importance'])
        plt.xlabel('')
        plt.ylabel('')
        plt.gca().invert_yaxis()  # 让最重要的特征显示在顶部
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"特征重要性图表已保存到: {save_path}")
            
        plt.show()
        
        return importance_df
        
    except Exception as e:
        logger.error(f"获取特征重要性失败: {str(e)}")
        return pd.DataFrame()


def analyze_catalyst_features(model: Any, 
                            X: pd.DataFrame, 
                            feature_names: List[str],
                            save_dir: Optional[str] = None,
                            n_samples: Optional[int] = None) -> Dict[str, Any]:
    """
    全面分析催化剂特征对性能的影响。
    
    Args:
        model: 训练好的模型
        X: 特征矩阵
        feature_names: 特征名称列表
        save_dir: 保存结果的目录，None表示不保存
        n_samples: 用于SHAP计算的样本数
        
    Returns:
        包含分析结果的字典
    """
    results = {}
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
    model_name = type(model).__name__
    
    # 1. 模型自带特征重要性（如果支持）
    if hasattr(model, 'feature_importances_'):
        logger.info("分析模型自带的特征重要性")
        model_imp_path = None
        if save_dir:
            model_imp_path = os.path.join(save_dir, f"{model_name}_feature_importance_{timestamp}.png")
            
        model_importance = get_model_feature_importance(
            model, 
            feature_names, 
            title=f"",
            save_path=model_imp_path
        )
        
        results['model_feature_importance'] = model_importance
        
        if save_dir and not model_importance.empty:
            model_imp_csv = os.path.join(save_dir, f"{model_name}_feature_importance_{timestamp}.csv")
            model_importance.to_csv(model_imp_csv, index=False)
            logger.info(f"模型特征重要性已保存到: {model_imp_csv}")
    
    # 2. SHAP特征重要性分析
    try:
        logger.info("开始SHAP分析")
        # 创建解释器
        explainer = explain_model(model, X, feature_names, n_samples)
        if explainer is None:
            return results
        
        # 计算SHAP值
        shap_values = calculate_shap_values(explainer, X)
        if shap_values is None:
            return results
        
        # 计算特征重要性
        importance_df = get_feature_importance(shap_values, feature_names)
        results['shap_feature_importance'] = importance_df
        
        # 保存结果
        if save_dir:
            # 确保目录存在
            os.makedirs(save_dir, exist_ok=True)
            
            # 保存重要性摘要图
            summary_path = os.path.join(save_dir, f"shap_summary_bar_{timestamp}.png")
            plot_summary(shap_values, X, feature_names, plot_type="bar", save_path=summary_path)
            
            dot_summary_path = os.path.join(save_dir, f"shap_summary_dot_{timestamp}.png")
            plot_summary(shap_values, X, feature_names, plot_type="dot", save_path=dot_summary_path)
            
            # 保存前三个最重要特征的依赖图
            for i, feature in enumerate(importance_df['Feature'].head(3)):
                feature_idx = feature_names.index(feature)
                dep_path = os.path.join(save_dir, f"shap_dependence_{feature}_{timestamp}.png")
                plot_dependence(shap_values, X, feature_idx, 
                              feature_names=feature_names, 
                              save_path=dep_path)
            
            # 保存特征重要性CSV
            shap_imp_csv = os.path.join(save_dir, f"shap_feature_importance_{timestamp}.csv")
            importance_df.to_csv(shap_imp_csv, index=False)
            logger.info(f"SHAP特征重要性已保存到: {shap_imp_csv}")
        
        logger.info("SHAP分析完成")
    except Exception as e:
        logger.error(f"SHAP分析失败: {str(e)}")
    
    return results


def main():
    """
    主函数，用于命令行调用。
    """
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='模型特征重要性分析工具')
    parser.add_argument('--model', type=str, required=True, help='保存的模型文件路径')
    parser.add_argument('--data', type=str, required=True, help='数据文件路径，包含特征和目标变量')
    parser.add_argument('--output', type=str, default=None, help='输出目录，默认为模型所在目录的importance_analysis子目录')
    parser.add_argument('--samples', type=int, default=100, help='用于SHAP分析的样本数，默认100')
    
    args = parser.parse_args()
    
    # 设置默认输出目录
    if args.output is None:
        model_dir = os.path.dirname(os.path.abspath(args.model))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        args.output = os.path.join(model_dir, f"importance_analysis_{timestamp}")
    
    # 加载模型和数据
    try:
        model = load_model(args.model)
        X, y = load_data(args.data)
        feature_names = X.columns.tolist()
        
        logger.info(f"开始分析模型 {type(model).__name__} 的特征重要性")
        logger.info(f"输出目录: {args.output}")
        
        # 分析特征重要性
        results = analyze_catalyst_features(
            model=model,
            X=X,
            feature_names=feature_names,
            save_dir=args.output,
            n_samples=args.samples
        )
        
        # 输出分析结果摘要
        logger.info("特征重要性分析完成")
        if 'model_feature_importance' in results and not results['model_feature_importance'].empty:
            print("\n模型自带特征重要性 (前10):\n")
            print(results['model_feature_importance'].head(10))
        
        if 'shap_feature_importance' in results and not results['shap_feature_importance'].empty:
            print("\nSHAP特征重要性 (前10):\n")
            print(results['shap_feature_importance'].head(10))
            
    except Exception as e:
        logger.error(f"分析过程中出错: {str(e)}")


if __name__ == "__main__":
    # 当直接运行此脚本时，设置默认参数
    import sys
    sys.argv = [
        __file__,
        '--model', '/Users/wukai/Desktop/project/wjob/src/wjob/ml/models/saved_models/ExtraTreesRegressor_20250618_151314.joblib',
        '--data', '/Users/wukai/Desktop/project/wjob/data/fea/final/reduced_dataset_90pct.csv',
        '--samples', '200'
    ]
    main()
