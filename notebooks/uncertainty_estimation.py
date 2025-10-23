import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class UncertaintyExtraTrees:
    """
    带有不确定性估计的ExtraTrees回归器
    
    这个类扩展了ExtraTreesRegressor，提供了多种不确定性估计方法：
    1. 基于树间方差的不确定性
    2. 预测区间估计
    3. 置信度评分
    """
    
    def __init__(self, **kwargs):
        """
        初始化模型
        
        Parameters:
        -----------
        **kwargs: ExtraTreesRegressor的参数
        """
        self.model = ExtraTreesRegressor(**kwargs)
        self.fitted = False
        
    def fit(self, X, y):
        """训练模型"""
        self.model.fit(X, y)
        self.fitted = True
        return self
    
    def predict_with_uncertainty(self, X, confidence_level=0.95):
        """
        预测并返回不确定性估计
        
        Parameters:
        -----------
        X: array-like, 输入特征
        confidence_level: float, 置信水平 (default: 0.95)
        
        Returns:
        --------
        predictions: array, 预测值
        uncertainties: array, 不确定性(标准差)
        lower_bounds: array, 置信区间下界
        upper_bounds: array, 置信区间上界
        """
        if not self.fitted:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # 获取每棵树的预测
        tree_predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # 计算均值预测
        predictions = np.mean(tree_predictions, axis=0)
        
        # 计算不确定性（树间预测的标准差）
        uncertainties = np.std(tree_predictions, axis=0)
        
        # 计算置信区间
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(1 - alpha/2)
        
        lower_bounds = predictions - z_score * uncertainties
        upper_bounds = predictions + z_score * uncertainties
        
        return predictions, uncertainties, lower_bounds, upper_bounds
    
    def predict(self, X):
        """标准预测方法，保持与sklearn兼容"""
        return self.model.predict(X)
    
    def get_prediction_confidence(self, X, threshold_percentile=75):
        """
        获取预测置信度评分
        
        Parameters:
        -----------
        X: array-like, 输入特征
        threshold_percentile: float, 用于分类高/低置信度的百分位数
        
        Returns:
        --------
        confidence_scores: array, 置信度评分 (0-1, 1表示最高置信度)
        confidence_labels: array, 置信度标签 ('high', 'medium', 'low')
        """
        _, uncertainties, _, _ = self.predict_with_uncertainty(X)
        
        # 将不确定性转换为置信度评分（不确定性越低，置信度越高）
        # 使用反向归一化
        max_uncertainty = np.max(uncertainties)
        min_uncertainty = np.min(uncertainties)
        
        if max_uncertainty == min_uncertainty:
            confidence_scores = np.ones_like(uncertainties)
        else:
            confidence_scores = 1 - (uncertainties - min_uncertainty) / (max_uncertainty - min_uncertainty)
        
        # 分类置信度
        high_threshold = np.percentile(confidence_scores, threshold_percentile)
        low_threshold = np.percentile(confidence_scores, 100 - threshold_percentile)
        
        confidence_labels = np.where(confidence_scores >= high_threshold, 'high',
                                   np.where(confidence_scores >= low_threshold, 'medium', 'low'))
        
        return confidence_scores, confidence_labels


def plot_uncertainty_analysis(y_true, predictions, uncertainties, lower_bounds, upper_bounds, 
                            dataset_name="Test", figsize=(15, 10)):
    """
    绘制不确定性分析图
    
    Parameters:
    -----------
    y_true: array, 真实值
    predictions: array, 预测值
    uncertainties: array, 不确定性
    lower_bounds: array, 置信区间下界
    upper_bounds: array, 置信区间上界
    dataset_name: str, 数据集名称
    figsize: tuple, 图形大小
    """
    # 设置中文字体和样式
    plt.rcParams['font.family'] = ['Arial', 'SimHei', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle(f'不确定性分析 - {dataset_name}数据集', fontsize=16, fontweight='bold')
    
    # 1. 预测vs真实值散点图（带误差条）
    ax1 = axes[0, 0]
    scatter = ax1.errorbar(y_true, predictions, yerr=uncertainties, 
                          fmt='o', alpha=0.6, capsize=3, capthick=1, 
                          color='blue', ecolor='lightblue', markersize=4)
    
    # 添加理想线
    min_val = min(np.min(y_true), np.min(predictions))
    max_val = max(np.max(y_true), np.max(predictions))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想预测线')
    
    ax1.set_xlabel('真实值 (eV)')
    ax1.set_ylabel('预测值 (eV)')
    ax1.set_title('预测vs真实值 (带不确定性)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 计算统计指标
    r2 = r2_score(y_true, predictions)
    rmse = np.sqrt(mean_squared_error(y_true, predictions))
    mean_uncertainty = np.mean(uncertainties)
    
    # 添加统计信息
    stats_text = f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\n平均不确定性 = {mean_uncertainty:.3f}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 2. 不确定性分布直方图
    ax2 = axes[0, 1]
    ax2.hist(uncertainties, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(np.mean(uncertainties), color='red', linestyle='--', 
                label=f'平均值: {np.mean(uncertainties):.3f}')
    ax2.axvline(np.median(uncertainties), color='orange', linestyle='--', 
                label=f'中位数: {np.median(uncertainties):.3f}')
    ax2.set_xlabel('不确定性 (eV)')
    ax2.set_ylabel('频次')
    ax2.set_title('不确定性分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 不确定性vs预测误差
    ax3 = axes[1, 0]
    errors = np.abs(y_true - predictions)
    ax3.scatter(uncertainties, errors, alpha=0.6, color='purple')
    
    # 计算相关性
    correlation = np.corrcoef(uncertainties, errors)[0, 1]
    ax3.set_xlabel('不确定性 (eV)')
    ax3.set_ylabel('|预测误差| (eV)')
    ax3.set_title(f'不确定性vs预测误差\n相关系数: {correlation:.3f}')
    ax3.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(uncertainties, errors, 1)
    p = np.poly1d(z)
    ax3.plot(uncertainties, p(uncertainties), "r--", alpha=0.8)
    
    # 4. 置信区间覆盖率分析
    ax4 = axes[1, 1]
    
    # 计算不同置信水平下的覆盖率
    confidence_levels = np.arange(0.1, 1.0, 0.05)
    coverage_rates = []
    
    for conf_level in confidence_levels:
        alpha = 1 - conf_level
        z_score = stats.norm.ppf(1 - alpha/2)
        temp_lower = predictions - z_score * uncertainties
        temp_upper = predictions + z_score * uncertainties
        coverage = np.mean((y_true >= temp_lower) & (y_true <= temp_upper))
        coverage_rates.append(coverage)
    
    ax4.plot(confidence_levels, coverage_rates, 'bo-', label='实际覆盖率')
    ax4.plot(confidence_levels, confidence_levels, 'r--', label='理论覆盖率')
    ax4.set_xlabel('置信水平')
    ax4.set_ylabel('覆盖率')
    ax4.set_title('置信区间校准图')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def analyze_prediction_confidence(model, X, y, feature_names=None):
    """
    分析预测置信度
    
    Parameters:
    -----------
    model: UncertaintyExtraTrees, 训练好的模型
    X: array, 输入特征
    y: array, 真实标签
    feature_names: list, 特征名称
    
    Returns:
    --------
    analysis_results: dict, 分析结果
    """
    predictions, uncertainties, lower_bounds, upper_bounds = model.predict_with_uncertainty(X)
    confidence_scores, confidence_labels = model.get_prediction_confidence(X)
    
    # 统计不同置信度级别的性能
    results = {}
    for conf_label in ['high', 'medium', 'low']:
        mask = confidence_labels == conf_label
        if np.sum(mask) > 0:
            subset_y = y[mask]
            subset_pred = predictions[mask]
            subset_unc = uncertainties[mask]
            
            results[conf_label] = {
                'count': np.sum(mask),
                'r2': r2_score(subset_y, subset_pred),
                'rmse': np.sqrt(mean_squared_error(subset_y, subset_pred)),
                'mean_uncertainty': np.mean(subset_unc),
                'percentage': np.sum(mask) / len(y) * 100
            }
    
    return results, confidence_scores, confidence_labels


def print_uncertainty_summary(results):
    """打印不确定性分析摘要"""
    print("=" * 60)
    print("不确定性分析摘要")
    print("=" * 60)
    
    for conf_level, metrics in results.items():
        print(f"\n{conf_level.upper()}置信度预测:")
        print(f"  样本数量: {metrics['count']} ({metrics['percentage']:.1f}%)")
        print(f"  R²得分: {metrics['r2']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  平均不确定性: {metrics['mean_uncertainty']:.4f}")
    
    print("\n" + "=" * 60)
    print("建议:")
    print("1. 高置信度预测可以直接使用")
    print("2. 中等置信度预测需要谨慎验证")
    print("3. 低置信度预测建议收集更多训练数据或特征工程")
    print("4. 不确定性与预测误差的相关性越高，模型的自我评估能力越好")
