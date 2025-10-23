
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')


class LearningCurveAnalyzer:
    """
    学习曲线分析器
    
    评估模型性能与训练数据量的关系，判断数据饱和度
    """
    
    def __init__(self, model_params=None):
        """
        初始化分析器
        
        Parameters:
        -----------
        model_params: dict, ExtraTreesRegressor参数
        """
        if model_params is None:
            model_params = {
                'max_features': 0.3,
                'min_samples_split': 3,
                'n_estimators': 179,
                'n_jobs': -1,
                'random_state': 42
            }
        self.model_params = model_params
        self.results = {}
    
    def generate_learning_curve(self, X, y, train_sizes=None, cv=5, random_state=42):
        """
        生成学习曲线数据
        
        Parameters:
        -----------
        X: array, 特征矩阵
        y: array, 目标变量
        train_sizes: array, 训练集大小（比例或绝对数量）
        cv: int, 交叉验证折数
        random_state: int, 随机种子
        
        Returns:
        --------
        train_sizes_abs: array, 绝对训练集大小
        train_scores: array, 训练分数
        val_scores: array, 验证分数
        """
        if train_sizes is None:
            # 从10%到100%，生成学习曲线
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        model = ExtraTreesRegressor(**self.model_params)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='neg_mean_squared_error',
            random_state=random_state,
            n_jobs=-1
        )
        
        # 转换为正的RMSE
        train_rmse = np.sqrt(-train_scores)
        val_rmse = np.sqrt(-val_scores)
        
        # 计算R²分数
        train_sizes_abs_r2, train_scores_r2, val_scores_r2 = learning_curve(
            model, X, y,
            train_sizes=train_sizes,
            cv=cv,
            scoring='r2',
            random_state=random_state,
            n_jobs=-1
        )
        
        self.results = {
            'train_sizes': train_sizes_abs,
            'train_rmse_mean': np.mean(train_rmse, axis=1),
            'train_rmse_std': np.std(train_rmse, axis=1),
            'val_rmse_mean': np.mean(val_rmse, axis=1),
            'val_rmse_std': np.std(val_rmse, axis=1),
            'train_r2_mean': np.mean(train_scores_r2, axis=1),
            'train_r2_std': np.std(train_scores_r2, axis=1),
            'val_r2_mean': np.mean(val_scores_r2, axis=1),
            'val_r2_std': np.std(val_scores_r2, axis=1)
        }
        
        return self.results
    
    def fit_saturation_curve(self, metric='val_r2'):
        """
        拟合饱和曲线来评估数据饱和度
        
        Parameters:
        -----------
        metric: str, 要拟合的指标 ('val_r2' 或 'val_rmse')
        
        Returns:
        --------
        fitted_params: array, 拟合参数
        r_squared: float, 拟合优度
        saturation_estimate: float, 饱和度估计
        """
        if not self.results:
            raise ValueError("请先运行generate_learning_curve")
        
        x = self.results['train_sizes']
        
        if metric == 'val_r2':
            y = self.results['val_r2_mean']
            # 使用幂律衰减函数拟合: y = a - b * x^(-c)
            def saturation_func(x, a, b, c):
                return a - b * np.power(x, -c)
        elif metric == 'val_rmse':
            y = self.results['val_rmse_mean']
            # 使用指数衰减函数拟合: y = a + b * exp(-c * x)
            def saturation_func(x, a, b, c):
                return a + b * np.exp(-c * x)
        else:
            raise ValueError("metric must be 'val_r2' or 'val_rmse'")
        
        try:
            # 拟合曲线
            if metric == 'val_r2':
                # 对R²使用合理的初始参数
                max_y = np.max(y)
                popt, _ = curve_fit(saturation_func, x, y, 
                                  p0=[max_y + 0.1, 0.5, 0.5],
                                  maxfev=2000)
            else:
                # 对RMSE使用合理的初始参数
                min_y = np.min(y)
                popt, _ = curve_fit(saturation_func, x, y,
                                  p0=[min_y, y[0] - min_y, 0.001],
                                  maxfev=2000)
            
            # 计算拟合优度
            y_pred = saturation_func(x, *popt)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # 估计饱和值
            if metric == 'val_r2':
                saturation_estimate = popt[0]  # a参数
            else:
                saturation_estimate = popt[0]  # a参数
            
            return popt, r_squared, saturation_estimate, saturation_func
            
        except Exception as e:
            print(f"拟合失败: {e}")
            return None, None, None, None
    
    def predict_performance_gain(self, target_sizes, metric='val_r2'):
        """
        预测在不同数据量下的性能提升
        
        Parameters:
        -----------
        target_sizes: array, 目标数据集大小
        metric: str, 预测指标
        
        Returns:
        --------
        predictions: array, 预测的性能值
        improvements: array, 相对于当前最大数据量的改进
        """
        current_max_size = np.max(self.results['train_sizes'])
        current_performance = self.results[f'{metric}_mean'][-1]
        
        # 获取拟合函数
        popt, r_squared, saturation_estimate, saturation_func = self.fit_saturation_curve(metric)
        
        if popt is None:
            print("无法拟合饱和曲线，使用线性外推")
            # 简单的线性外推
            slope = (current_performance - self.results[f'{metric}_mean'][0]) / current_max_size
            predictions = current_performance + slope * (target_sizes - current_max_size)
            if metric == 'val_rmse':
                predictions = np.maximum(predictions, current_performance * 0.5)  # 避免过度乐观
        else:
            predictions = saturation_func(target_sizes, *popt)
        
        if metric == 'val_r2':
            improvements = predictions - current_performance
        else:  # val_rmse
            improvements = current_performance - predictions  # RMSE减少是改进
        
        return predictions, improvements, r_squared
    
    def analyze_saturation(self, threshold=0.01):
        """
        分析数据饱和度
        
        Parameters:
        -----------
        threshold: float, 性能改进阈值
        
        Returns:
        --------
        analysis: dict, 饱和度分析结果
        """
        if not self.results:
            raise ValueError("请先运行generate_learning_curve")
        
        # 分析R²饱和度
        r2_popt, r2_r_sq, r2_sat, r2_func = self.fit_saturation_curve('val_r2')
        
        # 分析RMSE饱和度
        rmse_popt, rmse_r_sq, rmse_sat, rmse_func = self.fit_saturation_curve('val_rmse')
        
        current_size = np.max(self.results['train_sizes'])
        current_r2 = self.results['val_r2_mean'][-1]
        current_rmse = self.results['val_rmse_mean'][-1]
        
        # 预测双倍数据的性能
        double_size = current_size * 2
        if r2_popt is not None:
            pred_r2_double = r2_func(double_size, *r2_popt)
            r2_improvement = pred_r2_double - current_r2
        else:
            r2_improvement = 0
            
        if rmse_popt is not None:
            pred_rmse_double = rmse_func(double_size, *rmse_popt)
            rmse_improvement = current_rmse - pred_rmse_double
        else:
            rmse_improvement = 0
        
        # 判断饱和度
        is_saturated = (abs(r2_improvement) < threshold and abs(rmse_improvement) < threshold)
        
        analysis = {
            'current_size': current_size,
            'current_r2': current_r2,
            'current_rmse': current_rmse,
            'r2_saturation_fit': r2_r_sq,
            'rmse_saturation_fit': rmse_r_sq,
            'predicted_r2_at_double_size': pred_r2_double if r2_popt is not None else None,
            'predicted_rmse_at_double_size': pred_rmse_double if rmse_popt is not None else None,
            'r2_improvement_at_double_size': r2_improvement,
            'rmse_improvement_at_double_size': rmse_improvement,
            'is_saturated': is_saturated,
            'saturation_threshold': threshold
        }
        
        return analysis


def plot_learning_curve_analysis(analyzer, figsize=(16, 12)):
    """
    绘制学习曲线分析图
    
    Parameters:
    -----------
    analyzer: LearningCurveAnalyzer, 已运行学习曲线的分析器
    figsize: tuple, 图形大小
    """
    if not analyzer.results:
        raise ValueError("分析器尚未运行学习曲线分析")
    
    # 设置中文字体
    plt.rcParams['font.family'] = ['Arial', 'SimHei', 'DejaVu Sans']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Learning Curve Analysis - Dataset Size vs Performance', fontsize=16, fontweight='bold')
    
    results = analyzer.results
    train_sizes = results['train_sizes']
    
    # 1. RMSE学习曲线
    ax1 = axes[0, 0]
    ax1.errorbar(train_sizes, results['train_rmse_mean'], yerr=results['train_rmse_std'],
                 label='Train RMSE', marker='o', capsize=5, capthick=2)
    ax1.errorbar(train_sizes, results['val_rmse_mean'], yerr=results['val_rmse_std'],
                 label='Validation RMSE', marker='s', capsize=5, capthick=2)
    
    # 添加饱和曲线拟合
    rmse_popt, rmse_r_sq, rmse_sat, rmse_func = analyzer.fit_saturation_curve('val_rmse')
    if rmse_popt is not None:
        x_smooth = np.linspace(train_sizes[0], train_sizes[-1] * 1.5, 100)
        y_smooth = rmse_func(x_smooth, *rmse_popt)
        ax1.plot(x_smooth, y_smooth, '--', alpha=0.7, 
                label=f'Saturation Fit (R²={rmse_r_sq:.3f})')
    
    ax1.set_xlabel('Training Set Size')
    ax1.set_ylabel('RMSE (eV)')
    ax1.set_title('RMSE vs Training Set Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. R²学习曲线
    ax2 = axes[0, 1]
    ax2.errorbar(train_sizes, results['train_r2_mean'], yerr=results['train_r2_std'],
                 label='Train R²', marker='o', capsize=5, capthick=2)
    ax2.errorbar(train_sizes, results['val_r2_mean'], yerr=results['val_r2_std'],
                 label='Validation R²', marker='s', capsize=5, capthick=2)
    
    # 添加饱和曲线拟合
    r2_popt, r2_r_sq, r2_sat, r2_func = analyzer.fit_saturation_curve('val_r2')
    if r2_popt is not None:
        x_smooth = np.linspace(train_sizes[0], train_sizes[-1] * 1.5, 100)
        y_smooth = r2_func(x_smooth, *r2_popt)
        ax2.plot(x_smooth, y_smooth, '--', alpha=0.7,
                label=f'Saturation Fit (R²={r2_r_sq:.3f})')
    
    ax2.set_xlabel('Training Set Size')
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² vs Training Set Size')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 性能改进预测
    ax3 = axes[1, 0]
    current_size = np.max(train_sizes)
    future_sizes = np.linspace(current_size, current_size * 3, 50)
    
    try:
        pred_r2, improvements_r2, r2_fit_quality = analyzer.predict_performance_gain(future_sizes, 'val_r2')
        ax3.plot(future_sizes, improvements_r2, 'b-', label='R² Improvement', linewidth=2)
        ax3.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='Significant Threshold (0.01)')
        ax3.axvline(x=current_size, color='g', linestyle=':', alpha=0.7, label='Current Size')
        
        ax3.set_xlabel('Training Set Size')
        ax3.set_ylabel('R² Improvement')
        ax3.set_title('Predicted Performance Gain')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    except Exception as e:
        ax3.text(0.5, 0.5, f'预测失败: {str(e)}', transform=ax3.transAxes, ha='center')
    
    # 4. 饱和度分析摘要
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    try:
        analysis = analyzer.analyze_saturation()
        
        summary_text = f"""
Dataset Saturation Analysis

Current Dataset Size: {analysis['current_size']:,}
Current Performance:
• R² = {analysis['current_r2']:.4f}
• RMSE = {analysis['current_rmse']:.4f} eV

Predicted at 2× Size:
• R² = {analysis['predicted_r2_at_double_size']:.4f} (+{analysis['r2_improvement_at_double_size']:.4f})
• RMSE = {analysis['predicted_rmse_at_double_size']:.4f} eV ({analysis['rmse_improvement_at_double_size']:.4f})

Saturation Status: {'SATURATED' if analysis['is_saturated'] else 'NOT SATURATED'}

Recommendation:
"""
        
        if analysis['is_saturated']:
            summary_text += "• Current dataset size is sufficient\n• Additional DFT points unlikely to provide\n  substantial improvements\n• Focus on feature engineering or\n  model complexity"
        else:
            summary_text += "• Additional DFT points expected to\n  improve performance\n• Consider collecting more data\n• Target improvement possible"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=11, fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
                
    except Exception as e:
        ax4.text(0.5, 0.5, f'分析失败: {str(e)}', transform=ax4.transAxes, ha='center')
    
    plt.tight_layout()
    return fig


def quick_learning_curve_analysis(X, y, model_params=None, test_size=0.2, random_state=42):
    """
    快速学习曲线分析（一键运行）
    
    Parameters:
    -----------
    X: array, 特征矩阵
    y: array, 目标变量
    model_params: dict, 模型参数
    test_size: float, 测试集比例
    random_state: int, 随机种子
    
    Returns:
    --------
    analyzer: LearningCurveAnalyzer, 分析器对象
    analysis: dict, 饱和度分析结果
    """
    print("开始学习曲线分析...")
    
    # 创建分析器
    analyzer = LearningCurveAnalyzer(model_params)
    
    # 生成学习曲线
    print("生成学习曲线数据...")
    analyzer.generate_learning_curve(X, y, random_state=random_state)
    
    # 分析饱和度
    print("分析数据饱和度...")
    analysis = analyzer.analyze_saturation()
    
    # 绘制图形
    print("绘制分析图...")
    fig = plot_learning_curve_analysis(analyzer)
    
    # 打印结果
    print("\n" + "="*60)
    print("学习曲线分析结果")
    print("="*60)
    
    print(f"当前数据集大小: {analysis['current_size']:,}")
    print(f"当前性能: R² = {analysis['current_r2']:.4f}, RMSE = {analysis['current_rmse']:.4f} eV")
    
    if analysis['predicted_r2_at_double_size'] is not None:
        print(f"预测双倍数据性能: R² = {analysis['predicted_r2_at_double_size']:.4f} (+{analysis['r2_improvement_at_double_size']:.4f})")
        print(f"预测双倍数据性能: RMSE = {analysis['predicted_rmse_at_double_size']:.4f} eV ({analysis['rmse_improvement_at_double_size']:.4f})")
    
    print(f"数据饱和状态: {'已饱和' if analysis['is_saturated'] else '未饱和'}")
    
    if analysis['is_saturated']:
        print("\n建议:")
        print("• 当前数据集大小已足够")
        print("• 额外的DFT计算点不太可能显著提升性能")  
        print("• 建议关注特征工程或模型优化")
    else:
        print("\n建议:")
        print("• 增加DFT数据点预期能提升性能")
        print("• 建议收集更多训练数据")
        print("• 目标改进是可能的")
    
    plt.show()
    return analyzer, analysis
