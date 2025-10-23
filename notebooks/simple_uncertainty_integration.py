
"""
简单的不确定性估计集成代码
可以直接添加到你现有的jupyter notebook中

只需要替换你现有的ExtraTreesRegressor即可获得不确定性估计功能
"""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from scipy import stats
import matplotlib.pyplot as plt

def add_uncertainty_to_extratrees(model, X):
    """
    为已训练的ExtraTreesRegressor添加不确定性估计
    
    Parameters:
    -----------
    model: trained ExtraTreesRegressor
    X: input features
    
    Returns:
    --------
    predictions: 预测值
    uncertainties: 不确定性(标准差)
    confidence_intervals: 95%置信区间 (lower, upper)
    """
    # 获取每棵树的预测
    tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
    
    # 计算均值预测和不确定性
    predictions = np.mean(tree_predictions, axis=0)
    uncertainties = np.std(tree_predictions, axis=0)
    
    # 计算95%置信区间
    z_score = stats.norm.ppf(0.975)  # 95% confidence level
    lower_bounds = predictions - z_score * uncertainties
    upper_bounds = predictions + z_score * uncertainties
    
    return predictions, uncertainties, (lower_bounds, upper_bounds)


def plot_with_uncertainty(y_train, y_test, y_pred1, y_pred, y_test_uncertainty=None):
    """
    改进的绘图函数，添加不确定性信息
    可以替换你现有的plot_figure函数
    """
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 1.5
    
    if y_test_uncertainty is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 左图：原始图
        ax1.scatter(y_train, y_pred1, c='b', label='Train Data', alpha=1, edgecolor='k', s=70)
        ax1.scatter(y_test, y_pred, c='r', label='Test Data', alpha=1, edgecolor='k', s=70)
        
        # 右图：带不确定性
        ax2.scatter(y_train, y_pred1, c='b', label='Train Data', alpha=0.6, edgecolor='k', s=70)
        scatter = ax2.scatter(y_test, y_pred, c=y_test_uncertainty, 
                             cmap='plasma', alpha=0.8, edgecolor='k', s=70, label='Test Data')
        
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Uncertainty (eV)', fontsize=16, weight='bold')
        
        axes = [ax1, ax2]
        titles = ['Standard Prediction', 'Prediction with Uncertainty']
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(y_train, y_pred1, c='b', label='Train Data', alpha=1, edgecolor='k', s=70)
        ax.scatter(y_test, y_pred, c='r', label='Test Data', alpha=1, edgecolor='k', s=70)
        axes = [ax]
        titles = ['Prediction Results']
    
    # 共同设置
    for i, ax in enumerate(axes):
        # 添加边框
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        # 添加理想线
        min_val = min(min(y_train), min(y_test), min(y_pred1), min(y_pred))
        max_val = max(max(y_train), max(y_test), max(y_pred1), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Ideal Fit: y = x')
        
        ax.set_xlabel('True (eV)', fontsize=32, weight='bold')
        ax.set_ylabel('Predicted (eV)', fontsize=32, weight='bold')
        ax.set_title(titles[i], fontsize=24, weight='bold')
        ax.legend(fontsize=16)
        ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    plt.tight_layout()
    return fig


# =============================================================================
# 以下是集成到你现有代码的示例
# =============================================================================

"""
将以下代码添加到你的现有notebook中：

# 在你现有的模型训练代码之后添加：

# 原来的代码：
# model = ExtraTreesRegressor(max_features=0.3, min_samples_split=3, n_estimators=179,
#                     n_jobs=-1, random_state=42)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# y_pred1 = model.predict(X_train)

# 新增：获取不确定性估计
test_predictions, test_uncertainties, test_intervals = add_uncertainty_to_extratrees(model, X_test)
train_predictions, train_uncertainties, train_intervals = add_uncertainty_to_extratrees(model, X_train)

# 打印不确定性信息
print(f"平均测试不确定性: {np.mean(test_uncertainties):.4f} eV")
print(f"不确定性范围: {np.min(test_uncertainties):.4f} - {np.max(test_uncertainties):.4f} eV")

# 计算95%置信区间覆盖率
coverage = np.mean((y_test >= test_intervals[0]) & (y_test <= test_intervals[1]))
print(f"95%置信区间覆盖率: {coverage:.2%}")

# 原来的性能计算保持不变
test_mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
r1 = r2_score(y_train, y_pred1)
print(f"R2_test= {r2:.2f}")
print(f"R2_train= {r1:.2f}")
print(f"RMSE_test: {np.sqrt(test_mse):.2f}")
print(f"RMSE_train: {np.sqrt(train_mse):.2f}")

# 使用改进的绘图函数
plot_with_uncertainty(y_train, y_test, y_pred1, y_pred, test_uncertainties)

# 或者继续使用原来的plot_figure函数：
# plot_figure(y_train=y_train, y_test=y_test, y_pred=y_pred, y_pred1=y_pred1)
"""
