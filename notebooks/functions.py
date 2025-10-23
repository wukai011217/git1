#导入包
# def import_code ():
#     import pandas as pd
#     from sklearn.model_selection import train_test_split, GridSearchCV
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.pipeline import Pipeline
#     from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
#     from xgboost import XGBRegressor, XGBClassifier
#     import matplotlib.pyplot as plt
#     import seaborn as sns
#     import numpy as np
#     import xgboost as xgb
#     from sklearn.feature_selection import mutual_info_regression
#     from sklearn.linear_model import LinearRegression, Ridge
#     from sklearn.svm import SVR
#     from sklearn.tree import DecisionTreeRegressor
#     from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#     from sklearn.neighbors import KNeighborsRegressor

#绘图
def plot_figure(y_train=None, y_test=None,y_pred1=None,y_pred=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error
    # 创建图形
    # 设置全局字体参数 - 使用Arial字体并加大加粗
    plt.rcParams['font.family'] = 'Arial'  # 设置字体为Arial
    plt.rcParams['font.weight'] = 'bold'  # 全局设置字体为粗体
    plt.rcParams['axes.titleweight'] = 'bold'  # 坐标轴标题加粗
    plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签加粗
    plt.rcParams['font.size'] = 24  # 全局设置字体大小
    plt.rcParams['axes.linewidth'] = 1.5  # 增加坐标轴线宽
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 添加边框
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
        spine.set_color('black')

    # 训练集预测散点
    ax.scatter(y_train, y_pred1, c='b', label='Train Data', alpha=1, edgecolor='k', s=70)
    # 测试集预测散点
    ax.scatter(y_test, y_pred, c='r', label='Test Data', alpha=1, edgecolor='k', s=70)

    # 添加 y = x 的对角线 (提示完美拟合的参考线)
    min_val = min(min(y_train), min(y_test), min(y_pred1), min(y_pred))
    max_val = max(max(y_train), max(y_test), max(y_pred1), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'g--', linewidth=2, label='Ideal Fit: y = x')
    
    # 计算测试集的R²和RMSE
    test_r2 = r2_score(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # 设置标题和标签
    ax.set_xlabel('True (eV)', fontsize=36,weight='bold')
    ax.set_ylabel('Predicted (eV)', fontsize=36,weight='bold')

    # 设置网格线
    ax.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
    
    # 创建自定义图例，包含R²和RMSE值
    # 首先获取原始的图例内容（训练数据、测试数据和理想拟合线）
    handles, labels = ax.get_legend_handles_labels()
    
    # 添加R²和RMSE到图例中
    # 创建空对象作为占位符和文本标签
    handles.append(plt.Line2D([0], [0], color='white', alpha=0))  # 透明线作为占位符
    labels.append(f'Test R² = {test_r2:.2f}')
    
    handles.append(plt.Line2D([0], [0], color='white', alpha=0))  # 透明线作为占位符
    labels.append(f'Test RMSE = {test_rmse:.2f}')
    
    # 添加图例与调整字体大小
    ax.legend(handles=handles, labels=labels, fontsize=28, loc='lower right')

    # 调整坐标轴刻度字体
    ax.tick_params(axis='both', which='major', labelsize=26)

    # 展示图形
    plt.tight_layout()
    plt.show()


#模型训练
def model_train(x, y):
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

#模型测试
def model_test(x, y):
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

#模型分析
def model_analyze(model=None, X_train=None):
    import shap
    import matplotlib.pyplot as plt

    # 确保 X_train 是 Pandas DataFrame，列名应为物理/化学意义的特征名

    # Step 1: 创建 SHAP 解释器
    explainer = shap.Explainer(model, X_train)  # 使用 SHAP Explainer 解释训练模型

    # Step 2: 计算 SHAP 值
    shap_values = explainer(X_train)

    # Step 3: 绘制 SHAP 特征重要性柱状图 (Top 10)
    # 注意：shap.summary_plot 自动处理图形，无需调用 plt.figure()
    plt.title('Top 10 Features Impacting Adsorption Energy', fontsize=16, weight='bold')
    plt.xlabel('Mean Absolute SHAP Value (eV)', fontsize=14)
    plt.ylabel('Feature Names', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    shap.summary_plot(shap_values, X_train, plot_type="bar", max_display=10)

    # Step 4: 绘制标准特征重要性详细图
    # 注意：shap.summary_plot 自动处理图形，无需调用 plt.figure()
    plt.title('Feature Contributions to Adsorption Energy Prediction', fontsize=16, weight='bold')
    plt.xlabel('Feature Impact on Model Output (eV)', fontsize=14)
    plt.ylabel('Features Ordered by Importance', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    shap.summary_plot(shap_values, X_train, max_display=5, cmap='coolwarm')  # 默认显示20个特征

    # Step 5:（可选）绘制单一样本的 SHAP 决策可视化
    # 注意：shap.plots.waterfall 自动处理图形，无需调用 plt.figure()
    plt.title('SHAP Waterfall Plot for a Single Adsorption Prediction', fontsize=16, weight='bold')
    plt.tight_layout()
    shap.plots.waterfall(shap_values[1])

#数据统计
def data_analyze(correlation_matrix=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(figsize=(12, 8))
    # 绘制热图
    sns.heatmap(
        correlation_matrix,
        annot=True,          # 在热图中显示相关系数值
        annot_kws={"size": 10},  # 调整文字大小
        cmap="coolwarm",     # 设置颜色映射，适合化学领域
        fmt=".2f",           # 保留两位小数
        cbar=True,           # 显示颜色条
        cbar_kws={"shrink": 0.8},  # 调整颜色条的大小
        square=True,         # 将每个单元格绘制为正方形，提高美观
        linewidths=0.5,      # 单元格之间的网格线宽度
        linecolor="gray",    # 网格线颜色
    )
    # 添加标题和轴标签
    plt.title('Heatmap of Feature Correlations with Adsorption Energy', fontsize=14, weight='bold')
    plt.xticks(rotation=45, fontsize=13)  # 调整 x 轴标签角度
    plt.yticks(fontsize=13)  # 调整 y 轴标签字体大小

    # 显示图形
    plt.tight_layout()  # 自动调整布局以防止标签重叠
    plt.show()

#pca分析
def pca_analyze(x, y):
    from sklearn.model_selection import train_test_split