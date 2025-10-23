import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.pipeline import Pipeline
import time
import warnings
warnings.filterwarnings('ignore')
# matplotlib默认参数
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']  # 用于显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号



# 1. 数据加载
data_path = '/Users/wukai/Desktop/project/wjob/data/fea/final/reduced_dataset_90pct.csv'
data = pd.read_csv(data_path)

# 2. 数据预处理
# 删除包含缺失值的行
data = data.dropna()

# 分离特征和目标变量
X = data.drop(['element', 'structure_type', 'H2_adsorption_energy'], axis=1)
y = data['H2_adsorption_energy']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=50
)

# 3. 定义评估函数
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 计算指标
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    
    # 计算训练时间
    train_time = time.time() - start_time
    
    return {
        'model': model.__class__.__name__,
        'model_instance': model,  # 添加训练好的模型实例
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_time': train_time
    }

# 4. 定义模型列表
models = [
    # 线性模型
    ('Linear Regression', LinearRegression()),
    ('Ridge', Ridge(alpha=1.0, random_state=42)),
    ('Lasso', Lasso(alpha=0.1, random_state=42)),
    ('ElasticNet', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
    
    # 树模型
    ('Random Forest', RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
    ('Extra Trees', ExtraTreesRegressor(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)),
    ('Gradient Boosting', GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)),
    ('AdaBoost', AdaBoostRegressor(n_estimators=50, learning_rate=1.0, random_state=42)),
    
    # 其他集成模型
    ('XGBoost', XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, n_jobs=-1)),
    ('LightGBM', LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, n_jobs=-1)),
    ('CatBoost', CatBoostRegressor(iterations=100, learning_rate=0.1, depth=3, random_seed=42, verbose=0, thread_count=-1)),
    
    # 其他模型
    ('SVR', SVR(kernel='rbf', C=1.0, epsilon=0.1)),
    ('KNN', KNeighborsRegressor(n_neighbors=5, weights='uniform', n_jobs=-1))
]

# 5. 训练和评估所有模型
results = []
for name, model in models:
    print(f"正在训练 {name}...")
    try:
        result = evaluate_model(model, X_train, X_test, y_train, y_test)
        results.append(result)
        print(f"{name} 训练完成. 测试集 R² = {result['test_r2']:.4f}, RMSE = {result['test_rmse']:.4f}")
    except Exception as e:
        print(f"训练 {name} 时出错: {str(e)}")
        continue

# 6. 结果展示
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('test_r2', ascending=False)
print("\n模型性能比较 (按测试集 R² 排序):")
print(results_df[['model', 'test_r2', 'test_rmse', 'test_mae', 'train_time']].to_string(index=False))

# 7. 可视化结果
plt.figure(figsize=(15, 8))

# 绘制 R² 比较图
plt.subplot(1, 2, 1)
sns.barplot(x='test_r2', y='model', data=results_df, palette='viridis')
plt.title('模型比较 (测试集 R²)')
plt.xlabel('R² 分数')
plt.ylabel('模型')

# 绘制 RMSE 比较图
plt.subplot(1, 2, 2)
sns.barplot(x='test_rmse', y='model', data=results_df, palette='viridis')
plt.title('模型比较 (测试集 RMSE)')
plt.xlabel('RMSE')
plt.ylabel('')

plt.tight_layout()
plt.show()

# 8. 最佳模型分析
best_model_name = results_df.iloc[0]['model']

# 获取训练好的最佳模型实例
# 注意：需要在原始结果列表中找到正确的模型实例
best_trained_model = None
for result in results:
    if result['model'] == best_model_name:
        best_trained_model = result['model_instance']
        break

if best_trained_model is None:
    print(f"\n错误：无法找到最佳模型 '{best_model_name}' 的实例")
    exit(1)

print(f"\n最佳模型: {best_model_name}")

# 9. 保存最佳模型
import os
import joblib
from datetime import datetime

# 创建保存模型的目录
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models')
os.makedirs(model_dir, exist_ok=True)

# 生成文件名（包含模型名称和时间戳）
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"{best_model_name}_{timestamp}.joblib"
model_path = os.path.join(model_dir, model_filename)

# 保存训练好的模型
joblib.dump(best_trained_model, model_path)
print(f"训练好的模型已保存到: {model_path}")

# 保存模型元数据（可选）
model_info = {
    'model_name': best_model_name,
    'test_r2': results_df.iloc[0]['test_r2'],
    'test_rmse': results_df.iloc[0]['test_rmse'],
    'test_mae': results_df.iloc[0]['test_mae'],
    'train_time': results_df.iloc[0]['train_time'],
    'timestamp': timestamp
}

# 保存模型信息
info_filename = f"{best_model_name}_{timestamp}_info.joblib"
info_path = os.path.join(model_dir, info_filename)
joblib.dump(model_info, info_path)
print(f"模型信息已保存到: {info_path}")

