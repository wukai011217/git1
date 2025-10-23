from setuptools import setup, find_packages

setup(
    name="wjob",
    version="0.1.0",
    description="催化剂研究辅助工具集",
    author="",
    author_email="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "xgboost",  # 常用于特征重要性分析
        "shap",     # 用于模型解释
        "ase",      # 原子模拟环境
        # 根据实际需要添加其他依赖
    ],
    python_requires=">=3.7",
)
