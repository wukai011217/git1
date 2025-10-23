"""
全局配置文件，用于存储项目相关的默认参数。
"""

import os

# 默认计算目录位于tests目录下
DEFAULT_CALC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests", "calc")
# 默认模板目录位于tests目录下
DEFAULT_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "tests", "templates")
# POTCAR文件目录位于data/pot目录下
DEFAULT_POTCAR_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "pot")

#-----------------------------------------------------------------------------------
# VASP计算相关默认参数
DEFAULT_NODES = 1  # 默认节点数
DEFAULT_TASKS_PER_NODE = 24  # 每节点默认任务数
DEFAULT_TIME = "24:00:00"  # 默认时间限制
DEFAULT_VASP_CMD = "mpirun -np $SLURM_NTASKS vasp_std"  # 默认VASP命令

# INCAR默认参数
DEFAULT_INCAR_PARAMS = {
    "SYSTEM": "Default",
    "ISTART": "0",
    "ENCUT": "400",
    "EDIFF": "1E-5",
    "ISMEAR": "0",
    "SIGMA": "0.05",
    "NSW": "0"
}

#结构对应命名
DEFAULT_STRUCTURE_NAME = {
    "None" : "pristine",
    "Fir-1" : "Ov-surf1",
    "Fir-2" : "Ov-surf2",
    "Sec-1" : "Ov-sub1",
    "Sec-2" : "Ov-sub2",
}
    

# KPOINTS默认参数
DEFAULT_KPOINTS = [2, 2, 1]  # 默认K点网格
DEFAULT_KPOINTS_SCHEME = "Monkhorst-Pack"  # 默认K点方案

# SLURM相关默认参数
DEFAULT_SLURM_PARTITION = "normal"  # 默认分区
DEFAULT_SLURM_ACCOUNT = None  # 默认账户

# 机器学习相关默认参数
DEFAULT_ML_MODEL = "extra_tree"  # 默认模型
DEFAULT_CV_FOLDS = 5  # 默认交叉验证折数

# matplotlib默认参数
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']  # 用于显示中文
matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# SOAP特征提取相关参数
SOAP_PARAMS = {
    'r_cut': 6.0,       # 截断半径（埃）
    'n_max': 4,         # 基函数径向最大量子数
    'l_max': 4,         # 基函数角动量最大量子数
    'sigma': 0.3,       # 高斯宽度参数
    'periodic': True,   # 是否使用周期性边界条件
    'sparse': False     # 是否返回稀疏表示
}

# SOAP可视化相关参数
SOAP_VISUALIZATION = {
    'max_structures': 5,  # 每种结构类型最多显示的结构数量
    'output_dir': 'results/soap_analysis',  # 默认输出目录
    'save_pca': True,     # 是否保存PCA可视化
    'pca_components': 2    # PCA降维维度
}

# 结构类型映射
STRUCTURE_TYPE_MAPPING = {
    'M': '纯金属',
    'M-H': '金属-单氢',
    'M-2H': '金属-双氢'
}

# 金属和吸附原子默认设置
DEFAULT_METAL_SYMBOL = 'Ag'    # 默认金属元素符号
DEFAULT_ADSORBATE_SYMBOL = 'H'  # 默认吸附剂元素符号
