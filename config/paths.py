"""集中路径配置（Centralised path configuration）
================================================================
本项目所有脚本通过本模块获取数据 / 模型 / 输出路径，
不再使用硬编码绝对路径，从而保证跨机器、跨平台可移植。

用法（在每个脚本顶部加入）::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from config.paths import NORMALIZED_DIR, MODELS_DIR, TRAIN_NORM_CSV, ...

约定
----
- 所有路径均相对 ``PROJECT_ROOT`` 推导，clone 仓库后即可运行。
- 大数据与模型权重默认放在 ``data/``（已在 .gitignore 中排除）。
- 历史遗留的 ``dataset_split_correct`` / ``06_normalize_*`` 命名一律
  统一到本模块的 ``06_normalized`` / ``05_normalize_*`` 基准。
"""

from pathlib import Path

# ════════════════════════════════════════════════════════════════
# 项目根目录（本文件位于 <root>/config/paths.py）
# ════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ── 数据根 ───────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data"

# 各阶段目录（与 README 流程图编号一致）
RAW_DIR        = DATA_DIR / "00_raw"            # 原始数据
CM_RECLASS_DIR = DATA_DIR / "01_cm_reclass_input"  # 汇聚边缘细分产出（输入起点）
FILTERED_DIR   = DATA_DIR / "02_filtered"       # 筛选后
COMBINED_DIR   = DATA_DIR / "03_combined"       # GEOROC + PetDB 合并
SPLIT_DIR      = DATA_DIR / "04_split"          # 训练/测试切分 + 按类型拆分 + IQR clean
IMPUTED_DIR    = DATA_DIR / "05_imputed"        # 按类别 MissForest 插补
NORMALIZED_DIR = DATA_DIR / "06_normalized"     # 主量无水标准化 + 分位归一化
MODELS_DIR     = DATA_DIR / "models"            # 训练产出权重 .pth
ARCHEAN_DIR    = DATA_DIR / "archean"           # 太古代应用数据 + 案例 + 输出

# ════════════════════════════════════════════════════════════════
# 00_raw —— 原始数据文件
# ════════════════════════════════════════════════════════════════
GEOROC_RAW_CSV = RAW_DIR / "georoc" / "basalt_2025.csv"
PETDB_RAW_CSV  = RAW_DIR / "petdb" / "petDB_recent_downloads_merged.csv"

# ════════════════════════════════════════════════════════════════
# 01_cm_reclass_input —— 汇聚边缘细分项目（convergent_margin_reclass）的产出
# 本项目不含该细分代码，仅以其产出作为输入起点
# ════════════════════════════════════════════════════════════════
CM_CORE_CSV          = CM_RECLASS_DIR / "georoc_convergent_margin_core_training_high_confidence.csv"
CM_EXPANDED_CSV      = CM_RECLASS_DIR / "georoc_convergent_margin_expanded_training_reviewed.csv"
REFINED_EXPANDED_CSV = CM_RECLASS_DIR / "basalt_refined_expanded.csv"  # 用细分结果重分类后的 GEOROC

# ════════════════════════════════════════════════════════════════
# 02_filtered —— 经筛选规则后的数据
# ════════════════════════════════════════════════════════════════
GEOROC_FILTERED_CSV = FILTERED_DIR / "basalt_refined_expanded_filtered.csv"
PETDB_FILTERED_CSV  = FILTERED_DIR / "petDB.csv"

# ════════════════════════════════════════════════════════════════
# 03_combined —— GEOROC + PetDB 合并
# ════════════════════════════════════════════════════════════════
COMBINED_CSV = COMBINED_DIR / "01_basalt_number_year.csv"

# ════════════════════════════════════════════════════════════════
# 04_split —— 训练/测试切分、按构造环境拆分、IQR clean
# ════════════════════════════════════════════════════════════════
TRAIN_RAW_CSV      = SPLIT_DIR / "01_basalt_number_year_train.csv"
TEST_RAW_CSV       = SPLIT_DIR / "01_basalt_number_year_test.csv"
SPLIT_SUMMARY_CSV  = SPLIT_DIR / "split_summary.csv"
SPLIT_BY_TYPE_DIR  = SPLIT_DIR / "preprocess" / "split"   # 按构造环境拆分的训练子集
CLEAN_DIR          = SPLIT_DIR / "preprocess" / "clean"   # IQR 去除离群后的 *_clean.csv

# ════════════════════════════════════════════════════════════════
# 05_imputed —— 按构造环境类别 MissForest 插补
# ════════════════════════════════════════════════════════════════
MISSFOREST_DIR    = IMPUTED_DIR / "MissForest"   # 各类别 *_clean_imputed.csv
TRAIN_IMPUTED_CSV = IMPUTED_DIR / "02_basalt_train_imputed.csv"
TEST_IMPUTED_CSV  = IMPUTED_DIR / "02_basalt_test_imputed.csv"

# ════════════════════════════════════════════════════════════════
# 06_normalized —— 主量元素无水标准化 + 分位数分箱归一化
# ════════════════════════════════════════════════════════════════
TRAIN_MAJOR_NORM_CSV = NORMALIZED_DIR / "03_basalt_train_major_normalize.csv"
TEST_MAJOR_NORM_CSV  = NORMALIZED_DIR / "03_basalt_test_major_normalize.csv"
TRAIN_NORM_CSV       = NORMALIZED_DIR / "05_normalize_basalt_train.csv"   # 最终喂模型/SHAP 的训练集
TEST_NORM_CSV        = NORMALIZED_DIR / "05_normalize_basalt_test.csv"    # 最终喂模型/SHAP 的测试集
QUANTILE_PARAMS_JSON = NORMALIZED_DIR / "quantile_params.json"            # 训练集分位参数（应用集共用）

# ════════════════════════════════════════════════════════════════
# models —— 训练产出权重
# ════════════════════════════════════════════════════════════════
GEODAN_DIR        = MODELS_DIR / "GeoDAN"
MAIN_MODEL_WEIGHT = MODELS_DIR / "Full_Model_(ViT+Transformer)_best_seed.pth"

# ════════════════════════════════════════════════════════════════
# archean —— 太古代应用（no_impute 模式）
# ════════════════════════════════════════════════════════════════
ARCHEAN_DATA_SUBDIR = ARCHEAN_DIR / "data"          # 太古代原始 CSV + 6 克拉通案例
ARCHEAN_S3_CSV      = ARCHEAN_DATA_SUBDIR / "archean_basalt.csv"
ARCHEAN_OUTPUT_DIR  = ARCHEAN_DIR / "outputs"        # 太古代预处理 / 预测输出

# ════════════════════════════════════════════════════════════════
# 论文图件输出
# ════════════════════════════════════════════════════════════════
FIGURES_DIR = DATA_DIR / "figures"
