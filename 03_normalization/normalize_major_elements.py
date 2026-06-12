import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    TRAIN_IMPUTED_CSV, TEST_IMPUTED_CSV,
    TRAIN_MAJOR_NORM_CSV, TEST_MAJOR_NORM_CSV,
)

# ============================================================
# 一次运行同时处理训练集和测试集。
# 主量无水标准化是逐行计算，不依赖全局拟合参数，不存在数据泄露。
# ============================================================

# -------------------- 路径配置 --------------------
# 训练集：插补后、SMOTE 前
TRAIN_INPUT_FILE  = str(TRAIN_IMPUTED_CSV)
TRAIN_OUTPUT_FILE = str(TRAIN_MAJOR_NORM_CSV)

# 测试集：插补后，不执行 SMOTE
TEST_INPUT_FILE  = str(TEST_IMPUTED_CSV)
TEST_OUTPUT_FILE = str(TEST_MAJOR_NORM_CSV)

# 10 个主量元素（训练与预测完全一致）
MAJOR_ELEMENTS = [
    "NA2O(WT%)", "MGO(WT%)", "AL2O3(WT%)", "SIO2(WT%)",
    "P2O5(WT%)", "K2O(WT%)", "CAO(WT%)", "TIO2(WT%)",
    "MNO(WT%)", "FEOT(WT%)"
]


# ============================================================
# 核心函数
# ============================================================

def normalize_major_elements(df):
    """
    无水归一化：将 10 个主量元素按行归一化到总和为 100%。

    该变换是纯行内运算，不依赖全局统计量，训练与预测使用完全
    相同的公式，不存在数据泄露问题。

    参数:
        df: DataFrame，需包含 MAJOR_ELEMENTS 中的所有列

    返回:
        df_normalized: 归一化后的 DataFrame（其余列不变）
    """
    missing_cols = [c for c in MAJOR_ELEMENTS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"输入数据缺少以下主量元素列: {missing_cols}")

    df_normalized = df.copy()
    major_values = df_normalized[MAJOR_ELEMENTS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_value_mask = major_values.isna() | ~np.isfinite(major_values)
    if invalid_value_mask.any().any():
        invalid_rows = int(invalid_value_mask.any(axis=1).sum())
        raise ValueError(f"存在 {invalid_rows} 行主量元素缺失或为非有限值")

    row_total = major_values.sum(axis=1)
    invalid_total_mask = row_total <= 0
    if invalid_total_mask.any():
        invalid_rows = int(invalid_total_mask.sum())
        raise ValueError(f"存在 {invalid_rows} 行主量元素总和不大于0")

    df_normalized.loc[:, MAJOR_ELEMENTS] = major_values.div(
        row_total,
        axis=0,
    ) * 100

    return df_normalized


def print_normalization_stats(df_original, df_normalized):
    """打印归一化前后主量总和的统计信息，便于验证。"""
    original_sum   = df_original[MAJOR_ELEMENTS].sum(axis=1)
    normalized_sum = df_normalized[MAJOR_ELEMENTS].sum(axis=1)

    print("\n主量归一化统计:")
    print(f"  原始数据行总和   —— mean: {original_sum.mean():.4f}  "
          f"std: {original_sum.std():.4f}  "
          f"min: {original_sum.min():.4f}  max: {original_sum.max():.4f}")
    print(f"  归一化后行总和   —— mean: {normalized_sum.mean():.4f}  "
          f"std: {normalized_sum.std():.6f}  "
          f"min: {normalized_sum.min():.4f}  max: {normalized_sum.max():.4f}")
    print("  （归一化后所有行总和应严格等于 100，std 应约为 0）")


def process_file(input_path, output_path, mode_label):
    """
    读取 → 归一化 → 保存，同时输出统计信息。

    参数:
        input_path  : str, 输入 CSV 路径
        output_path : str, 输出 CSV 路径
        mode_label  : str, 日志前缀（"TRAIN" 或 "TEST"）
    """
    # 1. 读取
    df = pd.read_csv(input_path)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    print(f"[{mode_label}] 读取数据: {input_path}  shape={df.shape}")

    # 2. 归一化
    df_normalized = normalize_major_elements(df)

    # 3. 统计
    print_normalization_stats(df, df_normalized)

    # 4. 保存
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df_normalized.to_csv(output_path, index=False)
    print(f"[{mode_label}] 主量归一化结果已保存至: {output_path}")


# ============================================================
# 主流程：一次完成训练集和测试集
# ============================================================

def main():
    """依次对训练集和测试集执行相同的主量无水标准化。"""
    print("=" * 60)
    print("训练集与测试集主量元素无水标准化")
    print("=" * 60)

    process_file(
        TRAIN_INPUT_FILE,
        TRAIN_OUTPUT_FILE,
        mode_label="TRAIN",
    )
    print()
    process_file(
        TEST_INPUT_FILE,
        TEST_OUTPUT_FILE,
        mode_label="TEST",
    )

    print("\n[完成] 训练集与测试集主量无水标准化全部完成")


if __name__ == "__main__":
    main()
