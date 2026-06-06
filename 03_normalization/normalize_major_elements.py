import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    TRAIN_IMPUTED_CSV, TRAIN_MAJOR_NORM_CSV,
    TEST_IMPUTED_CSV, TEST_MAJOR_NORM_CSV,
)

# ============================================================
# 主量无水归一化：逐行计算（每行除以该行 10 个主量元素之和 × 100），
# 无需全局拟合参数，训练与测试使用完全相同的公式，不存在数据泄露风险。
# 因此本脚本一次性先处理训练集、再直接 transform 测试集，无需切换运行模式。
# ============================================================

# -------------------- 路径配置 --------------------
# 训练集：插补后合并文件
TRAIN_INPUT_FILE  = str(TRAIN_IMPUTED_CSV)
TRAIN_OUTPUT_FILE = str(TRAIN_MAJOR_NORM_CSV)

# 测试集：插补后文件（未经 IQR clean）
PREDICT_INPUT_FILE  = str(TEST_IMPUTED_CSV)
PREDICT_OUTPUT_FILE = str(TEST_MAJOR_NORM_CSV)

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
    row_total = df_normalized[MAJOR_ELEMENTS].sum(axis=1)

    # 检查行总和为 0 的异常行（通常意味着所有主量均缺失）
    zero_total_mask = row_total == 0
    if zero_total_mask.any():
        n_zero = zero_total_mask.sum()
        print(f"[WARNING] {n_zero} 行的主量元素总和为 0，归一化后将产生 NaN，请检查插补结果。")

    for element in MAJOR_ELEMENTS:
        df_normalized[element] = df_normalized[element] / row_total * 100

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
        mode_label  : str, 日志前缀（"TRAIN" 或 "PREDICT"）
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
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_normalized.to_csv(output_path, index=False)
    print(f"[{mode_label}] 主量归一化结果已保存至: {output_path}")


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    # 主量无水归一化为逐行运算，训练与测试公式完全一致，
    # 一次性先处理训练集、再 transform 测试集，无需切换运行模式。
    print("=" * 50)
    print("主量无水归一化：训练集 + 测试集一次性处理")
    print("=" * 50)
    process_file(TRAIN_INPUT_FILE, TRAIN_OUTPUT_FILE, mode_label="TRAIN")
    process_file(PREDICT_INPUT_FILE, PREDICT_OUTPUT_FILE, mode_label="PREDICT")

    print("主量归一化完成")
