import pandas as pd
import numpy as np
import os
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    TRAIN_MAJOR_NORM_CSV, TEST_MAJOR_NORM_CSV,
    NORMALIZED_DIR, QUANTILE_PARAMS_JSON,
)

# ============================================================
# 分位数分箱归一化（quantile binning, 1–255）
# 训练集 fit 分位数边界 → 保存 JSON → transform；测试集加载同一 JSON 直接 transform。
# 本脚本一次性先处理训练集、再 transform 测试集（fit 必先于 transform），无需切换运行模式。
# ============================================================

# -------------------- 路径配置 --------------------
# 训练集：已经过主量无水标准化的训练集
TRAIN_INPUT_FILE  = str(TRAIN_MAJOR_NORM_CSV)
TRAIN_OUTPUT_DIR  = str(NORMALIZED_DIR)
TRAIN_OUTPUT_FILE = "05_normalize_basalt_train.csv"    # 相对于 TRAIN_OUTPUT_DIR

# 测试集：已经过主量无水标准化的测试集
PREDICT_INPUT_FILE  = str(TEST_MAJOR_NORM_CSV)
PREDICT_OUTPUT_DIR  = str(NORMALIZED_DIR)
PREDICT_OUTPUT_FILE = "05_normalize_basalt_test.csv"   # 相对于 PREDICT_OUTPUT_DIR

# 分位数边界参数文件（训练时写入，测试 / 应用集读取，保证与训练分布对齐）
QUANTILE_PARAMS_FILE = str(QUANTILE_PARAMS_JSON)
# 36 个元素特征列（训练与预测必须完全一致）
COLUMNS_TO_EXTRACT = [
    "NA2O(WT%)", "MGO(WT%)", "AL2O3(WT%)", "SIO2(WT%)", "P2O5(WT%)", "K2O(WT%)",
    "CAO(WT%)", "TIO2(WT%)", "MNO(WT%)", "FEOT(WT%)", "RB(PPM)", "V(PPM)",
    "CR(PPM)", "CO(PPM)", "NI(PPM)", "BA(PPM)", "SR(PPM)", "Y(PPM)", "ZR(PPM)",
    "NB(PPM)", "LA(PPM)", "CE(PPM)", "PR(PPM)", "ND(PPM)", "SM(PPM)", "EU(PPM)",
    "GD(PPM)", "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "YB(PPM)", "LU(PPM)",
    "HF(PPM)", "TA(PPM)", "TH(PPM)"
]


# ============================================================
# 核心函数
# ============================================================

def fit_quantile_boundaries(data, columns):
    """
    训练阶段：从训练数据计算每列的 254 个分位数边界。

    返回:
        params: dict, {列名: [q1, q2, ..., q254]}
    """
    params = {}
    for col in columns:
        col_data = data[col].dropna().values
        sorted_col = np.sort(col_data)
        n = len(sorted_col)
        # 计算 254 个分位数边界，与原始逻辑完全一致
        quantiles = [
            float(sorted_col[int(np.round(n * j / 255))])
            for j in range(1, 255)
        ]
        params[col] = quantiles
    return params


def save_quantile_params(params, file_path):
    """将分位数边界保存为 JSON 文件。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)
    print(f"[TRAIN] 分位数边界已保存至: {file_path}")


def load_quantile_params(file_path):
    """从 JSON 文件加载训练阶段保存的分位数边界。"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[PREDICT] 找不到分位数参数文件: {file_path}\n"
            "请先以 MODE='train' 运行一次以生成参数文件。"
        )
    with open(file_path, "r", encoding="utf-8") as f:
        params = json.load(f)
    print(f"[PREDICT] 已加载分位数边界: {file_path}")
    return params


def transform_column(col_series, quantiles):
    """
    使用已有分位数边界对一列数据做映射（transform only，不重新 fit）。

    映射规则（与原始逻辑一致）:
        NaN        → 0
        val <= q1  → 1
        val > q254 → 255
        其余       → 落在对应区间 j+1
    """
    q = quantiles  # list of 254 floats

    def map_value(val):
        if pd.isna(val):
            return 0
        if val <= q[0]:
            return 1
        if val > q[-1]:
            return 255
        # 二分查找替代线性遍历，速度更快
        lo, hi = 0, len(q) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if val <= q[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo + 1   # lo 对应边界索引，映射值为 lo+1（范围 1~254）

    return col_series.map(map_value)


def apply_quantile_transform(data, columns, params):
    """
    对 DataFrame 的指定列应用分位数映射（仅 transform）。

    参数:
        data    : 输入 DataFrame
        columns : 需要映射的列名列表
        params  : fit_quantile_boundaries() 或 load_quantile_params() 返回的边界字典

    返回:
        normalized_data: 仅含映射后特征列 + TECTONIC SETTING 的 DataFrame
    """
    # 检查预测数据中是否存在训练时未见过的列（反向也检查）
    missing_in_predict = [c for c in columns if c not in data.columns]
    if missing_in_predict:
        raise ValueError(f"预测数据缺少以下特征列: {missing_in_predict}")

    missing_in_params = [c for c in columns if c not in params]
    if missing_in_params:
        raise ValueError(
            f"参数文件中缺少以下列的边界（请重新以 train 模式生成参数）: {missing_in_params}"
        )

    normalized_data = pd.DataFrame()
    for col in columns:
        normalized_data[col] = transform_column(data[col], params[col])

    if 'TECTONIC SETTING' in data.columns:
        normalized_data['TECTONIC SETTING'] = data['TECTONIC SETTING']
    else:
        print("[WARNING] 输入数据中未找到 'TECTONIC SETTING' 列，预测模式下正常，训练模式下请检查。")

    return normalized_data


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":

    # 第一步：训练集 fit 分位数边界 → 保存 JSON → transform 训练集
    if True:
        # --------------------------------------------------
        # 训练：fit 边界 → 保存 → transform → 输出 CSV
        # --------------------------------------------------
        print("=" * 50)
        print("第一步: 训练集 fit + transform")
        print("=" * 50)

        # 1. 读取训练数据（SMOTE 后的均衡数据集）
        data = pd.read_csv(TRAIN_INPUT_FILE)
        print(f"读取训练数据: {TRAIN_INPUT_FILE}  shape={data.shape}")

        # 2. Fit：从训练数据计算分位数边界
        print("正在计算分位数边界 ...")
        params = fit_quantile_boundaries(data, COLUMNS_TO_EXTRACT)

        # 3. 保存边界到 JSON（供预测阶段复用）
        save_quantile_params(params, QUANTILE_PARAMS_FILE)

        # 4. Transform：用刚计算出的边界映射训练数据
        normalized_data = apply_quantile_transform(data, COLUMNS_TO_EXTRACT, params)

        # 5. 保存输出
        os.makedirs(TRAIN_OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(TRAIN_OUTPUT_DIR, TRAIN_OUTPUT_FILE)
        normalized_data.to_csv(out_path, index=False)
        print(f"[TRAIN] 归一化训练数据已保存至: {out_path}")
        print(f"[TRAIN] 输出数据 shape={normalized_data.shape}")
        print("标准化转换成功")

    # 第二步：测试集加载已保存边界 → transform（绝不重新 fit，保证与训练分布对齐）
    if True:
        # --------------------------------------------------
        # 测试：加载已保存边界 → transform → 输出 CSV
        # --------------------------------------------------
        print("=" * 50)
        print("第二步: 测试集 transform")
        print("=" * 50)

        # 1. 加载训练阶段保存的分位数边界
        params = load_quantile_params(QUANTILE_PARAMS_FILE)

        # 2. 读取预测数据（已经过主量归一化，但未 SMOTE）
        data = pd.read_csv(PREDICT_INPUT_FILE)
        print(f"读取预测数据: {PREDICT_INPUT_FILE}  shape={data.shape}")

        # 3. Transform：仅用已加载的边界做映射，不重新计算
        normalized_data = apply_quantile_transform(data, COLUMNS_TO_EXTRACT, params)

        # 4. 统计超出训练分布的样品数量（辅助置信度判断）
        for col in COLUMNS_TO_EXTRACT:
            out_of_range = (normalized_data[col] == 255).sum() + (normalized_data[col] == 1).sum()
            if out_of_range > 0:
                pct = out_of_range / len(normalized_data) * 100
                print(f"[WARNING] {col}: {out_of_range} 个样品 ({pct:.1f}%) 落在训练分布边界 (值=1或255)")

        # 5. 保存输出
        os.makedirs(PREDICT_OUTPUT_DIR, exist_ok=True)
        out_path = os.path.join(PREDICT_OUTPUT_DIR, PREDICT_OUTPUT_FILE)
        normalized_data.to_csv(out_path, index=False)
        print(f"[PREDICT] 归一化预测数据已保存至: {out_path}")
        print(f"[PREDICT] 输出数据 shape={normalized_data.shape}")
        print("标准化转换成功")
