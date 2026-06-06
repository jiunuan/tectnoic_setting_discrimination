"""
==========================================================================
玄武岩地球化学缺失值插补 —— 训练 + 测试集一体化脚本（MissForest）
==========================================================================
由 imputation_train.py（训练集插补 + 存模型）与 imputation_predict.py
（加载模型对测试集 transform 插补）合并而来。

核心改造：
  - 训练阶段为每个构造环境类别 fit 一套 MissForest（imputers + scaler +
    column_order），**仅保留在内存中，不再保存 .joblib 模型权重**。
  - 训练完立即用同一套内存模型对测试集做插补，省去存盘/读盘环节。

流程：
  阶段 1  训练集（clean 目录，按类别分文件）逐个 fit MissForest，
          同时对训练集自身做全量插补并输出。
  阶段 2  测试集（合并 CSV，含 TECTONIC SETTING）按类别匹配内存模型，
          仅 scaler.transform / RF.predict（不重新 fit），插补并输出。

训练-预测一致性原则：
  1. 测试数据不做 IQR 异常值清洗（与原预测脚本一致）。
  2. 标准化用训练阶段 fit 好的 scaler（只 transform）。
  3. 插补用训练阶段训练好的 RF（只 predict）。
  4. 逆标准化用同一个 scaler.inverse_transform。
==========================================================================
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import CLEAN_DIR, IMPUTED_DIR, TEST_RAW_CSV, TEST_IMPUTED_CSV


# ============================================================================
# 默认配置（直接修改这里即可）
# ============================================================================

# 训练集输入目录：按构造环境分类的 clean CSV 文件所在目录
TRAIN_INPUT_DIR = str(CLEAN_DIR)

# 训练集插补结果输出目录（结果落在 <该目录>\MissForest\ 下）
TRAIN_OUTPUT_DIR = str(IMPUTED_DIR)

# 测试集输入文件：合并后的原始数据（未经 IQR clean），含 TECTONIC SETTING 列
TEST_INPUT_CSV = str(TEST_RAW_CSV)

# 测试集插补结果输出文件
TEST_OUTPUT_CSV = str(TEST_IMPUTED_CSV)

# 临时开关：只跑单个训练文件；设为 None 可恢复处理全部 CSV
TEMP_ONLY_FILE = None

# RandomForest 超参数（与原训练脚本一致）
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=24,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
)

# 36 个需要插补的地球化学元素列
CHEMICAL_COLUMNS = [
    'NA2O(WT%)', 'MGO(WT%)', 'AL2O3(WT%)', 'SIO2(WT%)', 'P2O5(WT%)',
    'K2O(WT%)', 'CAO(WT%)', 'TIO2(WT%)', 'MNO(WT%)', 'FEOT(WT%)',
    'RB(PPM)', 'V(PPM)', 'CR(PPM)', 'CO(PPM)', 'NI(PPM)', 'BA(PPM)',
    'SR(PPM)', 'Y(PPM)', 'ZR(PPM)', 'NB(PPM)', 'LA(PPM)', 'CE(PPM)',
    'PR(PPM)', 'ND(PPM)', 'SM(PPM)', 'EU(PPM)', 'GD(PPM)', 'TB(PPM)',
    'DY(PPM)', 'HO(PPM)', 'ER(PPM)', 'YB(PPM)', 'LU(PPM)', 'HF(PPM)',
    'TA(PPM)', 'TH(PPM)'
]

# 需要原样保留的非化学列
PRESERVED_COLUMNS = ['TECTONIC SETTING', 'longitude', 'latitude', 'AGE']


# ============================================================================
# 数据预处理
# ============================================================================

def preprocess_chemical_data(
    df: pd.DataFrame,
    columns: List[str],
    missing_row_threshold: float = 0.8
) -> pd.DataFrame:
    """
    对化学元素列做基础预处理：
    1. 提取目标列并强制转为数值型
    2. 丢弃缺失比例超过阈值的行（默认 >80% 缺失即丢弃）
    !! 不做 IQR 异常值清洗 !!
    """
    df_out = df[columns].copy()
    df_out = df_out.apply(pd.to_numeric, errors='coerce')
    min_non_null = int(len(columns) * (1 - missing_row_threshold))
    df_out = df_out.dropna(thresh=min_non_null)
    return df_out


def clean_for_rf(X: pd.DataFrame) -> pd.DataFrame:
    """
    为 RandomForest 做数据清洗（仅作特征输入）：
    - ±inf -> float32 最大值
    - NaN  -> 0
    """
    X_clean = X.replace([np.inf, -np.inf], np.finfo(np.float32).max)
    return X_clean.fillna(0)


def normalize_setting_name(name: str, strip_clean_suffix: bool = False) -> str:
    """
    统一类别名格式，便于将测试集 TECTONIC SETTING 与训练文件名匹配。
    训练文件名形如 'mid_ocean_ridge_clean'，去掉 _clean 后与标签对齐。
    """
    normalized = str(name).strip()
    normalized = re.sub(r'\s+', '_', normalized)
    normalized = re.sub(r'_+', '_', normalized)
    if strip_clean_suffix:
        normalized = re.sub(r'_clean$', '', normalized, flags=re.IGNORECASE)
    return normalized.lower()


# ============================================================================
# MissForest 核心（fit / transform）
# ============================================================================

def missforest_fit(
    X_train: pd.DataFrame,
    rf_params: Optional[Dict] = None
) -> Tuple[Dict[str, RandomForestRegressor], List[str]]:
    """
    MissForest 训练：为每个特征列训练一个 RandomForestRegressor。
    对于第 j 列，用其余 35 列作为特征、第 j 列的非空值作为标签训练 RF。
    """
    if rf_params is None:
        rf_params = RF_PARAMS

    X_clean = clean_for_rf(X_train)
    imputers: Dict[str, RandomForestRegressor] = {}

    for col in X_clean.columns:
        feature_cols = [c for c in X_clean.columns if c != col]
        mask = X_train[col].notna()
        if mask.sum() == 0:
            print(f"    [跳过] 列 '{col}' 全空，无法训练")
            continue

        rf = RandomForestRegressor(**rf_params)
        rf.fit(X_clean.loc[mask, feature_cols], X_train.loc[mask, col])
        imputers[col] = rf

    return imputers, list(X_clean.columns)


def missforest_transform(
    X: pd.DataFrame,
    imputers: Dict[str, RandomForestRegressor],
    column_order: List[str],
    verbose: bool = False
) -> pd.DataFrame:
    """用已训练的 RF 模型填补缺失值（仅 transform，不 fit）"""
    X = X[column_order].copy()
    X_clean = clean_for_rf(X)
    X_imputed = X_clean.copy()

    imputed_count = 0
    for col in column_order:
        feature_cols = [c for c in column_order if c != col]
        missing = X[col].isnull()
        if missing.any() and col in imputers:
            X_imputed.loc[missing, col] = imputers[col].predict(
                X_clean.loc[missing, feature_cols]
            )
            imputed_count += int(missing.sum())

    if verbose:
        print(f"    [插补] 共填补 {imputed_count} 个缺失值")
    return X_imputed


# ============================================================================
# 阶段 1：训练集逐类别训练 + 训练集自身插补输出
# ============================================================================

def train_and_impute_trainset(
    train_input_dir: str,
    train_output_dir: str,
) -> Dict[str, Dict]:
    """
    扫描训练集 clean 目录，逐个类别文件：
      1. 预处理 + 标准化（fit scaler）
      2. fit MissForest（内存保留，不落盘）
      3. 对训练集自身做全量插补 → 逆标准化 → 输出

    返回内存模型字典：
        { 标准化类别名 -> {'imputers', 'column_order', 'scaler', 'file_name'} }
    """
    csv_files = sorted([
        os.path.join(train_input_dir, f)
        for f in os.listdir(train_input_dir)
        if f.endswith('.csv')
    ])

    if TEMP_ONLY_FILE is not None:
        csv_files = [f for f in csv_files
                     if os.path.basename(f) == TEMP_ONLY_FILE]
        print(f"[临时] 仅处理训练文件: {TEMP_ONLY_FILE}")

    if not csv_files:
        raise FileNotFoundError(
            f"在 {train_input_dir} 下未找到 CSV 文件"
            f"{'（指定文件不存在）' if TEMP_ONLY_FILE else ''}"
        )

    print(f"[训练输入] 共找到 {len(csv_files)} 个 CSV 文件\n")

    method_out_dir = os.path.join(train_output_dir, 'MissForest')
    os.makedirs(method_out_dir, exist_ok=True)

    trained_models: Dict[str, Dict] = {}

    for file_path in csv_files:
        file_name = os.path.basename(file_path).replace('.csv', '')
        setting_key = normalize_setting_name(file_name, strip_clean_suffix=True)
        tectonic_setting = file_name.replace('_', ' ').upper()

        print(f"{'=' * 70}")
        print(f"[训练] {tectonic_setting}  ({os.path.basename(file_path)})")
        print(f"{'=' * 70}")

        data = pd.read_csv(file_path)

        # 提取化学数据并预处理
        chemical_data = preprocess_chemical_data(data, CHEMICAL_COLUMNS)
        available_preserved = [c for c in PRESERVED_COLUMNS if c in data.columns]
        preserved_data = data[available_preserved].loc[chemical_data.index]

        # 标准化（fit）
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(chemical_data),
            columns=chemical_data.columns,
            index=chemical_data.index
        )

        # 训练 MissForest（内存保留）
        print("  [MissForest] 训练 36 个 RF 模型...")
        imputers, col_order = missforest_fit(X_scaled)
        trained_models[setting_key] = {
            'imputers': imputers,
            'column_order': col_order,
            'scaler': scaler,
            'file_name': file_name,
        }

        # 训练集自身全量插补
        X_full_imputed = missforest_transform(X_scaled, imputers, col_order, verbose=True)

        # 逆标准化回原始量纲
        X_original = pd.DataFrame(
            scaler.inverse_transform(X_full_imputed),
            columns=chemical_data.columns,
            index=X_full_imputed.index
        )

        # 拼接保留列并输出
        final = pd.concat([
            preserved_data.reset_index(drop=True),
            X_original.reset_index(drop=True)
        ], axis=1)

        out_path = os.path.join(method_out_dir, f"{file_name}_imputed.csv")
        final.to_csv(out_path, index=False)
        print(f"  [训练集输出] {out_path}\n")

    return trained_models


# ============================================================================
# 阶段 2：测试集按类别匹配内存模型插补
# ============================================================================

def impute_testset(
    test_input_csv: str,
    test_output_csv: str,
    trained_models: Dict[str, Dict],
) -> pd.DataFrame:
    """
    对含 TECTONIC SETTING 标签的测试集，按类别匹配内存模型插补：
      1. 按 TECTONIC SETTING 分组
      2. 每组按标准化类别名查找内存模型
      3. scaler.transform → missforest_transform → scaler.inverse_transform
      4. 合并所有组并输出
    """
    print(f"{'=' * 70}")
    print("[测试集] 按 TECTONIC SETTING 匹配内存模型插补")
    print("  !! 不做 IQR 异常值清洗；scaler / RF 均只 transform，不重新 fit !!")
    print(f"{'=' * 70}")

    df = pd.read_csv(test_input_csv)
    print(f"[测试输入] 读取 {len(df)} 条样品 × {df.shape[1]} 列")

    if 'TECTONIC SETTING' not in df.columns:
        raise ValueError("测试数据中缺少 'TECTONIC SETTING' 列")

    # 缺失概览
    chem_present = [c for c in CHEMICAL_COLUMNS if c in df.columns]
    miss_pct = df[chem_present].isnull().mean()
    print(f"[缺失] 各元素列平均缺失率: {miss_pct.mean():.2%}")

    print(f"\n[内存模型] 可用类别: {sorted(trained_models.keys())}\n")

    results = []

    for setting, group in df.groupby('TECTONIC SETTING'):
        setting_key = normalize_setting_name(setting)
        print(f"[类别] {setting} ({len(group)} 条样品)")

        if setting_key not in trained_models:
            print(f"  [警告] 未找到匹配模型 '{setting_key}'，保留原始数据")
            results.append(group)
            continue

        bundle = trained_models[setting_key]
        imputers = bundle['imputers']
        col_order = bundle['column_order']
        scaler: StandardScaler = bundle['scaler']
        print(f"  [匹配] 使用训练模型 {bundle['file_name']}")

        # 预处理（不做 IQR clean）
        chem_data = preprocess_chemical_data(group, CHEMICAL_COLUMNS)
        available_preserved = [c for c in PRESERVED_COLUMNS if c in group.columns]
        preserved = group[available_preserved].loc[chem_data.index]

        # 标准化：用训练阶段的 scaler（只 transform）
        X_scaled = pd.DataFrame(
            scaler.transform(chem_data),
            columns=chem_data.columns,
            index=chem_data.index
        )

        # 插补
        X_imputed = missforest_transform(X_scaled, imputers, col_order, verbose=True)

        # 逆标准化回原始量纲
        X_original = pd.DataFrame(
            scaler.inverse_transform(X_imputed),
            columns=chem_data.columns,
            index=X_imputed.index
        )

        # 拼接保留列
        final = pd.concat([
            preserved.reset_index(drop=True),
            X_original.reset_index(drop=True)
        ], axis=1)
        results.append(final)

    result = pd.concat(results, ignore_index=True)

    out_dir = os.path.dirname(test_output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    result.to_csv(test_output_csv, index=False)

    print(f"\n[测试集输出] {test_output_csv}")
    print(f"[测试集输出] {len(result)} 条 × {result.shape[1]} 列")

    # 验证残留 NaN
    chem_present_out = [c for c in CHEMICAL_COLUMNS if c in result.columns]
    remaining = int(result[chem_present_out].isnull().sum().sum())
    if remaining > 0:
        print(f"[警告] 仍有 {remaining} 个 NaN（可能是模型未覆盖的类别/列）")
    else:
        print("[验证] 化学元素列无残留 NaN")

    return result


# ============================================================================
# 主流程
# ============================================================================

if __name__ == '__main__':

    print("=" * 70)
    print("  玄武岩地球化学缺失值插补 —— 训练 + 测试集一体化（MissForest）")
    print("  训练集 fit -> 内存保留（不存权重）-> 测试集 transform 插补")
    print("=" * 70)
    print(f"  训练输入目录: {TRAIN_INPUT_DIR}")
    print(f"  训练输出目录: {TRAIN_OUTPUT_DIR}\\MissForest")
    print(f"  测试输入文件: {TEST_INPUT_CSV}")
    print(f"  测试输出文件: {TEST_OUTPUT_CSV}")
    print()

    # 阶段 1：训练 + 训练集插补输出（内存保留模型）
    trained_models = train_and_impute_trainset(TRAIN_INPUT_DIR, TRAIN_OUTPUT_DIR)

    # 阶段 2：测试集插补输出
    impute_testset(TEST_INPUT_CSV, TEST_OUTPUT_CSV, trained_models)

    print("\n[完成] 训练集与测试集插补全部执行完毕！")
