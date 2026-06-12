"""
太古代玄武岩应用口径筛选（与 PCA 一致性脚本同一处理口径）。

本脚本只负责：
1. 提取36个地球化学特征；
2. 按 NaN、0 或负值统计缺失数，保留缺失数小于18的样品；
3. 要求 SiO2、Al2O3、FeOT、MgO、CaO 为有效正值；
4. 对10个主量元素进行逐行无水标准化（仅用正值主量参与求和）；
5. 保留标准化后 SiO2 为44-53 wt%、MgO不超过18 wt%的样品。

`preprocess_archean()` 同时被正式预测脚本
archean_vit_transformer_dualstream_predict_analysis.py 复用，
用于 6 克拉通案例数据的统一预处理。

注意：扩展太古代应用集（3,483 条）的构建入口是
extended_archean_pool_analysis.py（SiO2 上限放宽到 54 wt%），
本脚本 main() 仅复现 Liu 数据严格口径（SiO2≤53）的筛选结果。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import ARCHEAN_S3_CSV, ARCHEAN_DATA_SUBDIR


# ============================================================================
# 配置（路径集中在 config/paths.py）
# ============================================================================

SOURCE_CSV = str(ARCHEAN_S3_CSV)

OUTPUT_CSV = str(ARCHEAN_DATA_SUBDIR / 'archean_basalt_filtered_2116.csv')

MAX_MISSING_FEATURES_EXCLUSIVE = 18
SIO2_MIN = 44.0
SIO2_MAX = 53.0
MGO_MAX = 18.0
EXPECTED_SAMPLE_COUNT = 2116

SHORT_CHEMICAL_COLUMNS = [
    'NA2O', 'MGO', 'AL2O3', 'SIO2', 'P2O5', 'K2O', 'CAO', 'TIO2',
    'MNO', 'FEOT', 'RB', 'V', 'CR', 'CO', 'NI', 'BA', 'SR', 'Y',
    'ZR', 'NB', 'LA', 'CE', 'PR', 'ND', 'SM', 'EU', 'GD', 'TB',
    'DY', 'HO', 'ER', 'YB', 'LU', 'HF', 'TA', 'TH',
]

MAJOR_COLUMNS = [
    'SIO2', 'TIO2', 'AL2O3', 'FEOT', 'MNO',
    'MGO', 'CAO', 'NA2O', 'K2O', 'P2O5',
]

REQUIRED_MAJOR_COLUMNS = ['SIO2', 'AL2O3', 'FEOT', 'MGO', 'CAO']

CHEMICAL_COLUMN_MAPPING = {
    'SIO2': 'SIO2(WT%)', 'TIO2': 'TIO2(WT%)',
    'AL2O3': 'AL2O3(WT%)', 'FEOT': 'FEOT(WT%)',
    'MNO': 'MNO(WT%)', 'MGO': 'MGO(WT%)', 'CAO': 'CAO(WT%)',
    'NA2O': 'NA2O(WT%)', 'K2O': 'K2O(WT%)', 'P2O5': 'P2O5(WT%)',
    'V': 'V(PPM)', 'CR': 'CR(PPM)', 'CO': 'CO(PPM)', 'NI': 'NI(PPM)',
    'RB': 'RB(PPM)', 'SR': 'SR(PPM)', 'Y': 'Y(PPM)', 'ZR': 'ZR(PPM)',
    'NB': 'NB(PPM)', 'BA': 'BA(PPM)', 'LA': 'LA(PPM)', 'CE': 'CE(PPM)',
    'PR': 'PR(PPM)', 'ND': 'ND(PPM)', 'SM': 'SM(PPM)', 'EU': 'EU(PPM)',
    'GD': 'GD(PPM)', 'TB': 'TB(PPM)', 'DY': 'DY(PPM)', 'HO': 'HO(PPM)',
    'ER': 'ER(PPM)', 'YB': 'YB(PPM)', 'LU': 'LU(PPM)', 'HF': 'HF(PPM)',
    'TA': 'TA(PPM)', 'TH': 'TH(PPM)',
}


def extract_chemical_features(data: pd.DataFrame) -> pd.DataFrame:
    """兼容短列名和标准长列名，提取36个化学特征并转为数值。"""
    features = pd.DataFrame(index=data.index)
    missing_columns = []
    for short_name in SHORT_CHEMICAL_COLUMNS:
        long_name = CHEMICAL_COLUMN_MAPPING[short_name]
        if short_name in data.columns:
            source_column = short_name
        elif long_name in data.columns:
            source_column = long_name
        else:
            missing_columns.append(f'{short_name}/{long_name}')
            continue
        features[short_name] = pd.to_numeric(
            data[source_column],
            errors='coerce',
        )

    if missing_columns:
        raise ValueError(f'太古代原始数据缺少化学特征列: {missing_columns}')

    return features[SHORT_CHEMICAL_COLUMNS]


def normalize_major_elements(features: pd.DataFrame) -> pd.DataFrame:
    """仅使用正值主量元素进行逐行无水标准化到100 wt%。"""
    normalized = features.copy()
    positive_majors = normalized[MAJOR_COLUMNS].where(
        normalized[MAJOR_COLUMNS] > 0
    )
    major_totals = positive_majors.sum(axis=1, min_count=1)
    invalid_mask = major_totals.isna() | (major_totals <= 0)
    if invalid_mask.any():
        raise ValueError(
            f'存在 {int(invalid_mask.sum())} 条样品没有有效主量元素总和'
        )

    normalized.loc[:, MAJOR_COLUMNS] = positive_majors.div(
        major_totals,
        axis=0,
    ) * 100.0
    return normalized


def preprocess_archean(
    data: pd.DataFrame,
    expected_sample_count: int | None = None,
) -> pd.DataFrame:
    """执行与 PCA 脚本完全一致的太古代样品筛选。"""
    features = extract_chemical_features(data)
    original_count = len(data)

    # NaN、0和负值均视为缺失。
    missing_counts = (features.isna() | (features <= 0)).sum(axis=1)
    keep_missing = missing_counts < MAX_MISSING_FEATURES_EXCLUSIVE
    features_after_missing = features.loc[keep_missing].copy()
    count_after_missing = len(features_after_missing)

    # 五个关键主量必须为有效正值。
    required_values = features_after_missing[REQUIRED_MAJOR_COLUMNS]
    keep_required = (
        required_values.notna().all(axis=1)
        & (required_values > 0).all(axis=1)
    )
    features_after_required = features_after_missing.loc[keep_required].copy()
    count_after_required = len(features_after_required)

    # SiO2和MgO筛选使用无水标准化后的数值。
    normalized_features = normalize_major_elements(features_after_required)
    keep_basalt = (
        normalized_features['SIO2'].between(SIO2_MIN, SIO2_MAX)
        & normalized_features['MGO'].le(MGO_MAX)
    )
    final_features = normalized_features.loc[keep_basalt].copy()
    final_indices = final_features.index

    result = data.loc[final_indices].copy()
    # 中文注释：统一补充短列名，保证案例表和全库表可进入同一预测流程。
    for column in SHORT_CHEMICAL_COLUMNS:
        if column in MAJOR_COLUMNS:
            result[column] = final_features[column]
        elif column not in result.columns:
            result[column] = features.loc[final_indices, column]
    result.loc[:, MAJOR_COLUMNS] = final_features[MAJOR_COLUMNS]
    result['missing_feature_count_36'] = missing_counts.loc[
        final_indices
    ].to_numpy()
    result.reset_index(drop=True, inplace=True)

    print(f'[原始样品] {original_count}')
    print(
        f'[缺失筛选] missing < {MAX_MISSING_FEATURES_EXCLUSIVE}: '
        f'{count_after_missing}'
    )
    print(f'[关键主量有效] {count_after_required}')
    print(
        f'[玄武岩筛选] SiO2={SIO2_MIN:g}-{SIO2_MAX:g}, '
        f'MgO<={MGO_MAX:g}: {len(result)}'
    )

    if expected_sample_count is not None and len(result) != expected_sample_count:
        raise ValueError(
            f'筛选结果应为 {expected_sample_count} 条，实际为 {len(result)} 条'
        )
    return result


def main() -> None:
    """读取原始太古代数据并输出筛选后的2116条样品。"""
    data = pd.read_csv(SOURCE_CSV, low_memory=False)
    result = preprocess_archean(
        data,
        expected_sample_count=EXPECTED_SAMPLE_COUNT,
    )
    result.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    major_sum = result[MAJOR_COLUMNS].sum(axis=1, min_count=1)
    print(f'[输出] {OUTPUT_CSV}')
    print(f'[输出形状] {result.shape[0]} 条 × {result.shape[1]} 列')
    print(
        f'[主量总和范围] {major_sum.min():.6f}-'
        f'{major_sum.max():.6f}'
    )


if __name__ == '__main__':
    main()
