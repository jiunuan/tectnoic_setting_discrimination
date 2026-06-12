"""
训练集拟合分位数边界，并统一转换训练集和测试集。

流程：
1. 从 SMOTE 前的原始训练集拟合每个特征的254个分位数边界；
2. 保存 quantile_params.json；
3. 使用同一套边界转换 SMOTE 后训练集、未 SMOTE 训练集和测试集，
   有效值编码为 1-255。

太古代应用集不在本脚本处理：正式太古代缺失编码预测见
06_archean_application/archean_vit_transformer_dualstream_predict_analysis.py，
其内部直接加载同一份 quantile_params.json，缺失元素数值固定编码为 0。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    TRAIN_MAJOR_NORM_CSV, TEST_MAJOR_NORM_CSV,
    TRAIN_SMOTE_CSV,
    TRAIN_NORM_CSV, TRAIN_NORM_NO_SMOTE_CSV, TEST_NORM_CSV,
    QUANTILE_PARAMS_JSON,
)


# ============================================================================
# 路径配置（集中在 config/paths.py）
# ============================================================================

# 中文注释：分位数参数只从SMOTE前的真实训练样本拟合，避免合成样本改变预处理尺度。
QUANTILE_FIT_INPUT_FILE = str(TRAIN_MAJOR_NORM_CSV)
TRAIN_INPUT_FILE = str(TRAIN_SMOTE_CSV)
TRAIN_OUTPUT_FILE = str(TRAIN_NORM_CSV)
NO_SMOTE_TRAIN_OUTPUT_FILE = str(TRAIN_NORM_NO_SMOTE_CSV)

TEST_INPUT_FILE = str(TEST_MAJOR_NORM_CSV)
TEST_OUTPUT_FILE = str(TEST_NORM_CSV)

QUANTILE_PARAMS_FILE = str(QUANTILE_PARAMS_JSON)


# ============================================================================
# 特征列
# ============================================================================

COLUMNS_TO_EXTRACT = [
    'NA2O(WT%)', 'MGO(WT%)', 'AL2O3(WT%)', 'SIO2(WT%)', 'P2O5(WT%)',
    'K2O(WT%)', 'CAO(WT%)', 'TIO2(WT%)', 'MNO(WT%)', 'FEOT(WT%)',
    'RB(PPM)', 'V(PPM)', 'CR(PPM)', 'CO(PPM)', 'NI(PPM)', 'BA(PPM)',
    'SR(PPM)', 'Y(PPM)', 'ZR(PPM)', 'NB(PPM)', 'LA(PPM)', 'CE(PPM)',
    'PR(PPM)', 'ND(PPM)', 'SM(PPM)', 'EU(PPM)', 'GD(PPM)', 'TB(PPM)',
    'DY(PPM)', 'HO(PPM)', 'ER(PPM)', 'YB(PPM)', 'LU(PPM)', 'HF(PPM)',
    'TA(PPM)', 'TH(PPM)',
]


# ============================================================================
# 分位数拟合和转换
# ============================================================================

def validate_feature_columns(data, columns, dataset_name):
    """检查输入数据是否包含全部36个特征，并转换为数值。"""
    missing_columns = [column for column in columns if column not in data.columns]
    if missing_columns:
        raise ValueError(f'{dataset_name} 缺少特征列: {missing_columns}')

    numeric_features = data[columns].apply(pd.to_numeric, errors='coerce')
    if numeric_features.isna().any().any():
        nan_count = int(numeric_features.isna().sum().sum())
        raise ValueError(f'{dataset_name} 仍有 {nan_count} 个缺失特征值')
    if not np.isfinite(numeric_features.to_numpy(dtype=float)).all():
        raise ValueError(f'{dataset_name} 包含非有限特征值')
    return numeric_features


def fit_quantile_boundaries(data, columns):
    """从训练集拟合每列的254个分位数边界。"""
    features = validate_feature_columns(data, columns, '训练集')
    params = {}

    for column in columns:
        sorted_values = np.sort(features[column].to_numpy(dtype=float))
        sample_count = len(sorted_values)
        if sample_count == 0:
            raise ValueError(f'训练特征 {column} 没有有效样品')

        indices = [
            min(int(np.round(sample_count * rank / 255)), sample_count - 1)
            for rank in range(1, 255)
        ]
        params[column] = [float(sorted_values[index]) for index in indices]

    return params


def save_quantile_params(params):
    """保存训练集拟合得到的分位数边界。"""
    Path(QUANTILE_PARAMS_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(QUANTILE_PARAMS_FILE, 'w', encoding='utf-8') as file:
        json.dump(params, file, indent=2)
    print(f'[参数输出] {QUANTILE_PARAMS_FILE}')


def apply_quantile_transform(data, columns, params, dataset_name):
    """使用训练集边界将36个特征映射到1-255。"""
    features = validate_feature_columns(data, columns, dataset_name)
    missing_params = [column for column in columns if column not in params]
    if missing_params:
        raise ValueError(f'分位数参数缺少特征: {missing_params}')

    transformed = pd.DataFrame(index=data.index)
    for column in columns:
        boundaries = np.asarray(params[column], dtype=float)
        values = features[column].to_numpy(dtype=float)
        transformed[column] = (
            np.searchsorted(boundaries, values, side='left') + 1
        ).clip(1, 255).astype(np.int16)

    if 'TECTONIC SETTING' in data.columns:
        transformed['TECTONIC SETTING'] = data['TECTONIC SETTING'].to_numpy()
    return transformed.reset_index(drop=True)


def print_boundary_stats(transformed, dataset_name):
    """输出落在训练分布两端的特征值比例。"""
    feature_data = transformed[COLUMNS_TO_EXTRACT]
    boundary_count = int(
        ((feature_data == 1) | (feature_data == 255)).sum().sum()
    )
    total_count = feature_data.size
    print(
        f'[{dataset_name}] 边界编码数量: {boundary_count}/{total_count} '
        f'({boundary_count / total_count:.2%})'
    )


# ============================================================================
# 输出
# ============================================================================

def save_csv(data, output_path, dataset_name, encoding=None):
    """保存 CSV 并打印输出信息。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False, encoding=encoding)
    print(f'[{dataset_name}输出] {output_path}，shape={data.shape}')


def main():
    """一次完成参数拟合及训练/测试集的分位数转换。"""
    print('=' * 70)
    print('分位数分箱：SMOTE前训练集 fit -> SMOTE后训练/未SMOTE训练/测试集 transform')
    print('=' * 70)

    quantile_fit_data = pd.read_csv(
        QUANTILE_FIT_INPUT_FILE,
        low_memory=False,
    )
    train_data = pd.read_csv(TRAIN_INPUT_FILE, low_memory=False)
    test_data = pd.read_csv(TEST_INPUT_FILE, low_memory=False)

    print(f'[分位数拟合输入] {quantile_fit_data.shape}')
    print(f'[SMOTE后训练输入] {train_data.shape}')
    print(f'[测试输入] {test_data.shape}')

    params = fit_quantile_boundaries(
        quantile_fit_data,
        COLUMNS_TO_EXTRACT,
    )
    save_quantile_params(params)

    normalized_train = apply_quantile_transform(
        train_data,
        COLUMNS_TO_EXTRACT,
        params,
        '训练集',
    )
    # 中文注释：额外输出未经过SMOTE的真实训练集，供类别加权交叉熵实验使用。
    normalized_train_no_smote = apply_quantile_transform(
        quantile_fit_data,
        COLUMNS_TO_EXTRACT,
        params,
        '未SMOTE训练集',
    )
    normalized_test = apply_quantile_transform(
        test_data,
        COLUMNS_TO_EXTRACT,
        params,
        '测试集',
    )

    print_boundary_stats(normalized_train, '训练集')
    print_boundary_stats(normalized_train_no_smote, '未SMOTE训练集')
    print_boundary_stats(normalized_test, '测试集')

    save_csv(normalized_train, TRAIN_OUTPUT_FILE, '训练集')
    save_csv(
        normalized_train_no_smote,
        NO_SMOTE_TRAIN_OUTPUT_FILE,
        '未SMOTE训练集',
    )
    save_csv(normalized_test, TEST_OUTPUT_FILE, '测试集')

    print('[完成] 训练/测试集已使用同一训练集分位数参数完成转换')


if __name__ == '__main__':
    main()
