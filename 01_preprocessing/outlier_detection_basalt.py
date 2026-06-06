import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import SPLIT_BY_TYPE_DIR

# 输入：按构造环境拆分的训练子集；输出：同级 clean 目录（IQR 去离群后）
input_dir = SPLIT_BY_TYPE_DIR
output_dir = input_dir.parent / 'clean'
# 确保输出目录存在
output_dir.mkdir(parents=True, exist_ok=True)

# 要分析的列
columns_to_extract = [
    "NA2O(WT%)", "MGO(WT%)", "AL2O3(WT%)", "SIO2(WT%)", "P2O5(WT%)", "K2O(WT%)",
    "CAO(WT%)", "TIO2(WT%)", "MNO(WT%)", "FEOT(WT%)", "RB(PPM)", "V(PPM)",
    "CR(PPM)", "CO(PPM)", "NI(PPM)", "BA(PPM)", "SR(PPM)", "Y(PPM)", "ZR(PPM)",
    "NB(PPM)", "LA(PPM)", "CE(PPM)", "PR(PPM)", "ND(PPM)", "SM(PPM)", "EU(PPM)",
    "GD(PPM)", "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "YB(PPM)", "LU(PPM)",
    "HF(PPM)", "TA(PPM)", "TH(PPM)"
]


# 异常值检测函数
def mad_based_outlier(points, threshold=3.5):
    median = np.median(points)
    diff = np.abs(points - median)
    mad = np.median(diff)
    modified_z_score = 0.6745 * diff / mad
    return modified_z_score > threshold


def iqr_based_outlier(data, threshold):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (iqr * threshold)
    upper_bound = q3 + (iqr * threshold)
    return (data < lower_bound) | (data > upper_bound)


def oi_ratio_outlier(data, k=1.5):
    mean = np.mean(data)
    std_dev = np.std(data)
    lower_bound = mean - k * std_dev
    upper_bound = mean + k * std_dev
    return (data < lower_bound) | (data > upper_bound)


# 处理单个文件的函数
def process_file(file_path):
    print(f"Processing file: {file_path}")
    data = pd.read_csv(file_path)
    original_count = len(data)

    iqr_outliers = pd.DataFrame(index=data.index)
    threshold = 6
    # file_names = ['BACK-ARC_BASIN.csv', 'SPREADING_CENTER.csv']
    #
    # if file_path.name in file_names:
    #     threshold = 12
    for column in columns_to_extract:
        if column in data.columns:
            column_data = data[column].dropna()
            iqr_outliers[column] = iqr_based_outlier(column_data, threshold)

    iqr_row_outliers = iqr_outliers.any(axis=1)
    iqr_row_outliers = iqr_row_outliers.reindex(data.index, fill_value=False)

    print(f"基于IQR的异常值行数: {iqr_row_outliers.sum()}")

    data_cleaned = data[~iqr_row_outliers]

    output_file_name = file_path.stem + "_clean.csv"
    output_path = output_dir / output_file_name
    data_cleaned.to_csv(output_path, index=False)

    print(f"清理后的数据已保存到: {output_path}\n")
    return {
        "file_name": file_path.name,
        "before_count": original_count,
        "after_count": len(data_cleaned),
        "removed_count": original_count - len(data_cleaned),
    }


# 处理目录中的所有CSV文件
cleaning_stats = []
for file in input_dir.glob('*.csv'):
    cleaning_stats.append(process_file(file))

# 汇总打印每类文件清洗前后的数量变化
print("每类清洗前后的数量变化:")
for stats in cleaning_stats:
    print(
        f"{stats['file_name']}: "
        f"清洗前 {stats['before_count']} 行, "
        f"清洗后 {stats['after_count']} 行, "
        f"减少 {stats['removed_count']} 行"
    )

print("所有文件处理完成。")
