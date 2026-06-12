"""选择性 SMOTE：仅对训练集少数类过采样（测试集绝不参与）。

对四个原始样本明显偏少的少数类（Island arc / Intra-oceanic arc /
BACK-ARC_BASIN / OCEANIC PLATEAU），外加召回率偏低的 CONTINENTAL_RIFT，
共五类用普通 SMOTE 补到 3,000 条；其余类别保留真实分布、不做合成。

输入为主量无水标准化后的训练集（插补后、分位数编码前）；
输出供 normalize.py 做分位数编码。分位数边界仍从 SMOTE 前的
真实训练集拟合，避免合成样本改变预处理尺度。

SMOTE 新增样品在已插补的完整特征空间里合成，因此它们的缺失 mask
在模型训练阶段统一记为 0（见 04_model/ablation_v4_vit_transformer.py）。
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import TRAIN_MAJOR_NORM_CSV, TRAIN_SMOTE_CSV

INPUT_FILE = str(TRAIN_MAJOR_NORM_CSV)
OUTPUT_FILE = str(TRAIN_SMOTE_CSV)
LABEL_COLUMN = "TECTONIC SETTING"

# 中文注释：只对原始样本数明显偏少的4类及召回率偏低的CR补到3000，
# 其余类别保留真实分布，避免合成样本过多。
CLASS_TARGETS = {
    "CONTINENTAL_RIFT": 3000,
    "Island arc": 3000,
    "BACK-ARC_BASIN": 3000,
    "Intra-oceanic arc": 3000,
    "OCEANIC PLATEAU": 3000,
}

FEATURE_COLUMNS = [
    "NA2O(WT%)", "MGO(WT%)", "AL2O3(WT%)", "SIO2(WT%)",
    "P2O5(WT%)", "K2O(WT%)", "CAO(WT%)", "TIO2(WT%)",
    "MNO(WT%)", "FEOT(WT%)", "RB(PPM)", "V(PPM)",
    "CR(PPM)", "CO(PPM)", "NI(PPM)", "BA(PPM)", "SR(PPM)",
    "Y(PPM)", "ZR(PPM)", "NB(PPM)", "LA(PPM)", "CE(PPM)",
    "PR(PPM)", "ND(PPM)", "SM(PPM)", "EU(PPM)", "GD(PPM)",
    "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "YB(PPM)",
    "LU(PPM)", "HF(PPM)", "TA(PPM)", "TH(PPM)",
]

RANDOM_STATE = 42
K_NEIGHBORS = 5


def validate_input(df):
    """检查标签列和建模特征，避免静默丢弃训练样本。"""
    required_columns = FEATURE_COLUMNS + [LABEL_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise KeyError(f"缺少 SMOTE 所需列: {missing_columns}")

    numeric_features = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    invalid_mask = ~np.isfinite(numeric_features.to_numpy()).all(axis=1)
    if invalid_mask.any():
        raise ValueError(
            f"SMOTE 输入仍有 {int(invalid_mask.sum())} 行缺失值或无穷值，"
            "请先完成训练集插补。"
        )
    return numeric_features


def build_sampling_strategy(labels):
    """仅为指定少数类建立目标数量，不改变其他类别。"""
    counts = Counter(labels)
    missing_classes = [label for label in CLASS_TARGETS if label not in counts]
    if missing_classes:
        raise ValueError(f"训练集中找不到目标类别: {missing_classes}")

    strategy = {
        label: target
        for label, target in CLASS_TARGETS.items()
        if counts[label] < target
    }
    return counts, strategy


def preserve_metadata(df, resampled_features, resampled_labels):
    """原始行保留元数据，合成行无法继承的元数据保持为空。"""
    result = pd.DataFrame(resampled_features, columns=FEATURE_COLUMNS)
    result[LABEL_COLUMN] = resampled_labels

    metadata_columns = [
        column
        for column in df.columns
        if column not in FEATURE_COLUMNS + [LABEL_COLUMN]
    ]
    original_rows = len(df)
    for column in metadata_columns:
        result[column] = pd.NA
        result.loc[: original_rows - 1, column] = df[column].to_numpy()

    return result[df.columns]


def main():
    """只对训练集执行普通SMOTE，不处理测试集。"""
    data = pd.read_csv(INPUT_FILE, low_memory=False)
    features = validate_input(data)
    labels = data[LABEL_COLUMN].astype(str)
    counts, sampling_strategy = build_sampling_strategy(labels)

    print("SMOTE 前类别分布:")
    print(pd.Series(counts).sort_values(ascending=False).to_string())
    print(f"\n采样目标: {sampling_strategy}")

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    if not sampling_strategy:
        data.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"所有目标类别均已达到设定数量，原数据已写入: {OUTPUT_FILE}")
        return

    # 近邻计算前做标准化，避免 wt.% 与 ppm 的量纲差异主导距离。
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    smallest_target_class = min(counts[label] for label in sampling_strategy)
    k_neighbors = min(K_NEIGHBORS, smallest_target_class - 1)

    # 中文注释：普通SMOTE在同类近邻之间进行线性插值，
    # 不额外集中生成分类边界附近的困难样本。
    sampler = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=RANDOM_STATE,
        k_neighbors=k_neighbors,
    )
    resampled_scaled, resampled_labels = sampler.fit_resample(
        scaled_features,
        labels,
    )
    resampled_features = scaler.inverse_transform(resampled_scaled)
    # 中文注释：逆标准化可能产生约1e-15的浮点负误差，化学含量统一截断为0。
    resampled_features = np.clip(resampled_features, 0.0, None)
    result = preserve_metadata(data, resampled_features, resampled_labels)
    result.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\nSMOTE 后类别分布:")
    print(result[LABEL_COLUMN].value_counts().to_string())
    print(f"\n结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
