import pandas as pd


# 用于无水标准化的 10 个主量氧化物；FeOT 必须先按统一铁换算规则补全。
MAJOR_OXIDES = [
    "NA2O(WT%)",
    "MGO(WT%)",
    "AL2O3(WT%)",
    "SIO2(WT%)",
    "P2O5(WT%)",
    "K2O(WT%)",
    "CAO(WT%)",
    "TIO2(WT%)",
    "MNO(WT%)",
    "FEOT(WT%)",
]


def normalize_major_oxides_for_filtering(df):
    """
    将完整的 10 项主量氧化物逐行归一到无水 100% 基准。

    缺失主量氧化物的样品不参与归一化，避免把“未测”错误地当成 0。
    返回值中的布尔掩码用于筛除不能可靠归一化的样品。
    """
    missing_columns = [column for column in MAJOR_OXIDES if column not in df.columns]
    if missing_columns:
        raise KeyError(f"缺少无水标准化所需列: {missing_columns}")

    numeric_oxides = df[MAJOR_OXIDES].apply(pd.to_numeric, errors="coerce")
    complete_mask = numeric_oxides.notna().all(axis=1)
    row_total = numeric_oxides.sum(axis=1, min_count=len(MAJOR_OXIDES))
    valid_mask = complete_mask & row_total.gt(0)

    normalized = numeric_oxides.div(row_total, axis=0).mul(100)
    normalized.loc[~valid_mask, :] = float("nan")
    return normalized, valid_mask
