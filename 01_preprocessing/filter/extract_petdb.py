"""PetDB 玄武岩正式筛选脚本（非交互）。

清洗规则与 petdb_filter_analyzer.py 保持一致：
1. 兼容 PetDB 2.0 列名，先合并被拆到多行的互补分析值（consolidate_petdb2_rows）；
2. 按统一优先级补全 FeOT（PetDB 允许元素 Fe 回退）；
3. 剔除年代文本（Geologic Age Prefix + Geologic Age）含 ARCHEAN 的太古代样品；
4. 36 个目标元素中至少 20 个非空；
5. 关键主量 SiO2/Al2O3/FeOT/MgO/CaO 必须非空；
6. 十个主量氧化物换算到无水 100% 后，按玄武质范围
   SiO2=45-53、MgO=4.5-12、Al2O3=12-19 wt% 筛选；
7. 标签名规范（如 SPREADING CENTER→SPREADING_CENTER），每类最多 20000 条；
8. 从 REFERENCES 提取发表年份（PUBLICATION_YEAR）。
"""

import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iron_normalization import add_standard_columns, calculate_feot
from major_element_normalization import normalize_major_oxides_for_filtering
from petdb_filter_analyzer import consolidate_petdb2_rows
from config.paths import PETDB_RAW_CSV, PETDB_FILTERED_CSV

OUTPUT_METADATA_COLUMNS = [
    'TECTONIC SETTING',
    'longitude',
    'latitude',
    'AGE',
    'PUBLICATION_YEAR',
]

# 中文注释：主要氧化物，要求非空（不能缺失）。
REQUIRED_MAJOR_OXIDES = ["SIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)", "MGO(WT%)", "CAO(WT%)"]

COLUMNS_TO_EXTRACT = [
    "NA2O(WT%)", "MGO(WT%)", "AL2O3(WT%)", "SIO2(WT%)", "P2O5(WT%)", "K2O(WT%)",
    "CAO(WT%)", "TIO2(WT%)", "MNO(WT%)", "FEOT(WT%)", "RB(PPM)", "V(PPM)",
    "CR(PPM)", "CO(PPM)", "NI(PPM)", "BA(PPM)", "SR(PPM)", "Y(PPM)", "ZR(PPM)",
    "NB(PPM)", "LA(PPM)", "CE(PPM)", "PR(PPM)", "ND(PPM)", "SM(PPM)", "EU(PPM)",
    "GD(PPM)", "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "YB(PPM)", "LU(PPM)",
    "HF(PPM)", "TA(PPM)", "TH(PPM)",
]


def extract_years_from_reference(ref_str):
    """从参考文献字符串中提取所有年份（形如 "AUTHOR, YYYY"）。"""
    if pd.isna(ref_str):
        return ""

    matches = re.findall(r'[A-Z-]+,\s*(\d{4})', str(ref_str))
    if matches:
        return "; ".join(matches)
    return ""


def preprocess_data(file_path, columns_to_extract):
    try:
        df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None

    # 中文注释：兼容 PetDB 2.0 列名，并在所有筛选前补全 FeOT。
    metadata_aliases = {
        "TECTONIC SETTING": ["Tectonic Setting"],
        "LONGITUDE": ["Longitude"],
        "LATITUDE": ["Latitude"],
        "AGE": ["Geologic Age"],
        "AGE PREFIX": ["Geologic Age Prefix"],
        "REFERENCES": ["Citation"],
    }
    df = add_standard_columns(
        df,
        ["TECTONIC SETTING", "LONGITUDE", "LATITUDE"],
        aliases=metadata_aliases,
    )
    for optional_column in ["AGE", "AGE PREFIX", "REFERENCES"]:
        try:
            df = add_standard_columns(
                df,
                [optional_column],
                aliases=metadata_aliases,
            )
        except KeyError:
            pass
    df, consolidation_stats = consolidate_petdb2_rows(df, columns_to_extract)
    print(f"PetDB 2.0 拆行重组统计：{consolidation_stats}")
    df, feot_source_counts = calculate_feot(df, allow_elemental_fe=True)
    df = add_standard_columns(df, columns_to_extract)
    df[columns_to_extract] = df[columns_to_extract].apply(
        pd.to_numeric,
        errors="coerce",
    )
    print(f"FeOT 来源统计：{feot_source_counts}")

    # 中文注释：与 GeoROC 保持一致，在化学筛选前剔除太古代样品。
    if "AGE" in df.columns:
        age_text = df["AGE"].fillna("").astype(str)
        if "AGE PREFIX" in df.columns:
            age_text = df["AGE PREFIX"].fillna("").astype(str) + " " + age_text
        df = df.loc[
            ~age_text.str.upper().str.contains("ARCHEAN", na=False)
        ].copy()

    # 处理 REFERENCES 列，提取年份
    if 'REFERENCES' in df.columns:
        df['YEAR'] = df['REFERENCES'].apply(extract_years_from_reference)
        print("已从 REFERENCES 列提取年份")

    # 替换标签名
    df['TECTONIC SETTING'] = df['TECTONIC SETTING'].replace({
        'OCEAN_ISLAND': 'OCEAN ISLAND',
        'OCEANIC_PLATEAU': 'OCEANIC PLATEAU',
        'BACK-ARC BASIN': 'BACK-ARC_BASIN',
        'CONTINENTAL RIFT': 'CONTINENTAL_RIFT',
        'SPREADING CENTER': 'SPREADING_CENTER',
    })
    tectonic_setting_counts = Counter(df['TECTONIC SETTING'])
    print("筛选前 TECTONIC SETTING 列的值及其出现次数:")
    for tectonic_setting, count in tectonic_setting_counts.items():
        print(f"{tectonic_setting}: {count}")

    # 中文注释：与 GeoROC 一致，36个目标化学元素中至少20个非空。
    count_non_nans = df[columns_to_extract].notna().sum(axis=1)
    mask = count_non_nans >= 20

    # 只保留和 georoc 对齐的化学特征列，避免把无关元数据写入输出文件
    filtered_df = df.loc[mask, columns_to_extract].copy()

    print(f"筛选后数据行数: {len(filtered_df)}")

    # 中文注释：主要氧化物 SiO2/Al2O3/FeOT/MgO/CaO 不能缺失。
    oxide_ok = filtered_df[REQUIRED_MAJOR_OXIDES].apply(
        lambda col: pd.to_numeric(col, errors='coerce')
    ).notna().all(axis=1)
    filtered_df = filtered_df[oxide_ok].copy()
    print(f"主要氧化物非空筛选后数据行数: {len(filtered_df)}")

    # 先将完整的 10 项主量氧化物换算到无水 100%，再筛选玄武质成分范围。
    normalized_major, normalization_ok = normalize_major_oxides_for_filtering(filtered_df)
    filtered_df = filtered_df.loc[normalization_ok].copy()
    normalized_major = normalized_major.loc[filtered_df.index]
    filtered_df = filtered_df[
        normalized_major["SIO2(WT%)"].between(45, 53, inclusive="both")
    ]
    print(f"SIO2筛选后数据行数: {len(filtered_df)}")

    # 筛选 MgO 和 Al2O3 含量
    filtered_df = filtered_df[
        normalized_major.loc[filtered_df.index, "MGO(WT%)"].between(
            4.5, 12, inclusive="both"
        )
        & normalized_major.loc[filtered_df.index, "AL2O3(WT%)"].between(
            12, 19, inclusive="both"
        )
    ].copy()

    filtered_df = filtered_df.round(3)
    filtered_df = filtered_df.apply(pd.to_numeric, errors='coerce')

    # 追加 georoc 目标表中需要保留的元数据列
    filtered_df['TECTONIC SETTING'] = df.loc[filtered_df.index, 'TECTONIC SETTING']
    filtered_df['longitude'] = df.loc[filtered_df.index, 'LONGITUDE']
    filtered_df['latitude'] = df.loc[filtered_df.index, 'LATITUDE']
    filtered_df['AGE'] = df.loc[filtered_df.index, 'AGE']
    if 'YEAR' in df.columns:
        filtered_df['PUBLICATION_YEAR'] = df.loc[filtered_df.index, 'YEAR']
    else:
        filtered_df['PUBLICATION_YEAR'] = np.nan

    filtered_df = filtered_df[columns_to_extract + OUTPUT_METADATA_COLUMNS]

    filtered_df.drop_duplicates(inplace=True)

    filtered_df = filter_tectonic_setting(filtered_df)
    return filtered_df


def filter_tectonic_setting(df):
    """只保留 PetDB 目标构造环境，并将每类裁剪到 20000 条以内。"""
    allowed_settings = [
        "OCEAN ISLAND",       # 洋岛
        "BACK-ARC_BASIN",     # 弧后盆地
        "CONTINENTAL_RIFT",   # 大陆裂谷
        "OCEANIC PLATEAU",    # 海洋高原
        "SPREADING_CENTER",   # 洋中脊（扩张中心）
    ]
    final_count = 20000
    filtered_df = df[df["TECTONIC SETTING"].isin(allowed_settings)]
    for setting in allowed_settings:
        setting_count = len(filtered_df[filtered_df['TECTONIC SETTING'] == setting])
        if setting_count > final_count:
            drop_count = setting_count - final_count
            to_drop = filtered_df[filtered_df['TECTONIC SETTING'] == setting].sample(drop_count).index
            filtered_df = filtered_df.drop(to_drop)
    tectonic_setting_counts = Counter(filtered_df['TECTONIC SETTING'])
    print("筛选后 TECTONIC SETTING 列的值及其出现次数:")
    for tectonic_setting, count in tectonic_setting_counts.items():
        print(f"{tectonic_setting}: {count}")
    return filtered_df


def ensure_parent_dir(file_path):
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


if __name__ == "__main__":
    file_path = str(PETDB_RAW_CSV)
    output_path = str(PETDB_FILTERED_CSV)
    df = preprocess_data(file_path, COLUMNS_TO_EXTRACT)
    if df is None:
        raise RuntimeError(f'Failed to preprocess data from {file_path}')
    df = df.reset_index(drop=True)
    ensure_parent_dir(output_path)
    try:
        df.to_csv(output_path, index=False)
        print('文件转换成功')
    except PermissionError:
        fallback_output_path = os.path.splitext(output_path)[0] + '_aligned_preview.csv'
        df.to_csv(fallback_output_path, index=False)
        print(f'目标文件被占用，结果已保存到: {fallback_output_path}')
