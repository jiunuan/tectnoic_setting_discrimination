"""GEOROC 玄武岩正式筛选脚本（非交互）。

清洗规则与 georoc_filter_analyzer.py 保持一致：
1. 所有筛选前先按统一优先级补全 FeOT（iron_normalization.calculate_feot）；
2. 剔除 AGE 含 ARCHEAN（含 Neo/Meso/Paleo/Eoarchean）的太古代样品；
3. 关键主量 SiO2/Al2O3/FeOT/MgO/CaO 必须非空；
4. 36 个目标元素中至少 20 个非空；
5. 十个主量氧化物换算到无水 100% 后，按玄武质范围
   SiO2=45-53、MgO=4.5-12、Al2O3=12-19 wt% 筛选；LOI<5 wt% 或缺失；
6. 保留目标构造环境并统一标签名，每类最多 20000 条；
7. 按参考文献编号映射发表年份（PUBLICATION_YEAR）。
"""

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iron_normalization import calculate_feot
from major_element_normalization import normalize_major_oxides_for_filtering
from config.paths import (
    REFINED_EXPANDED_CSV,
    GEOROC_FILTERED_CSV,
    GEOROC_REFERENCES_CSV,
)

DEFAULT_FILE_PATH = str(REFINED_EXPANDED_CSV)
DEFAULT_OUTPUT_PATH = str(GEOROC_FILTERED_CSV)
REFERENCES_PATH = str(GEOROC_REFERENCES_CSV)

COLUMNS_TO_EXTRACT = [
    "NA2O(WT%)", "MGO(WT%)", "AL2O3(WT%)", "SIO2(WT%)", "P2O5(WT%)", "K2O(WT%)",
    "CAO(WT%)", "TIO2(WT%)", "MNO(WT%)", "FEOT(WT%)", "RB(PPM)", "V(PPM)",
    "CR(PPM)", "CO(PPM)", "NI(PPM)", "BA(PPM)", "SR(PPM)", "Y(PPM)", "ZR(PPM)",
    "NB(PPM)", "LA(PPM)", "CE(PPM)", "PR(PPM)", "ND(PPM)", "SM(PPM)", "EU(PPM)",
    "GD(PPM)", "TB(PPM)", "DY(PPM)", "HO(PPM)", "ER(PPM)", "YB(PPM)", "LU(PPM)",
    "HF(PPM)", "TA(PPM)", "TH(PPM)",
]

METADATA_COLUMNS = [
    "CITATIONS",
    "LOI(WT%)",
    "MIN. AGE (YRS.)",
    "MAX. AGE (YRS.)",
    "AGE",
    "ALTERATION",
    "TECTONIC SETTING",
]

# 中文注释：主要氧化物，要求非空（不能缺失）。
REQUIRED_MAJOR_OXIDES = ["SIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)", "MGO(WT%)", "CAO(WT%)"]

ALLOWED_SETTINGS = [
    "OCEAN ISLAND",
    "Continental arc",
    "Island arc",
    "Intra-oceanic arc",
    "Back-arc basin",
    "CONTINENTAL FLOOD BASALT",
    "RIFT VOLCANICS",
    "OCEANIC PLATEAU",
]


def build_parser():
    parser = argparse.ArgumentParser(description="筛选 GEOROC 玄武岩数据并导出建模特征表。")
    parser.add_argument("--file-path", default=DEFAULT_FILE_PATH, help="输入 GEOROC CSV 完整路径。")
    parser.add_argument("--output-path", default=DEFAULT_OUTPUT_PATH, help="输出 CSV 完整路径。")
    return parser


def read_csv_fallback(file_path):
    """读取 CSV，UTF-8 失败时回退到 ISO-8859-1。"""
    try:
        return pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)


def clean_age_string(age_str):
    """清理年龄字符串中的中括号编号。"""
    if pd.isna(age_str):
        return ""
    return re.sub(r"\s*\[\d+\]", "", str(age_str)).strip()


def conuter_tectonicSetting_counts(df):
    """打印构造环境数量，保留历史函数名以兼容旧调用。"""
    print("筛选后的 TECTONIC SETTING 数量:")
    for setting, count in Counter(df["TECTONIC SETTING"]).items():
        print(f"{setting}: {count}")


def filter_tectonic_setting(df):
    """保留目标构造环境并统一标签名称。"""
    filtered_df = df.loc[df["TECTONIC SETTING"].isin(ALLOWED_SETTINGS)].copy()

    # 中文注释：每类最多保留 20000 条，避免极端类别规模。
    for setting in ALLOWED_SETTINGS:
        setting_rows = filtered_df.loc[filtered_df["TECTONIC SETTING"] == setting]
        if len(setting_rows) > 20000:
            filtered_df = filtered_df.drop(setting_rows.sample(len(setting_rows) - 20000).index)

    filtered_df.loc[:, "TECTONIC SETTING"] = filtered_df["TECTONIC SETTING"].replace(
        {"RIFT VOLCANICS": "CONTINENTAL_RIFT", "Back-arc basin": "BACK-ARC_BASIN"}
    )
    filtered_df = filtered_df.drop(
        columns=["LOI(WT%)", "MIN. AGE (YRS.)", "MAX. AGE (YRS.)", "ALTERATION"],
        errors="ignore",
    )
    conuter_tectonicSetting_counts(filtered_df)
    return filtered_df


def process_citations(df):
    """将 CITATIONS 中的参考文献编号映射为发表年份。"""
    refs_df = pd.read_csv(REFERENCES_PATH, low_memory=False)
    ref_year_map = dict(zip(refs_df["reference_number"], refs_df["year"]))

    def extract_years(citations):
        if pd.isna(citations):
            return "N/A"
        citation_numbers = re.findall(r"\[(\d+)\]", str(citations))
        years = sorted(
            {str(ref_year_map.get(int(number))) for number in citation_numbers if ref_year_map.get(int(number)) != "N/A"}
        )
        return "; ".join(years) if years else "N/A"

    result = df.copy()
    result["PUBLICATION_YEAR"] = result["CITATIONS"].apply(extract_years)
    result = result.drop(columns=["CITATIONS"])
    columns = [column for column in result.columns if column not in {"AGE", "PUBLICATION_YEAR"}]
    return result[columns + ["AGE", "PUBLICATION_YEAR"]]


def preprocess_data(file_path, columns_to_extract):
    """执行 GEOROC 主线地球化学筛选。"""
    raw_df = read_csv_fallback(file_path)
    # 中文注释：在完整度和主量元素筛选前，按统一优先级补全 FeOT。
    raw_df, feot_source_counts = calculate_feot(
        raw_df,
        allow_elemental_fe=False,
    )
    print(f"FeOT 来源统计：{feot_source_counts}")
    print("原始数据 TECTONIC SETTING 数量:")
    for setting, count in Counter(raw_df["TECTONIC SETTING"]).items():
        print(f"{setting}: {count}")

    chemical_df = raw_df[columns_to_extract].apply(pd.to_numeric, errors="coerce")
    extracted_df = pd.concat([chemical_df, raw_df[METADATA_COLUMNS]], axis=1)
    # 先将完整的 10 项主量氧化物换算到无水 100%，再筛选玄武质成分范围。
    normalized_major, normalization_ok = normalize_major_oxides_for_filtering(extracted_df)
    extracted_df = extracted_df.loc[normalization_ok].copy()
    normalized_major = normalized_major.loc[extracted_df.index]

    # 中文注释：按历史主线条件筛选玄武岩主量元素与 LOI。
    extracted_df = extracted_df[
        normalized_major["SIO2(WT%)"].between(45, 53, inclusive="both")
        & normalized_major["MGO(WT%)"].between(4.5, 12, inclusive="both")
        & normalized_major["AL2O3(WT%)"].between(12, 19, inclusive="both")
        & ((extracted_df["LOI(WT%)"] < 5) | extracted_df["LOI(WT%)"].isna())
    ].copy()

    extracted_df["MIN. AGE (YRS.)"] = pd.to_numeric(
        extracted_df["MIN. AGE (YRS.)"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    extracted_df["MAX. AGE (YRS.)"] = pd.to_numeric(
        extracted_df["MAX. AGE (YRS.)"].astype(str).str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    extracted_df["AGE"] = extracted_df["AGE"].fillna("").apply(clean_age_string)

    # 中文注释：剔除 AGE 含 ARCHEAN（含 Neo/Meso/Paleo/Eoarchean）的太古代样品，避免污染现代玄武岩判别数据。
    extracted_df = extracted_df.loc[
        ~extracted_df["AGE"].str.upper().str.contains("ARCHEAN", na=False)
    ].copy()

    # 中文注释：主要氧化物 SiO2/Al2O3/FeOT/MgO/CaO 不能缺失。
    extracted_df = extracted_df.loc[
        extracted_df[REQUIRED_MAJOR_OXIDES].notna().all(axis=1)
    ].copy()

    # 中文注释：仅统计36个目标化学元素，与 PetDB 使用一致的完整度标准。
    extracted_df = extracted_df.loc[
        extracted_df[columns_to_extract].notna().sum(axis=1) >= 20
    ].copy()
    extracted_df["longitude"] = (
        raw_df.loc[extracted_df.index, "LONGITUDE MIN"]
        + raw_df.loc[extracted_df.index, "LONGITUDE MAX"]
    ) / 2
    extracted_df["latitude"] = (
        raw_df.loc[extracted_df.index, "LATITUDE MIN"]
        + raw_df.loc[extracted_df.index, "LATITUDE MAX"]
    ) / 2

    extracted_df = extracted_df.round(3).drop_duplicates().copy()
    extracted_df = filter_tectonic_setting(extracted_df)
    return process_citations(extracted_df)


def main():
    args = build_parser().parse_args()
    result = preprocess_data(args.file_path, COLUMNS_TO_EXTRACT).reset_index(drop=True)
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result.to_csv(args.output_path, index=False, encoding="utf-8-sig")
    print(f"已保存 GEOROC 筛选结果: {args.output_path}")


if __name__ == "__main__":
    main()
