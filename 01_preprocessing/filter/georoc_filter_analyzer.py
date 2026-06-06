import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.paths import REFINED_EXPANDED_CSV, CM_RECLASS_DIR

# 默认使用回贴 expanded(高+中置信度)汇聚边缘标签后的总表，以保证岛弧样本量。
DEFAULT_FILE_PATH = str(REFINED_EXPANDED_CSV)
DEFAULT_REFERENCES_PATH = str(CM_RECLASS_DIR / "references_structured.csv")
DEFAULT_COLUMNS_TO_EXTRACT = [
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
    "RB(PPM)",
    "V(PPM)",
    "CR(PPM)",
    "CO(PPM)",
    "NI(PPM)",
    "BA(PPM)",
    "SR(PPM)",
    "Y(PPM)",
    "ZR(PPM)",
    "NB(PPM)",
    "LA(PPM)",
    "CE(PPM)",
    "PR(PPM)",
    "ND(PPM)",
    "SM(PPM)",
    "EU(PPM)",
    "GD(PPM)",
    "TB(PPM)",
    "DY(PPM)",
    "HO(PPM)",
    "ER(PPM)",
    "YB(PPM)",
    "LU(PPM)",
    "HF(PPM)",
    "TA(PPM)",
    "TH(PPM)",
]
DEFAULT_ALLOWED_SETTINGS = [
    "OCEAN ISLAND",
    "Continental arc",
    "Island arc",
    "Intra-oceanic arc",
    "Back-arc basin",
    "CONTINENTAL FLOOD BASALT",
    "RIFT VOLCANICS",
    "OCEANIC PLATEAU",
]
ADDITIONAL_COLUMNS = [
    "CITATIONS",
    "LOI(WT%)",
    "MIN. AGE (YRS.)",
    "MAX. AGE (YRS.)",
    "AGE",
    "ALTERATION",
    "TECTONIC SETTING",
]
LOCATION_COLUMNS = [
    "LONGITUDE MIN",
    "LONGITUDE MAX",
    "LATITUDE MIN",
    "LATITUDE MAX",
]
TECTONIC_REPLACEMENTS = {
    "RIFT VOLCANICS": "CONTINENTAL_RIFT",
    "Back-arc basin": "BACK-ARC_BASIN",
}


@dataclass
class FilterParams:
    columns_to_extract: list[str] = field(default_factory=lambda: DEFAULT_COLUMNS_TO_EXTRACT.copy())
    allowed_settings: list[str] = field(default_factory=lambda: DEFAULT_ALLOWED_SETTINGS.copy())
    sio2_min: float = 44.0
    sio2_max: float = 53.0
    use_sio2_filter: bool = True
    mgo_min: float = 4.5
    mgo_max: float = 12.0
    al2o3_min: float = 12.0
    al2o3_max: float = 19.0
    use_mgo_al2o3_filter: bool = True
    loi_max: float = 5.0
    keep_loi_nan: bool = True
    use_loi_filter: bool = True
    min_non_nan: int = 16
    use_min_non_nan_filter: bool = True
    exclude_archean_age: bool = True
    drop_duplicates: bool = True
    final_count_per_setting: int = 20000
    apply_setting_filter: bool = True
    rename_tectonic_settings: bool = True
    add_publication_year: bool = True
    keep_all_columns: bool = False


def clean_age_string(age_str):
    if pd.isna(age_str):
        return ""
    cleaned = re.sub(r"\s*\[\d+\]", "", str(age_str))
    return cleaned.strip()


def read_csv_with_fallback(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)


def summarize_tectonic_counts(df):
    if df is None or "TECTONIC SETTING" not in df.columns:
        return {}
    counts = Counter(df["TECTONIC SETTING"].fillna("UNKNOWN"))
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def append_step(stats, step_name, df):
    stats.append(
        {
            "step": step_name,
            "total_rows": len(df),
            "counts": summarize_tectonic_counts(df),
        }
    )


def prepare_extracted_dataframe(raw_df, columns_to_extract):
    required_columns = columns_to_extract + ADDITIONAL_COLUMNS + LOCATION_COLUMNS
    missing_columns = [column for column in required_columns if column not in raw_df.columns]
    if missing_columns:
        raise KeyError(f"Missing required columns: {missing_columns}")

    numeric_df = raw_df[columns_to_extract].apply(pd.to_numeric, errors="coerce")
    metadata_df = raw_df[ADDITIONAL_COLUMNS].copy()
    return pd.concat([numeric_df, metadata_df], axis=1)


def parse_age_columns(df):
    parsed_df = df.copy()
    parsed_df["MIN. AGE (YRS.)"] = (
        parsed_df["MIN. AGE (YRS.)"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
    )
    parsed_df["MAX. AGE (YRS.)"] = (
        parsed_df["MAX. AGE (YRS.)"].astype(str).str.extract(r"(\d+)", expand=False).astype(float)
    )
    parsed_df["AGE"] = parsed_df["AGE"].fillna("").apply(clean_age_string)
    return parsed_df


def finalize_columns(df, raw_df):
    finalized_df = df.copy()
    finalized_df["longitude"] = (
        raw_df.loc[finalized_df.index, "LONGITUDE MIN"]
        + (
            raw_df.loc[finalized_df.index, "LONGITUDE MAX"]
            - raw_df.loc[finalized_df.index, "LONGITUDE MIN"]
        )
        / 2
    )
    finalized_df["latitude"] = (
        raw_df.loc[finalized_df.index, "LATITUDE MIN"]
        + (
            raw_df.loc[finalized_df.index, "LATITUDE MAX"]
            - raw_df.loc[finalized_df.index, "LATITUDE MIN"]
        )
        / 2
    )
    return finalized_df.round(3)


def restore_original_columns(filtered_df, raw_df):
    """恢复筛选后样品的原始列，并同步更新最终标签与派生坐标列。"""
    restored_df = raw_df.loc[filtered_df.index].copy()
    restored_df["TECTONIC SETTING"] = filtered_df["TECTONIC SETTING"].values

    if "AGE" in restored_df.columns and "AGE" in filtered_df.columns:
        restored_df["AGE"] = filtered_df["AGE"].values

    restored_df["longitude"] = filtered_df["longitude"].values
    restored_df["latitude"] = filtered_df["latitude"].values
    return restored_df


def filter_tectonic_setting(df, params):
    filtered_df = df.loc[df["TECTONIC SETTING"].isin(params.allowed_settings)].copy()
    if params.final_count_per_setting > 0:
        for setting in params.allowed_settings:
            setting_rows = filtered_df.loc[filtered_df["TECTONIC SETTING"] == setting]
            if len(setting_rows) > params.final_count_per_setting:
                drop_count = len(setting_rows) - params.final_count_per_setting
                to_drop = setting_rows.sample(drop_count, random_state=42).index
                filtered_df = filtered_df.drop(to_drop)

    if params.rename_tectonic_settings:
        filtered_df.loc[:, "TECTONIC SETTING"] = filtered_df["TECTONIC SETTING"].replace(
            TECTONIC_REPLACEMENTS
        )

    columns_to_drop = ["LOI(WT%)", "MIN. AGE (YRS.)", "MAX. AGE (YRS.)", "ALTERATION"]
    drop_candidates = [column for column in columns_to_drop if column in filtered_df.columns]
    return filtered_df.drop(columns=drop_candidates, axis=1)


def process_citations(df, references_path=DEFAULT_REFERENCES_PATH):
    refs_df = read_csv_with_fallback(references_path)
    ref_year_map = dict(zip(refs_df["reference_number"], refs_df["year"]))

    def extract_years(citations):
        if pd.isna(citations):
            return "N/A"
        citation_numbers = re.findall(r"\[(\d+)\]", str(citations))
        years = [ref_year_map.get(int(num), "N/A") for num in citation_numbers]
        years = sorted(set(str(year) for year in years if year != "N/A"))
        return "; ".join(years) if years else "N/A"

    processed_df = df.copy()
    processed_df["PUBLICATION_YEAR"] = processed_df["CITATIONS"].apply(extract_years)
    processed_df = processed_df.drop(columns=["CITATIONS"], axis=1)

    cols = processed_df.columns.tolist()
    if "AGE" in cols:
        cols.remove("AGE")
        cols.append("AGE")
    if "PUBLICATION_YEAR" in cols:
        cols.remove("PUBLICATION_YEAR")
        cols.append("PUBLICATION_YEAR")
    return processed_df[cols]


def analyze_filters(
    file_path=DEFAULT_FILE_PATH,
    params=None,
    references_path=DEFAULT_REFERENCES_PATH,
):
    params = params or FilterParams()
    raw_df = read_csv_with_fallback(file_path)
    extracted_df = prepare_extracted_dataframe(raw_df, params.columns_to_extract)
    extracted_df = parse_age_columns(extracted_df)

    stats = []
    append_step(stats, "raw_input", raw_df)
    append_step(stats, "numeric_extract", extracted_df)

    working_df = extracted_df.copy()

    if params.exclude_archean_age:
        # 中文注释：剔除 AGE 含 ARCHEAN（含 Neo/Meso/Paleo/Eoarchean）的太古代样品，避免污染现代玄武岩判别数据。
        archean_mask = working_df["AGE"].astype(str).str.upper().str.contains("ARCHEAN", na=False)
        working_df = working_df[~archean_mask].copy()
        append_step(stats, "after_exclude_archean", working_df)

    if params.use_sio2_filter:
        sio2_mask = (working_df["SIO2(WT%)"] > params.sio2_min) & (
            working_df["SIO2(WT%)"] < params.sio2_max
        )
        working_df = working_df[sio2_mask].copy()
        append_step(stats, "after_sio2", working_df)

    if params.use_mgo_al2o3_filter:
        mgo_mask = (working_df["MGO(WT%)"] >= params.mgo_min) & (
            working_df["MGO(WT%)"] <= params.mgo_max
        )
        al2o3_mask = (working_df["AL2O3(WT%)"] >= params.al2o3_min) & (
            working_df["AL2O3(WT%)"] <= params.al2o3_max
        )
        working_df = working_df[mgo_mask & al2o3_mask].copy()
        append_step(stats, "after_mgo_al2o3", working_df)

    if params.use_loi_filter:
        if params.keep_loi_nan:
            loi_mask = (working_df["LOI(WT%)"] < params.loi_max) | working_df["LOI(WT%)"].isna()
        else:
            loi_mask = working_df["LOI(WT%)"] < params.loi_max
        working_df = working_df[loi_mask].copy()
        append_step(stats, "after_loi", working_df)

    if params.use_min_non_nan_filter:
        count_non_nans = working_df.notna().sum(axis=1)
        working_df = working_df.loc[count_non_nans >= params.min_non_nan].copy()
        append_step(stats, "after_min_non_nan", working_df)

    working_df = finalize_columns(working_df, raw_df)
    append_step(stats, "after_location_finalize", working_df)

    if params.drop_duplicates:
        working_df = working_df.drop_duplicates().copy()
        append_step(stats, "after_drop_duplicates", working_df)

    if params.apply_setting_filter:
        working_df = filter_tectonic_setting(working_df, params)
        append_step(stats, "after_setting_filter", working_df)

    if params.keep_all_columns:
        working_df = restore_original_columns(working_df, raw_df)

    if params.add_publication_year:
        working_df = process_citations(working_df, references_path=references_path)
        append_step(stats, "final_output", working_df)
    else:
        append_step(stats, "final_output", working_df)

    return working_df, stats

