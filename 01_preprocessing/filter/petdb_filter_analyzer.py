import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from iron_normalization import add_standard_columns, calculate_feot, find_column
from major_element_normalization import normalize_major_oxides_for_filtering
from config.paths import PETDB_RAW_CSV

DEFAULT_FILE_PATH = str(PETDB_RAW_CSV)
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
    "BACK-ARC_BASIN",
    "CONTINENTAL_RIFT",
    "OCEANIC PLATEAU",
    "SPREADING_CENTER",
]
COLUMN_ALIASES = {
    "NA2O(WT%)": ["NA2O"],
    "MGO(WT%)": ["MGO"],
    "AL2O3(WT%)": ["AL2O3"],
    "SIO2(WT%)": ["SIO2"],
    "P2O5(WT%)": ["P2O5"],
    "K2O(WT%)": ["K2O"],
    "CAO(WT%)": ["CAO"],
    "TIO2(WT%)": ["TIO2"],
    "MNO(WT%)": ["MNO"],
    "FEOT(WT%)": ["FEOT"],
    "RB(PPM)": ["RB"],
    "V(PPM)": ["V"],
    "CR(PPM)": ["CR"],
    "CO(PPM)": ["CO", "CO.1"],
    "NI(PPM)": ["NI", "NI.1"],
    "BA(PPM)": ["BA"],
    "SR(PPM)": ["SR"],
    "Y(PPM)": ["Y"],
    "ZR(PPM)": ["ZR"],
    "NB(PPM)": ["NB"],
    "LA(PPM)": ["LA"],
    "CE(PPM)": ["CE"],
    "PR(PPM)": ["PR"],
    "ND(PPM)": ["ND"],
    "SM(PPM)": ["SM"],
    "EU(PPM)": ["EU"],
    "GD(PPM)": ["GD"],
    "TB(PPM)": ["TB"],
    "DY(PPM)": ["DY"],
    "HO(PPM)": ["HO"],
    "ER(PPM)": ["ER"],
    "YB(PPM)": ["YB"],
    "LU(PPM)": ["LU"],
    "HF(PPM)": ["HF"],
    "TA(PPM)": ["TA"],
    "TH(PPM)": ["TH"],
}
METADATA_ALIASES = {
    "TECTONIC SETTING": ["Tectonic Setting"],
    "LONGITUDE": ["Longitude"],
    "LATITUDE": ["Latitude"],
    "AGE": ["Geologic Age"],
    "AGE PREFIX": ["Geologic Age Prefix"],
    "REFERENCES": ["Citation"],
}
PETDB2_IRON_COLUMNS = [
    "FE2O3(WT%)",
    "FE2O3T(WT%)",
    "FEO(WT%)",
    "FEOT(WT%)",
    "FE(WT%)",
]
SETTING_REPLACEMENTS = {
    "OCEAN_ISLAND": "OCEAN ISLAND",
    "OCEANIC_PLATEAU": "OCEANIC PLATEAU",
    "BACK-ARC BASIN": "BACK-ARC_BASIN",
    "CONTINENTAL RIFT": "CONTINENTAL_RIFT",
    "SPREADING CENTER": "SPREADING_CENTER",
}
# 中文注释：主要氧化物，要求非空（不能缺失）。
REQUIRED_MAJOR_OXIDES = ["SIO2(WT%)", "AL2O3(WT%)", "FEOT(WT%)", "MGO(WT%)", "CAO(WT%)"]


@dataclass
class FilterParams:
    columns_to_extract: list[str] = field(default_factory=lambda: DEFAULT_COLUMNS_TO_EXTRACT.copy())
    allowed_settings: list[str] = field(default_factory=lambda: DEFAULT_ALLOWED_SETTINGS.copy())
    min_non_nan: int = 20
    use_min_non_nan_filter: bool = True
    require_major_oxides: bool = True
    use_sio2_filter: bool = True
    sio2_min: float = 45.0
    sio2_max: float = 53.0
    use_mgo_al2o3_filter: bool = True
    mgo_min: float = 4.5
    mgo_max: float = 12
    al2o3_min: float = 12.0
    al2o3_max: float = 19.0
    drop_duplicates: bool = True
    apply_setting_filter: bool = True
    final_count_per_setting: int = 20000
    extract_publication_year: bool = True
    normalize_setting_names: bool = True
    exclude_archean_age: bool = True


def read_csv_with_fallback(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)


def align_column_names(df, columns_to_extract):
    # 中文注释：兼容 PetDB 旧版大写列名和 2.0 版带空格的列名。
    return add_standard_columns(
        df,
        columns_to_extract,
        aliases=COLUMN_ALIASES,
    )


def consolidate_petdb2_rows(df, columns_to_extract):
    """
    合并 PetDB 2.0 中被拆到多行的互补分析值。

    仅合并“样品+文献”内各元素没有多个不同测值的分组；
    存在冲突测值的分组保留原始行，避免混合不同分析结果。
    """
    if "Sample Name" not in df.columns or "Citation" not in df.columns:
        return df.copy(), {
            "safe_groups": 0,
            "conflict_groups": 0,
            "conflict_rows": 0,
        }

    working_df = df.copy()
    working_df["_row_order"] = range(len(working_df))
    working_df["_sample_group_key"] = working_df["Sample Name"].astype("string")

    # 中文注释：空样品名不能互相合并，每一行使用独立内部键。
    empty_sample = working_df["_sample_group_key"].isna()
    working_df.loc[empty_sample, "_sample_group_key"] = [
        f"__EMPTY_SAMPLE_ROW_{index}__" for index in working_df.index[empty_sample]
    ]
    working_df["_citation_group_key"] = (
        working_df["Citation"].astype("string").fillna("__EMPTY_CITATION__")
    )
    group_columns = ["_sample_group_key", "_citation_group_key"]

    chemical_columns = []
    for expected_column in columns_to_extract + PETDB2_IRON_COLUMNS:
        source_column = find_column(
            working_df,
            expected_column,
            COLUMN_ALIASES.get(expected_column, []),
        )
        if source_column is not None and source_column not in chemical_columns:
            chemical_columns.append(source_column)

    numeric_values = working_df[chemical_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    conflict_table = pd.concat(
        [working_df[group_columns], numeric_values],
        axis=1,
    ).groupby(group_columns, dropna=False)[chemical_columns].nunique(dropna=True)
    conflict_groups = conflict_table.gt(1).any(axis=1)
    conflict_index = conflict_groups[conflict_groups].index

    row_group_index = pd.MultiIndex.from_frame(working_df[group_columns])
    conflict_row_mask = row_group_index.isin(conflict_index)

    safe_rows = working_df.loc[~conflict_row_mask]
    consolidated_safe = (
        safe_rows.groupby(group_columns, dropna=False, sort=False)
        .first()
        .reset_index()
    )
    conflict_rows = working_df.loc[conflict_row_mask].copy()
    consolidated_df = pd.concat(
        [consolidated_safe, conflict_rows],
        ignore_index=True,
        sort=False,
    )
    consolidated_df = (
        consolidated_df.sort_values("_row_order")
        .drop(
            columns=[
                "_row_order",
                "_sample_group_key",
                "_citation_group_key",
            ]
        )
        .reset_index(drop=True)
    )

    return consolidated_df, {
        "safe_groups": len(consolidated_safe),
        "conflict_groups": int(conflict_groups.sum()),
        "conflict_rows": int(conflict_row_mask.sum()),
    }


def extract_years_from_reference(ref_str):
    if pd.isna(ref_str):
        return ""
    matches = re.findall(r"[A-Z-]+,\s*(\d{4})", str(ref_str))
    return "; ".join(matches) if matches else ""


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


def filter_tectonic_setting(df, params):
    filtered_df = df.loc[df["TECTONIC SETTING"].isin(params.allowed_settings)].copy()
    if params.final_count_per_setting > 0:
        for setting in params.allowed_settings:
            setting_rows = filtered_df.loc[filtered_df["TECTONIC SETTING"] == setting]
            if len(setting_rows) > params.final_count_per_setting:
                drop_count = len(setting_rows) - params.final_count_per_setting
                to_drop = setting_rows.sample(drop_count, random_state=42).index
                filtered_df = filtered_df.drop(to_drop)
    return filtered_df


def analyze_filters(file_path=DEFAULT_FILE_PATH, params=None):
    params = params or FilterParams()
    raw_df = read_csv_with_fallback(file_path)
    # 中文注释：先统一新版元数据列名，再重组拆行数据并补全 FeOT。
    metadata_columns = ["TECTONIC SETTING", "LONGITUDE", "LATITUDE"]
    raw_df = add_standard_columns(raw_df, metadata_columns, aliases=METADATA_ALIASES)
    for optional_column in ["AGE", "AGE PREFIX", "REFERENCES"]:
        try:
            raw_df = add_standard_columns(
                raw_df,
                [optional_column],
                aliases=METADATA_ALIASES,
            )
        except KeyError:
            pass

    stats = []
    append_step(stats, "raw_input", raw_df)
    raw_df, _ = consolidate_petdb2_rows(raw_df, params.columns_to_extract)
    append_step(stats, "after_petdb2_consolidate", raw_df)
    raw_df, _ = calculate_feot(raw_df, allow_elemental_fe=True)
    working_df = align_column_names(raw_df, params.columns_to_extract)
    working_df[params.columns_to_extract] = working_df[
        params.columns_to_extract
    ].apply(pd.to_numeric, errors="coerce")

    if params.extract_publication_year and "REFERENCES" in working_df.columns:
        working_df = working_df.copy()
        working_df["YEAR"] = working_df["REFERENCES"].apply(extract_years_from_reference)

    if params.normalize_setting_names and "TECTONIC SETTING" in working_df.columns:
        working_df = working_df.copy()
        working_df["TECTONIC SETTING"] = working_df["TECTONIC SETTING"].replace(SETTING_REPLACEMENTS)
        append_step(stats, "after_setting_normalize", working_df)

    if params.exclude_archean_age and "AGE" in working_df.columns:
        # 中文注释：合并年代前缀与年代名称，剔除所有含 ARCHEAN 的太古代样品。
        age_text = working_df["AGE"].fillna("").astype(str)
        if "AGE PREFIX" in working_df.columns:
            age_text = (
                working_df["AGE PREFIX"].fillna("").astype(str)
                + " "
                + age_text
            )
        archean_mask = age_text.str.upper().str.contains("ARCHEAN", na=False)
        working_df = working_df.loc[~archean_mask].copy()
        append_step(stats, "after_exclude_archean", working_df)

    if params.use_min_non_nan_filter:
        count_non_nans = working_df[params.columns_to_extract].notna().sum(axis=1)
        working_df = working_df.loc[count_non_nans >= params.min_non_nan].copy()
        append_step(stats, "after_min_non_nan", working_df)

    if params.require_major_oxides:
        # 中文注释：主要氧化物 SiO2/Al2O3/FeOT/MgO/CaO 不能缺失。
        oxide_ok = working_df[REQUIRED_MAJOR_OXIDES].apply(
            lambda col: pd.to_numeric(col, errors="coerce")
        ).notna().all(axis=1)
        working_df = working_df.loc[oxide_ok].copy()
        append_step(stats, "after_require_major_oxides", working_df)

    normalized_major = None
    if params.use_sio2_filter or params.use_mgo_al2o3_filter:
        # 成分范围必须使用完整主量氧化物换算后的无水 100% 基准值。
        normalized_major, normalization_ok = normalize_major_oxides_for_filtering(working_df)
        working_df = working_df.loc[normalization_ok].copy()
        normalized_major = normalized_major.loc[working_df.index]
        append_step(stats, "after_anhydrous_normalization_ready", working_df)

    if params.use_sio2_filter:
        sio2_mask = normalized_major["SIO2(WT%)"].between(
            params.sio2_min,
            params.sio2_max,
            inclusive="both",
        )
        working_df = working_df.loc[sio2_mask].copy()
        normalized_major = normalized_major.loc[working_df.index]
        append_step(stats, "after_sio2", working_df)

    if params.use_mgo_al2o3_filter:
        working_df = working_df[
            normalized_major["MGO(WT%)"].between(
                params.mgo_min,
                params.mgo_max,
                inclusive="both",
            )
            & normalized_major["AL2O3(WT%)"].between(
                params.al2o3_min,
                params.al2o3_max,
                inclusive="both",
            )
        ].copy()
        append_step(stats, "after_mgo_al2o3", working_df)

    output_columns = params.columns_to_extract.copy()
    for column in ["TECTONIC SETTING", "LONGITUDE", "LATITUDE", "AGE"]:
        if column in working_df.columns and column not in output_columns:
            output_columns.append(column)

    output_df = working_df.loc[:, [column for column in output_columns if column in working_df.columns]].copy()
    output_df = output_df.round(3)
    output_df = output_df.apply(pd.to_numeric, errors="coerce")
    output_df["TECTONIC SETTING"] = working_df.loc[output_df.index, "TECTONIC SETTING"]
    output_df["longitude"] = working_df.loc[output_df.index, "LONGITUDE"]
    output_df["latitude"] = working_df.loc[output_df.index, "LATITUDE"]
    if "AGE" in working_df.columns:
        output_df["AGE"] = working_df.loc[output_df.index, "AGE"]
    if params.extract_publication_year and "YEAR" in working_df.columns:
        output_df["PUBLICATION_YEAR"] = working_df.loc[output_df.index, "YEAR"]
    append_step(stats, "after_finalize", output_df)

    if params.drop_duplicates:
        output_df = output_df.drop_duplicates().copy()
        append_step(stats, "after_drop_duplicates", output_df)

    if params.apply_setting_filter:
        output_df = filter_tectonic_setting(output_df, params)
        append_step(stats, "final_output", output_df)

    return output_df, stats
