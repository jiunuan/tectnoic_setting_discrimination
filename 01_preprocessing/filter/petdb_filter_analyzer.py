import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
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
SETTING_REPLACEMENTS = {
    "OCEAN_ISLAND": "OCEAN ISLAND",
    "OCEANIC_PLATEAU": "OCEANIC PLATEAU",
}


@dataclass
class FilterParams:
    columns_to_extract: list[str] = field(default_factory=lambda: DEFAULT_COLUMNS_TO_EXTRACT.copy())
    allowed_settings: list[str] = field(default_factory=lambda: DEFAULT_ALLOWED_SETTINGS.copy())
    min_non_nan: int = 10
    use_min_non_nan_filter: bool = True
    use_sio2_filter: bool = False
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


def read_csv_with_fallback(file_path):
    try:
        return pd.read_csv(file_path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding="ISO-8859-1", low_memory=False)


def align_column_names(df, columns_to_extract):
    aligned_df = df.copy()
    missing_columns = []

    for expected_column in columns_to_extract:
        if expected_column in aligned_df.columns:
            continue

        candidates = COLUMN_ALIASES.get(expected_column, [])
        source_column = next((candidate for candidate in candidates if candidate in aligned_df.columns), None)
        if source_column is None:
            missing_columns.append(expected_column)
            continue

        aligned_df[expected_column] = aligned_df[source_column]

    if missing_columns:
        raise KeyError(f"Missing required columns after alias mapping: {missing_columns}")

    return aligned_df


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
    working_df = align_column_names(raw_df, params.columns_to_extract)

    stats = []
    append_step(stats, "raw_input", working_df)

    if params.extract_publication_year and "REFERENCES" in working_df.columns:
        working_df = working_df.copy()
        working_df["YEAR"] = working_df["REFERENCES"].apply(extract_years_from_reference)

    if params.normalize_setting_names and "TECTONIC SETTING" in working_df.columns:
        working_df = working_df.copy()
        working_df["TECTONIC SETTING"] = working_df["TECTONIC SETTING"].replace(SETTING_REPLACEMENTS)
        append_step(stats, "after_setting_normalize", working_df)

    if params.use_min_non_nan_filter:
        count_non_nans = working_df[params.columns_to_extract].notna().sum(axis=1)
        working_df = working_df.loc[count_non_nans >= params.min_non_nan].copy()
        append_step(stats, "after_min_non_nan", working_df)

    if params.use_sio2_filter:
        sio2_mask = working_df["SIO2(WT%)"].isna() | (
            (working_df["SIO2(WT%)"] > params.sio2_min) & (working_df["SIO2(WT%)"] < params.sio2_max)
        )
        working_df = working_df.loc[sio2_mask].copy()
        append_step(stats, "after_sio2", working_df)

    if params.use_mgo_al2o3_filter:
        working_df = working_df[
            (working_df["MGO(WT%)"] >= params.mgo_min)
            & (working_df["MGO(WT%)"] <= params.mgo_max)
            & (working_df["AL2O3(WT%)"] >= params.al2o3_min)
            & (working_df["AL2O3(WT%)"] <= params.al2o3_max)
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
