# -*- coding: utf-8 -*-
import re

import pandas as pd


FE2O3_TO_FEO_FACTOR = 0.8998


def normalize_column_key(column_name):
    """统一列名格式，忽略大小写、空格和下划线差异。"""
    return re.sub(r"[\s_]+", "", str(column_name)).upper()


def find_column(df, expected_name, aliases=None):
    """按标准名及别名查找实际列名。"""
    candidates = [expected_name] + list(aliases or [])
    normalized_columns = {
        normalize_column_key(column): column for column in df.columns
    }
    for candidate in candidates:
        source_column = normalized_columns.get(normalize_column_key(candidate))
        if source_column is not None:
            return source_column
    return None


def add_standard_columns(df, expected_columns, aliases=None):
    """将不同来源的列名映射为筛选流程使用的标准列名。"""
    result = df.copy()
    aliases = aliases or {}
    missing_columns = []

    for expected_column in expected_columns:
        if expected_column in result.columns:
            continue
        source_column = find_column(
            result,
            expected_column,
            aliases.get(expected_column, []),
        )
        if source_column is None:
            missing_columns.append(expected_column)
        else:
            result[expected_column] = result[source_column]

    if missing_columns:
        raise KeyError(f"缺少必要列：{missing_columns}")
    return result


def calculate_feot(df, allow_elemental_fe=False):
    """
    在筛选前按优先级补全 FEOT(WT%)。

    优先级：FeOT；Fe2O3T×0.8998；FeO+Fe2O3×0.8998；
    仅 FeO；仅 Fe2O3×0.8998；PetDB 最后允许直接使用单质 Fe。
    """
    result = df.copy()
    source_names = {
        "feot": ("FEOT(WT%)", ["FeOT (wt%)", "FEOT"]),
        "fe2o3t": ("FE2O3T(WT%)", ["Fe2O3T (wt%)", "FE2O3T"]),
        "feo": ("FEO(WT%)", ["FeO (wt%)", "FEO"]),
        "fe2o3": ("FE2O3(WT%)", ["Fe2O3 (wt%)", "FE2O3"]),
        "fe": ("FE(WT%)", ["Fe (wt%)", "FE"]),
    }

    values = {}
    for key, (expected_name, aliases) in source_names.items():
        source_column = find_column(result, expected_name, aliases)
        if source_column is None:
            values[key] = pd.Series(float("nan"), index=result.index, dtype="float64")
        else:
            values[key] = pd.to_numeric(result[source_column], errors="coerce")

    feot = values["feot"].copy()
    source = pd.Series(pd.NA, index=result.index, dtype="string")
    source.loc[feot.notna()] = "FeOT"

    use_fe2o3t = feot.isna() & values["fe2o3t"].notna()
    feot.loc[use_fe2o3t] = values["fe2o3t"].loc[use_fe2o3t] * FE2O3_TO_FEO_FACTOR
    source.loc[use_fe2o3t] = "Fe2O3T"

    use_feo_fe2o3 = (
        feot.isna() & values["feo"].notna() & values["fe2o3"].notna()
    )
    feot.loc[use_feo_fe2o3] = (
        values["feo"].loc[use_feo_fe2o3]
        + values["fe2o3"].loc[use_feo_fe2o3] * FE2O3_TO_FEO_FACTOR
    )
    source.loc[use_feo_fe2o3] = "FeO+Fe2O3"

    use_feo = feot.isna() & values["feo"].notna()
    feot.loc[use_feo] = values["feo"].loc[use_feo]
    source.loc[use_feo] = "FeO"

    use_fe2o3 = feot.isna() & values["fe2o3"].notna()
    feot.loc[use_fe2o3] = values["fe2o3"].loc[use_fe2o3] * FE2O3_TO_FEO_FACTOR
    source.loc[use_fe2o3] = "Fe2O3"

    if allow_elemental_fe:
        use_fe = feot.isna() & values["fe"].notna()
        feot.loc[use_fe] = values["fe"].loc[use_fe]
        source.loc[use_fe] = "Fe"

    result["FEOT(WT%)"] = feot
    source_counts = source.value_counts(dropna=False).to_dict()
    return result, source_counts
