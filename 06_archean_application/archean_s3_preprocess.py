from __future__ import annotations

# === ?????????? config/paths.py????????????===
import sys as _cfg_sys
from pathlib import Path as _cfg_Path
_cfg_sys.path.insert(0, str(_cfg_Path(__file__).resolve().parents[1]))
from config.paths import (
    ARCHEAN_DIR, TRAIN_NORM_CSV, TEST_NORM_CSV,
    MAIN_MODEL_WEIGHT, QUANTILE_PARAMS_JSON,
)

# ──────────────────────────────────────────────────────────────────────────────
# 标准库导入
# ──────────────────────────────────────────────────────────────────────────────
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# 第三方库导入
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
#  ★ 配置区：所有路径均直接写完整地址
# ══════════════════════════════════════════════════════════════════════════════

# ── 1. Liu 2024 全球太古代数据集 ─────────────────────────────────────────────
SOURCE_S3_CSV_PATH = (ARCHEAN_DIR / "data/archean_basalt.csv")

# quantile 归一化参数 JSON（由训练集生成，所有应用集共用）
QUANTILE_PATH = QUANTILE_PARAMS_JSON

# 全部训练样品共同训练得到的全局 MissForest 插补器
GLOBAL_MISSFOREST_MODEL_PATH = ARCHEAN_DIR / "global_train_missforest.joblib"

# Liu 2024 全球数据集预处理模式开关：
#   "imputed" / "no_impute" / "both"
PREPROCESS_MODE = "no_impute"

# 太古代样品缺失特征数必须小于此值；缺失数超过即剔除
MAX_MISSING_FEATURES_EXCLUSIVE = 16

# Liu 2024 全球数据集预处理输出（插补 / 不插补）
IMPUTED_PREPROCESS_OUTPUT_DIR = (ARCHEAN_DIR / "outputs/vit_transformer_v1/preprocess_imputed")
IMPUTED_MODEL_INPUT_S3_PATH = IMPUTED_PREPROCESS_OUTPUT_DIR / "archean_s3_model_input.csv"
IMPUTED_QUANTILE_FEATURES_PATH = IMPUTED_PREPROCESS_OUTPUT_DIR / "archean_features_quantile_1_255.csv"

NO_IMPUTE_PREPROCESS_OUTPUT_DIR = (ARCHEAN_DIR / "outputs/vit_transformer_v1/preprocess_no_impute")
NO_IMPUTE_MODEL_INPUT_S3_PATH = NO_IMPUTE_PREPROCESS_OUTPUT_DIR / "archean_s3_model_input.csv"
NO_IMPUTE_QUANTILE_FEATURES_PATH = NO_IMPUTE_PREPROCESS_OUTPUT_DIR / "archean_features_quantile_1_255.csv"


# ── 2. 6 个克拉通案例研究 ────────────────────────────────────────────────────
# 是否运行案例研究的预处理（True 时除 Liu 2024 外，还会处理 6 个案例 CSV）
RUN_CASE_STUDIES = True

# 案例 CSV 所在目录
CASE_STUDY_INPUT_DIR = (ARCHEAN_DIR / "data")

# 案例研究预处理输出根目录；每个案例会建一个子目录
CASE_STUDY_OUTPUT_ROOT = (ARCHEAN_DIR / "outputs/vit_transformer_v1/case_studies")

# 6 个案例的 (case_label, csv_filename, approx_age_ga) 配置
# 注意：case_label 用作输出目录名，避免使用文件系统不友好字符（"&" 替换为 "_"）
# 案例顺序按地质年龄从老到新排列（Isua 最老 → North_China_Craton 最年轻）
CASE_STUDIES: list[tuple[str, str, float]] = [
    ("Isua",                 "Isua.csv",                 3.75),  # Isua / West Greenland
    ("Barberton",            "Barberton.csv",            3.46),  # Barberton / Kaapvaal
    ("Pilbara",              "Pilbara.csv",              3.38),  # Pilbara 整体，替代 Cleaverville + Whundo
    ("Belingwe_Zimbabwe",    "Belingwe_Zimbabwe.csv",    2.87),  # Belingwe + Zimbabwe
    ("Superior_Abitibi",     "Superior_Abitibi.csv",     2.72),  # Superior，含 Abitibi
    ("North_China_Craton",   "North_China_Craton.csv",   2.54),  # North China Craton，含 Wutai 等
]

# 案例研究只生成插补版本（不插补对小样本案例意义不大）
CASE_STUDY_USE_MISSFOREST = True


# ──────────────────────────────────────────────────────────────────────────────
# 特征列定义
# ──────────────────────────────────────────────────────────────────────────────

# 模型输入的36个地球化学特征列（矩阵分支：6×6 元素周期表顺序，与训练时 ORIGINAL_IMAGE_COLUMNS 一致）
COLUMNS_TO_EXTRACT = [
    'NA2O(WT%)', 'MGO(WT%)',   'CR(PPM)',    'AL2O3(WT%)', 'SIO2(WT%)',  'P2O5(WT%)',
    'K2O(WT%)',  'CAO(WT%)',   'TIO2(WT%)',  'V(PPM)',     'MNO(WT%)',   'FEOT(WT%)',
    'RB(PPM)',   'SR(PPM)',    'Y(PPM)',     'NB(PPM)',    'CO(PPM)',    'NI(PPM)',
    'BA(PPM)',   'LA(PPM)',    'CE(PPM)',    'PR(PPM)',    'ND(PPM)',    'ZR(PPM)',
    'SM(PPM)',   'EU(PPM)',    'GD(PPM)',    'TB(PPM)',    'DY(PPM)',    'HO(PPM)',
    'TH(PPM)',   'ER(PPM)',    'YB(PPM)',    'LU(PPM)',    'HF(PPM)',    'TA(PPM)',
]

# 主量元素列，用于归一化到 100 wt%
MAJOR_COLUMNS = [
    "NA2O(WT%)", "MGO(WT%)",   "AL2O3(WT%)", "SIO2(WT%)",
    "P2O5(WT%)", "K2O(WT%)",   "CAO(WT%)",   "TIO2(WT%)",
    "MNO(WT%)",  "FEOT(WT%)",
]

# Liu 2024 archean_basalt.csv 的列名映射（短名 → 训练集长名）
# 案例 CSV 不需要这个映射（列名直接就是训练集长名）
LIU2024_COLUMN_MAPPING = {
    "SIO2":"SIO2(WT%)",   "TIO2":"TIO2(WT%)",   "AL2O3":"AL2O3(WT%)",
    "FEOT":"FEOT(WT%)",   "MNO":"MNO(WT%)",     "MGO":"MGO(WT%)",
    "CAO":"CAO(WT%)",     "NA2O":"NA2O(WT%)",   "K2O":"K2O(WT%)",
    "P2O5":"P2O5(WT%)",   "V":"V(PPM)",         "CR":"CR(PPM)",
    "CO":"CO(PPM)",       "NI":"NI(PPM)",       "RB":"RB(PPM)",
    "SR":"SR(PPM)",       "Y":"Y(PPM)",         "ZR":"ZR(PPM)",
    "NB":"NB(PPM)",       "BA":"BA(PPM)",       "LA":"LA(PPM)",
    "CE":"CE(PPM)",       "PR":"PR(PPM)",       "ND":"ND(PPM)",
    "SM":"SM(PPM)",       "EU":"EU(PPM)",       "GD":"GD(PPM)",
    "TB":"TB(PPM)",       "DY":"DY(PPM)",       "HO":"HO(PPM)",
    "ER":"ER(PPM)",       "YB":"YB(PPM)",       "LU":"LU(PPM)",
    "HF":"HF(PPM)",       "TA":"TA(PPM)",       "TH":"TH(PPM)",
}


# ──────────────────────────────────────────────────────────────────────────────
# 配置数据类
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreprocessConfig:
    """
    预处理任务的配置。

    column_mapping = None 表示输入 CSV 的列名已经是训练集长名格式（如 'SIO2(WT%)'），
    直接选列即可（案例研究 CSV 走这条路径）。
    column_mapping = dict 表示走 "短名 → 长名" 映射（Liu 2024 archean_basalt.csv 走这条路径）。

    case_label 仅用于日志/输出文件名前缀；非 case study 时为空字符串。
    """
    source_s3_csv_path: Path
    quantile_path: Path
    global_missforest_model_path: Path
    preprocess_output_dir: Path
    model_input_s3_path: Path
    quantile_features_path: Path
    max_missing_features_exclusive: int
    use_missforest_imputation: bool
    column_mapping: Optional[dict[str, str]] = None
    case_label: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# I/O 与核心算法（保留原实现）
# ──────────────────────────────────────────────────────────────────────────────

def read_csv_fallback(path: Path) -> pd.DataFrame:
    """读取 CSV，优先 UTF-8，失败则回退到 ISO-8859-1。"""
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="ISO-8859-1")


def normalize_major_elements(df: pd.DataFrame) -> pd.DataFrame:
    """将主量元素归一化到 100 wt%，主量总和为 0 的样品置为 NaN。"""
    out = df.copy()
    totals = out[MAJOR_COLUMNS].sum(axis=1, skipna=True).replace(0, np.nan)
    out[MAJOR_COLUMNS] = out[MAJOR_COLUMNS].div(totals, axis=0) * 100.0
    return out


def clean_for_missforest(X: pd.DataFrame) -> pd.DataFrame:
    """复用训练脚本 MissForest 的输入清洗策略。"""
    X_clean = X.replace([np.inf, -np.inf], np.finfo(np.float32).max)
    return X_clean.fillna(0)


def missforest_transform_bundle(features: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """使用全局 MissForest 模型包插补 S3 特征。"""
    scaler = bundle["scaler"]
    imputers = bundle["imputers"]
    column_order = list(bundle["column_order"])

    X_raw = features[column_order].copy()
    X_scaled = pd.DataFrame(
        scaler.transform(X_raw),
        columns=column_order,
        index=X_raw.index,
    )
    X_clean = clean_for_missforest(X_scaled)
    X_imputed = X_clean.copy()

    for col in column_order:
        missing = X_scaled[col].isna()
        if missing.any() and col in imputers:
            feature_cols = [c for c in column_order if c != col]
            X_imputed.loc[missing, col] = imputers[col].predict(
                X_clean.loc[missing, feature_cols]
            )

    X_original = pd.DataFrame(
        scaler.inverse_transform(X_imputed),
        columns=column_order,
        index=X_imputed.index,
    )
    return X_original[COLUMNS_TO_EXTRACT]


def impute_s3_by_global_missforest(features: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    """对无标签太古代 S3 使用一个全局 MissForest 模型插补。"""
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("需要安装 joblib 才能加载 MissForest 插补模型。") from exc

    if not model_path.exists():
        raise FileNotFoundError(
            f"未找到全局 MissForest 插补模型: {model_path}\n"
            "请先运行 E:\\program\\CNNtest\\data_interpolation\\train_global_missforest.py"
        )

    print(f"      插补模型: {model_path}")
    bundle = joblib.load(model_path)
    imputed = missforest_transform_bundle(features, bundle)
    return imputed.clip(lower=0.0)


def apply_quantile_transform(df: pd.DataFrame, params: dict[str, list[float]]) -> pd.DataFrame:
    """对每个地化特征列执行 quantile 分箱变换，将值映射到 1~255。"""
    transformed = pd.DataFrame(index=df.index)
    for col in COLUMNS_TO_EXTRACT:
        if col not in params:
            raise KeyError(f"缺少列 {col} 的 quantile 参数")
        q = np.asarray(params[col], dtype=np.float64)
        values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        mapped = np.searchsorted(q, values, side="left") + 1
        mapped[np.isnan(values)] = 0
        transformed[col] = np.clip(mapped, 0, 255).astype(np.float32)
    return transformed


def extract_and_filter_features(
    s3: pd.DataFrame, config: PreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    从原始 CSV 中提取36个训练特征，并按缺失数过滤样品。

    根据 config.column_mapping 选择两种提取方式：
      · None：CSV 列名已是 'SIO2(WT%)' 等训练集长名 → 直接选列（案例研究走这条）
      · dict：走 "短名 → 长名" 映射（Liu 2024 走这条）

    缺失列在两种路径下均允许存在；缺列不报错而是填 NaN，
    样品按 missing_feature_count_36 < max_missing_features_exclusive 过滤。
    """
    features = pd.DataFrame(index=s3.index)

    if config.column_mapping is None:
        # 案例 CSV：列名已是训练集长名格式
        for train_col in COLUMNS_TO_EXTRACT:
            if train_col in s3.columns:
                features[train_col] = pd.to_numeric(s3[train_col], errors="coerce")
            else:
                # 整列缺失 → 填 NaN，由后续 MissForest 处理
                features[train_col] = np.nan
    else:
        # Liu 2024 CSV：走短名 → 长名映射
        for s3_col, train_col in config.column_mapping.items():
            if s3_col not in s3.columns:
                raise KeyError(f"输入 CSV 缺少必需列: {s3_col}")
            features[train_col] = pd.to_numeric(s3[s3_col], errors="coerce")

    features = normalize_major_elements(features[COLUMNS_TO_EXTRACT].copy())
    missing_counts = features[COLUMNS_TO_EXTRACT].isna().sum(axis=1).astype(int)
    keep_mask = missing_counts < config.max_missing_features_exclusive

    s3_model = s3.loc[keep_mask].copy()
    s3_model["missing_feature_count_36"] = missing_counts.loc[keep_mask].to_numpy()

    raw_features = features.loc[keep_mask].copy()
    return s3_model, raw_features


# ──────────────────────────────────────────────────────────────────────────────
# 主流水线
# ──────────────────────────────────────────────────────────────────────────────

def run(config: PreprocessConfig) -> None:
    """执行预处理：缺失过滤、MissForest 插补和 quantile 变换。"""
    config.preprocess_output_dir.mkdir(parents=True, exist_ok=True)

    label = f"[{config.case_label}] " if config.case_label else ""
    print(f"[1/4] {label}读取 CSV: {config.source_s3_csv_path}")
    s3 = read_csv_fallback(config.source_s3_csv_path)
    print(f"      原始样品数: {len(s3)}")

    print(f"[2/4] {label}提取 36 个特征并按缺失数过滤")
    s3_model, raw_features = extract_and_filter_features(s3, config)
    print(
        "      缺失过滤: "
        f"{len(s3_model)}/{len(s3)} 样品进入模型 "
        f"(missing_feature_count_36 < {config.max_missing_features_exclusive})"
    )
    if len(s3_model) == 0:
        print(f"      ⚠ {label}没有样品通过缺失过滤，跳过该任务。")
        return

    if config.use_missforest_imputation:
        print(f"[3/4] {label}执行全局 MissForest 插补")
        imputed_features = impute_s3_by_global_missforest(
            raw_features,
            config.global_missforest_model_path,
        )
    else:
        print(f"[3/4] {label}跳过缺失值插补，缺失值将在 quantile 后编码为 0")
        imputed_features = raw_features.copy()

    print(f"[4/4] {label}执行 quantile 变换并保存预处理结果")
    with config.quantile_path.open("r", encoding="utf-8") as f:
        params = json.load(f)
    quantile_features = apply_quantile_transform(imputed_features, params)

    # 只保存后续预测脚本真正读取的两张表，避免输出多余中间文件。
    s3_model.to_csv(config.model_input_s3_path, index=False, encoding="utf-8-sig")
    quantile_features.to_csv(config.quantile_features_path, index=False)

    zero_count = int((quantile_features[COLUMNS_TO_EXTRACT] == 0).sum().sum())
    print(f"      quantile 后 0 值数量: {zero_count}")
    print(f"      {label}完成。预处理输出: {config.preprocess_output_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# 配置工厂
# ──────────────────────────────────────────────────────────────────────────────

def build_imputed_config() -> PreprocessConfig:
    """Liu 2024 全球数据集 — MissForest 插补版本。"""
    return PreprocessConfig(
        source_s3_csv_path=SOURCE_S3_CSV_PATH,
        quantile_path=QUANTILE_PATH,
        global_missforest_model_path=GLOBAL_MISSFOREST_MODEL_PATH,
        preprocess_output_dir=IMPUTED_PREPROCESS_OUTPUT_DIR,
        model_input_s3_path=IMPUTED_MODEL_INPUT_S3_PATH,
        quantile_features_path=IMPUTED_QUANTILE_FEATURES_PATH,
        max_missing_features_exclusive=MAX_MISSING_FEATURES_EXCLUSIVE,
        use_missforest_imputation=True,
        column_mapping=LIU2024_COLUMN_MAPPING,
        case_label="",
    )


def build_no_impute_config() -> PreprocessConfig:
    """Liu 2024 全球数据集 — 不插补版本。"""
    return PreprocessConfig(
        source_s3_csv_path=SOURCE_S3_CSV_PATH,
        quantile_path=QUANTILE_PATH,
        global_missforest_model_path=GLOBAL_MISSFOREST_MODEL_PATH,
        preprocess_output_dir=NO_IMPUTE_PREPROCESS_OUTPUT_DIR,
        model_input_s3_path=NO_IMPUTE_MODEL_INPUT_S3_PATH,
        quantile_features_path=NO_IMPUTE_QUANTILE_FEATURES_PATH,
        max_missing_features_exclusive=MAX_MISSING_FEATURES_EXCLUSIVE,
        use_missforest_imputation=False,
        column_mapping=LIU2024_COLUMN_MAPPING,
        case_label="",
    )


def build_case_study_config(case_label: str, csv_filename: str) -> PreprocessConfig:
    """
    案例研究的预处理配置。
    输出目录结构：CASE_STUDY_OUTPUT_ROOT / {case_label} /
        ├ s3_model_input.csv
        ├ features_quantile_1_255.csv
    """
    src = CASE_STUDY_INPUT_DIR / csv_filename
    out_dir = CASE_STUDY_OUTPUT_ROOT / case_label
    return PreprocessConfig(
        source_s3_csv_path=src,
        quantile_path=QUANTILE_PATH,
        global_missforest_model_path=GLOBAL_MISSFOREST_MODEL_PATH,
        preprocess_output_dir=out_dir,
        model_input_s3_path=out_dir / "s3_model_input.csv",
        quantile_features_path=out_dir / "features_quantile_1_255.csv",
        max_missing_features_exclusive=MAX_MISSING_FEATURES_EXCLUSIVE,
        use_missforest_imputation=CASE_STUDY_USE_MISSFOREST,
        column_mapping=None,  # 案例 CSV 列名已是训练集长名格式
        case_label=case_label,
    )


def selected_configs(preprocess_mode: str) -> list[PreprocessConfig]:
    """根据预处理模式开关返回 Liu 2024 全球数据集的配置列表。"""
    if preprocess_mode == "imputed":
        return [build_imputed_config()]
    if preprocess_mode == "no_impute":
        return [build_no_impute_config()]
    if preprocess_mode == "both":
        return [build_imputed_config(), build_no_impute_config()]
    raise ValueError("PREPROCESS_MODE 只能是 'imputed'、'no_impute' 或 'both'")


def selected_case_configs() -> list[PreprocessConfig]:
    """返回 6 个克拉通案例研究的预处理配置列表。"""
    return [build_case_study_config(label, fname) for label, fname, _ in CASE_STUDIES]


if __name__ == "__main__":
    try:
        # ── Liu 2024 全球数据集预处理 ────────────────────────────────────────
        for config in selected_configs(PREPROCESS_MODE):
            print("=" * 80)
            print(f"[Liu 2024 全球数据集 | 模式] "
                  f"{'MissForest 插补' if config.use_missforest_imputation else '不插补'}")
            run(config)

        # ── 6 个克拉通案例研究预处理 ─────────────────────────────────────────
        if RUN_CASE_STUDIES:
            print("\n" + "#" * 80)
            print(f"# 案例研究预处理（共 {len(CASE_STUDIES)} 个克拉通案例）")
            print("#" * 80)
            for case_config in selected_case_configs():
                print("=" * 80)
                print(f"[案例] {case_config.case_label}  "
                      f"({case_config.source_s3_csv_path.name})")
                if not case_config.source_s3_csv_path.exists():
                    print(f"  ⚠ 输入 CSV 不存在，跳过: {case_config.source_s3_csv_path}")
                    continue
                run(case_config)
    except Exception as exc:
        print(f"[ERROR] {exc}")
        raise
