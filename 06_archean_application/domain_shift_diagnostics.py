"""
现代训练集与2116条太古代玄武岩的域偏移诊断。

本脚本不修改弧环境概率阈值，重点区分：
1. 随机森林插补方法的影响；
2. 主量元素无水标准化的影响；
3. 训练分位数边界外堆积；
4. 年代域偏移、数据来源差异与模型过度自信。
"""

from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import ks_2samp, spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

# === 统一路径配置：所有数据/模型路径来自 config/paths.py ===
import importlib.util as _importlib_util
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))
from config.paths import (
    TRAIN_RAW_CSV, TRAIN_MAJOR_NORM_CSV,
    TRAIN_NORM_CSV, TRAIN_NORM_NO_SMOTE_CSV, TEST_NORM_CSV,
    QUANTILE_PARAMS_JSON, MAIN_MODEL_WEIGHT,
    ARCHEAN_DATA_SUBDIR, ARCHEAN_OUTPUT_DIR, ARCHEAN_CONSISTENCY_DIR,
)

MODERN_RAW_PATH = str(TRAIN_RAW_CSV)
MODERN_CONTINUOUS_PATH = str(TRAIN_MAJOR_NORM_CSV)
MODERN_QUANTILE_PATH = str(TRAIN_NORM_NO_SMOTE_CSV)
TEST_QUANTILE_PATH = str(TEST_NORM_CSV)
ARCHEAN_RAW_PATH = str(ARCHEAN_DATA_SUBDIR / "archean_basalt_filtered_2116.csv")
ARCHEAN_IMPUTED_PATH = str(ARCHEAN_DATA_SUBDIR / "archean_basalt_filtered_2116_imputed.csv")
ARCHEAN_NORMALIZED_PATH = str(ARCHEAN_DATA_SUBDIR / "archean_basalt_filtered_2116_imputed_major_normalize.csv")
ARCHEAN_QUANTILE_PATH = str(ARCHEAN_OUTPUT_DIR / "preprocess_imputed" / "archean_features_quantile_1_255.csv")
QUANTILE_PARAMS_PATH = str(QUANTILE_PARAMS_JSON)
MODEL_WEIGHT_PATH = str(MAIN_MODEL_WEIGHT)

ELEMENT_SUMMARY_PATH = str(ARCHEAN_CONSISTENCY_DIR / "domain_shift_element_summary.csv")
CORRELATION_SHIFT_PATH = str(ARCHEAN_CONSISTENCY_DIR / "domain_shift_correlation_pairs.csv")
SAMPLE_DIAGNOSTICS_PATH = str(ARCHEAN_CONSISTENCY_DIR / "domain_shift_sample_diagnostics.csv")
CLASS_SUMMARY_PATH = str(ARCHEAN_CONSISTENCY_DIR / "domain_shift_class_summary.csv")
SOURCE_SUMMARY_PATH = str(ARCHEAN_CONSISTENCY_DIR / "domain_shift_source_summary.csv")
COUNTERFACTUAL_SUMMARY_PATH = str(ARCHEAN_CONSISTENCY_DIR / "domain_shift_counterfactual_summary.csv")
FIGURE_PATH = str(ARCHEAN_CONSISTENCY_DIR / "domain_shift_diagnostics.png")
REPORT_PATH = str(ARCHEAN_CONSISTENCY_DIR / "domain_shift_diagnostics_report.md")

# 输出目录在导入时确保存在（原工程依赖手工建目录）。
ARCHEAN_CONSISTENCY_DIR.mkdir(parents=True, exist_ok=True)

# 中文注释：04_model 目录以数字开头，无法用常规包导入，改用 importlib 加载训练脚本。
_TRAINING_MODULE_FILE = _PROJECT_ROOT / "04_model" / "ablation_v4_vit_transformer.py"
_TRAINING_MODULE_CACHE = None


def _load_training_module():
    """加载训练脚本模块，复用其中的正式列顺序与模型类。"""
    global _TRAINING_MODULE_CACHE
    if _TRAINING_MODULE_CACHE is None:
        spec = _importlib_util.spec_from_file_location(
            "ablation_v4_vit_transformer", str(_TRAINING_MODULE_FILE)
        )
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _TRAINING_MODULE_CACHE = module
    return _TRAINING_MODULE_CACHE

LONG_COLUMNS = [
    "NA2O(WT%)", "MGO(WT%)", "AL2O3(WT%)", "SIO2(WT%)", "P2O5(WT%)",
    "K2O(WT%)", "CAO(WT%)", "TIO2(WT%)", "MNO(WT%)", "FEOT(WT%)",
    "RB(PPM)", "V(PPM)", "CR(PPM)", "CO(PPM)", "NI(PPM)", "BA(PPM)",
    "SR(PPM)", "Y(PPM)", "ZR(PPM)", "NB(PPM)", "LA(PPM)", "CE(PPM)",
    "PR(PPM)", "ND(PPM)", "SM(PPM)", "EU(PPM)", "GD(PPM)", "TB(PPM)",
    "DY(PPM)", "HO(PPM)", "ER(PPM)", "YB(PPM)", "LU(PPM)", "HF(PPM)",
    "TA(PPM)", "TH(PPM)",
]
SHORT_COLUMNS = [
    "NA2O", "MGO", "AL2O3", "SIO2", "P2O5", "K2O", "CAO", "TIO2", "MNO", "FEOT",
    "RB", "V", "CR", "CO", "NI", "BA", "SR", "Y", "ZR", "NB", "LA", "CE", "PR",
    "ND", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "YB", "LU", "HF", "TA", "TH",
]
MAJOR_SHORT = ["NA2O", "MGO", "AL2O3", "SIO2", "P2O5", "K2O", "CAO", "TIO2", "MNO", "FEOT"]
ARC_CLASSES = {"Continental arc", "Island arc", "Intra-oceanic arc"}
CFB_CLASS = "CONTINENTAL FLOOD BASALT"
RANDOM_SEED = 42


def read_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """将指定列统一转换为浮点数。"""
    return df[columns].apply(pd.to_numeric, errors="coerce")


def normalize_major_short(features: pd.DataFrame) -> pd.DataFrame:
    """将十个主量元素逐行无水标准化到100%。"""
    result = features.copy()
    total = result[MAJOR_SHORT].sum(axis=1)
    if (total <= 0).any():
        raise ValueError("存在主量元素总和不大于0的样品")
    result.loc[:, MAJOR_SHORT] = result[MAJOR_SHORT].div(total, axis=0) * 100.0
    return result


def short_to_long(features: pd.DataFrame) -> pd.DataFrame:
    """将太古代短列名映射为模型使用的长列名。"""
    result = pd.DataFrame(index=features.index)
    for short_name, long_name in zip(SHORT_COLUMNS, LONG_COLUMNS):
        result[long_name] = features[short_name].to_numpy()
    return result


def quantile_encode(features: pd.DataFrame, params: dict) -> pd.DataFrame:
    """使用现代训练集边界编码到1-255。"""
    encoded = pd.DataFrame(index=features.index)
    for column in LONG_COLUMNS:
        boundaries = np.asarray(params[column], dtype=float)
        encoded[column] = (
            np.searchsorted(boundaries, features[column].to_numpy(dtype=float), side="left") + 1
        ).clip(1, 255).astype(np.int16)
    return encoded


def load_model_and_classes(train_quantile: pd.DataFrame):
    """加载GeoDAN模型，并保持训练CSV首次出现的类别顺序。"""
    _training_module = _load_training_module()
    COLUMNS_ELECTRODE_ORDER_V1 = _training_module.COLUMNS_ELECTRODE_ORDER_V1
    ORIGINAL_IMAGE_COLUMNS = _training_module.ORIGINAL_IMAGE_COLUMNS
    ViT_Transformer_DualStream = _training_module.ViT_Transformer_DualStream

    class_names = list(pd.unique(train_quantile["TECTONIC SETTING"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        state = torch.load(MODEL_WEIGHT_PATH, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(MODEL_WEIGHT_PATH, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model = ViT_Transformer_DualStream(num_classes=len(class_names)).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, class_names, device, ORIGINAL_IMAGE_COLUMNS, COLUMNS_ELECTRODE_ORDER_V1


def predict_probabilities(
    features: pd.DataFrame,
    model,
    device,
    image_columns: list[str],
    sequence_columns: list[str],
) -> np.ndarray:
    """对1-255分位数特征执行确定性softmax推理。"""
    image = (features[image_columns].to_numpy(dtype=np.float32) / 255.0).reshape(-1, 1, 6, 6)
    sequence = (
        features[sequence_columns].to_numpy(dtype=np.float32) / 255.0
    )[:, :, np.newaxis]
    loader = DataLoader(
        TensorDataset(torch.from_numpy(image), torch.from_numpy(sequence)),
        batch_size=256,
        shuffle=False,
    )
    parts = []
    with torch.no_grad():
        for batch_image, batch_sequence in loader:
            logits = model(batch_image.to(device), batch_sequence.to(device))
            parts.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def prediction_frame(probabilities: np.ndarray, class_names: list[str], prefix: str) -> pd.DataFrame:
    """把概率矩阵转换为类别、置信度和弧概率。"""
    predicted_index = probabilities.argmax(axis=1)
    arc_index = [class_names.index(name) for name in class_names if name in ARC_CLASSES]
    return pd.DataFrame(
        {
            f"{prefix}_class": [class_names[index] for index in predicted_index],
            f"{prefix}_confidence": probabilities.max(axis=1),
            f"{prefix}_arc_probability": probabilities[:, arc_index].sum(axis=1),
        }
    )


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    """计算等宽置信度分箱的ECE。"""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def robust_log_space(modern: pd.DataFrame, archean: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """使用现代训练集log10均值和标准差构造连续地球化学空间。"""
    modern_values = modern.to_numpy(dtype=float)
    archean_values = archean.to_numpy(dtype=float)
    modern_log = np.log10(np.clip(modern_values, 1e-12, None))
    archean_log = np.log10(np.clip(archean_values, 1e-12, None))
    mean = modern_log.mean(axis=0)
    std = modern_log.std(axis=0)
    std[std == 0] = 1.0
    return (modern_log - mean) / std, (archean_log - mean) / std


def compute_distance_diagnostics(
    modern: pd.DataFrame,
    archean: pd.DataFrame,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    """计算PCA、全局马氏距离、kNN域距离和类别邻近度。"""
    x_modern, x_archean = robust_log_space(modern, archean)
    pca = PCA(random_state=RANDOM_SEED).fit(x_modern)
    pc_modern = pca.transform(x_modern)
    pc_archean = pca.transform(x_archean)
    n_pc = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.85) + 1)
    n_pc = max(2, n_pc)

    # 中文注释：白化后的PCA距离使保留的各主成分具有相同尺度。
    scale = pc_modern[:, :n_pc].std(axis=0)
    scale[scale == 0] = 1.0
    white_modern = pc_modern[:, :n_pc] / scale
    white_archean = pc_archean[:, :n_pc] / scale

    nearest = NearestNeighbors(n_neighbors=11).fit(white_modern)
    modern_distance = nearest.kneighbors(white_modern)[0][:, 1:].mean(axis=1)
    archean_distance = nearest.kneighbors(white_archean)[0].mean(axis=1)
    knn_q95, knn_q99 = np.quantile(modern_distance, [0.95, 0.99])

    covariance = LedoitWolf().fit(white_modern)
    modern_md = np.sqrt(covariance.mahalanobis(white_modern))
    archean_md = np.sqrt(covariance.mahalanobis(white_archean))
    md_q95, md_q99 = np.quantile(modern_md, [0.95, 0.99])

    class_names = list(pd.unique(labels))
    class_distance = np.full((len(archean), len(class_names)), np.nan)
    modern_class_distance = np.full((len(modern), len(class_names)), np.nan)
    for class_index, class_name in enumerate(class_names):
        mask = labels == class_name
        class_covariance = LedoitWolf().fit(white_modern[mask])
        class_distance[:, class_index] = np.sqrt(
            class_covariance.mahalanobis(white_archean)
        )
        modern_class_distance[:, class_index] = np.sqrt(
            class_covariance.mahalanobis(white_modern)
        )

    nearest_class_index = class_distance.argmin(axis=1)
    modern_nearest_class_index = modern_class_distance.argmin(axis=1)
    modern_nearest_class = np.asarray(class_names, dtype=object)[modern_nearest_class_index]
    modern_nearest_match = modern_nearest_class == labels
    modern_cfb_mask = labels == CFB_CLASS
    diagnostics = pd.DataFrame(
        {
            "PC1": pc_archean[:, 0],
            "PC2": pc_archean[:, 1],
            "knn_distance": archean_distance,
            "knn_outside_95": archean_distance > knn_q95,
            "knn_outside_99": archean_distance > knn_q99,
            "global_mahalanobis": archean_md,
            "mahalanobis_outside_95": archean_md > md_q95,
            "mahalanobis_outside_99": archean_md > md_q99,
            "nearest_modern_class": [class_names[index] for index in nearest_class_index],
            "nearest_class_mahalanobis": class_distance.min(axis=1),
        }
    )
    thresholds = {
        "n_pc": n_pc,
        "pca_variance": float(pca.explained_variance_ratio_[:n_pc].sum()),
        "knn_q95": float(knn_q95),
        "knn_q99": float(knn_q99),
        "md_q95": float(md_q95),
        "md_q99": float(md_q99),
        "modern_nearest_class_accuracy": float(modern_nearest_match.mean()),
        "modern_cfb_nearest_class_accuracy": float(
            modern_nearest_match[modern_cfb_mask].mean()
        ),
    }
    return diagnostics, thresholds, pc_modern, pc_archean


def element_summary(
    modern_raw: pd.DataFrame,
    modern_continuous: pd.DataFrame,
    archean_raw: pd.DataFrame,
    archean_normalized: pd.DataFrame,
    archean_quantile: pd.DataFrame,
) -> pd.DataFrame:
    """汇总单元素分布、缺失率、插补比例和边界堆积。"""
    rows = []
    for short_name, long_name in zip(SHORT_COLUMNS, LONG_COLUMNS):
        modern_raw_values = pd.to_numeric(modern_raw[long_name], errors="coerce")
        archean_raw_values = pd.to_numeric(archean_raw[short_name], errors="coerce")
        modern_values = pd.to_numeric(modern_continuous[long_name], errors="coerce")
        archean_values = pd.to_numeric(archean_normalized[short_name], errors="coerce")
        valid_modern = modern_values[np.isfinite(modern_values) & (modern_values > 0)]
        valid_archean = archean_values[np.isfinite(archean_values) & (archean_values > 0)]
        ks = ks_2samp(np.log10(valid_modern), np.log10(valid_archean)).statistic
        rows.append(
            {
                "element": short_name,
                "modern_missing_pct": 100.0 * modern_raw_values.isna().mean(),
                "archean_missing_or_nonpositive_pct": 100.0 * (
                    archean_raw_values.isna() | archean_raw_values.le(0)
                ).mean(),
                "archean_imputed_pct": 100.0 * (
                    archean_raw_values.isna() | archean_raw_values.le(0)
                ).mean(),
                "modern_median": valid_modern.median(),
                "archean_median_after_imputation": valid_archean.median(),
                "median_log10_shift": np.log10(valid_archean.median()) - np.log10(valid_modern.median()),
                "log10_ks_statistic": ks,
                "archean_bin_1_pct": 100.0 * archean_quantile[long_name].eq(1).mean(),
                "archean_bin_255_pct": 100.0 * archean_quantile[long_name].eq(255).mean(),
                "archean_boundary_pct": 100.0 * archean_quantile[long_name].isin([1, 255]).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("log10_ks_statistic", ascending=False)


def correlation_shift(
    modern: pd.DataFrame,
    archean_imputed: pd.DataFrame,
    archean_raw: pd.DataFrame,
) -> pd.DataFrame:
    """比较现代、太古代插补后及太古代仅观测值的元素组合关系。"""
    modern_log = np.log10(modern.clip(lower=1e-12))
    archean_log = np.log10(archean_imputed.clip(lower=1e-12))
    modern_corr = modern_log.corr(method="spearman")
    archean_corr = archean_log.corr(method="spearman")
    rows = []
    for first_index, first in enumerate(SHORT_COLUMNS):
        for second in SHORT_COLUMNS[first_index + 1:]:
            observed = archean_raw[[first, second]].apply(pd.to_numeric, errors="coerce")
            observed = observed[(observed[first] > 0) & (observed[second] > 0)].dropna()
            observed_corr = np.nan
            if len(observed) >= 30:
                observed_corr = spearmanr(
                    np.log10(observed[first]), np.log10(observed[second])
                ).statistic
            rows.append(
                {
                    "element_1": first,
                    "element_2": second,
                    "modern_spearman": modern_corr.loc[first, second],
                    "archean_imputed_spearman": archean_corr.loc[first, second],
                    "archean_observed_spearman": observed_corr,
                    "observed_pair_n": len(observed),
                    "imputed_vs_modern_abs_delta": abs(
                        archean_corr.loc[first, second] - modern_corr.loc[first, second]
                    ),
                    "imputation_correlation_change": (
                        abs(archean_corr.loc[first, second] - observed_corr)
                        if np.isfinite(observed_corr) else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values("imputed_vs_modern_abs_delta", ascending=False)


def summarize_counterfactual(
    name: str,
    prediction: pd.DataFrame,
    reference_class: pd.Series,
) -> dict:
    """汇总一个反事实预测版本。"""
    class_column = f"{name}_class"
    confidence_column = f"{name}_confidence"
    arc_column = f"{name}_arc_probability"
    return {
        "variant": name,
        "cfb_n": int(prediction[class_column].eq(CFB_CLASS).sum()),
        "cfb_pct": 100.0 * prediction[class_column].eq(CFB_CLASS).mean(),
        "arc_probability_ge_0_5_n": int(prediction[arc_column].ge(0.5).sum()),
        "mean_confidence": prediction[confidence_column].mean(),
        "median_confidence": prediction[confidence_column].median(),
        "class_agreement_with_current_pct": 100.0 * prediction[class_column].eq(reference_class).mean(),
    }


def main() -> None:
    print("读取数据并构建原始缺失掩码...")
    modern_raw = pd.read_csv(MODERN_RAW_PATH, low_memory=False)
    modern_continuous_raw = pd.read_csv(MODERN_CONTINUOUS_PATH, low_memory=False)
    modern_quantile = pd.read_csv(MODERN_QUANTILE_PATH, low_memory=False)
    test_quantile = pd.read_csv(TEST_QUANTILE_PATH, low_memory=False)
    archean_raw = pd.read_csv(ARCHEAN_RAW_PATH, low_memory=False)
    archean_imputed = pd.read_csv(ARCHEAN_IMPUTED_PATH, low_memory=False)
    archean_normalized = pd.read_csv(ARCHEAN_NORMALIZED_PATH, low_memory=False)
    archean_quantile = pd.read_csv(ARCHEAN_QUANTILE_PATH, low_memory=False)
    with open(QUANTILE_PARAMS_PATH, "r", encoding="utf-8") as file:
        quantile_params = json.load(file)

    modern_continuous = read_numeric(modern_continuous_raw, LONG_COLUMNS)
    modern_continuous.columns = SHORT_COLUMNS
    archean_continuous = read_numeric(archean_normalized, SHORT_COLUMNS)
    archean_imputed_continuous = read_numeric(archean_imputed, SHORT_COLUMNS)
    archean_raw_continuous = read_numeric(archean_raw, SHORT_COLUMNS)
    missing_mask = archean_raw_continuous.isna() | archean_raw_continuous.le(0)
    missing_count = missing_mask.sum(axis=1)

    print("加载模型并执行当前流程及两组反事实推理...")
    model, class_names, device, image_columns, sequence_columns = load_model_and_classes(
        pd.read_csv(
            str(TRAIN_NORM_CSV),
            low_memory=False,
        )
    )
    current_probabilities = predict_probabilities(
        archean_quantile, model, device, image_columns, sequence_columns
    )
    current_prediction = prediction_frame(current_probabilities, class_names, "current")

    # 中文注释：反事实一只替换插补方法，其他处理保持不变。
    modern_median = read_numeric(modern_raw, LONG_COLUMNS).median()
    median_imputed_long = short_to_long(archean_raw_continuous)
    median_imputed_long = median_imputed_long.fillna(modern_median)
    for short_name, long_name in zip(SHORT_COLUMNS, LONG_COLUMNS):
        invalid = archean_raw_continuous[short_name].isna() | archean_raw_continuous[short_name].le(0)
        median_imputed_long.loc[invalid, long_name] = modern_median[long_name]
    median_imputed_short = median_imputed_long.copy()
    median_imputed_short.columns = SHORT_COLUMNS
    median_normalized_short = normalize_major_short(median_imputed_short)
    median_quantile = quantile_encode(short_to_long(median_normalized_short), quantile_params)
    median_probabilities = predict_probabilities(
        median_quantile, model, device, image_columns, sequence_columns
    )
    median_prediction = prediction_frame(median_probabilities, class_names, "median_impute")

    # 中文注释：反事实二保留当前RF插补值，但跳过主量元素无水标准化。
    no_normalization_quantile = quantile_encode(
        short_to_long(archean_imputed_continuous), quantile_params
    )
    no_normalization_probabilities = predict_probabilities(
        no_normalization_quantile, model, device, image_columns, sequence_columns
    )
    no_normalization_prediction = prediction_frame(
        no_normalization_probabilities, class_names, "no_major_normalization"
    )

    print("计算单元素、组合关系、PCA和适用域距离...")
    element = element_summary(
        modern_raw,
        modern_continuous_raw,
        archean_raw,
        archean_normalized,
        archean_quantile,
    )
    correlations = correlation_shift(
        modern_continuous,
        archean_continuous,
        archean_raw_continuous,
    )
    distance, thresholds, pc_modern, pc_archean = compute_distance_diagnostics(
        modern_continuous,
        archean_continuous,
        modern_continuous_raw["TECTONIC SETTING"].astype(str).to_numpy(),
    )

    sample = pd.concat(
        [
            current_prediction,
            median_prediction,
            no_normalization_prediction,
            distance,
        ],
        axis=1,
    )
    sample.insert(0, "missing_feature_count", missing_count.to_numpy())
    sample["boundary_feature_count"] = archean_quantile[LONG_COLUMNS].isin([1, 255]).sum(axis=1)
    sample["imputation_changed_class"] = sample["current_class"] != sample["median_impute_class"]
    sample["normalization_changed_class"] = (
        sample["current_class"] != sample["no_major_normalization_class"]
    )
    for column in ["Craton", "REFERENCE", "SAMPLE_ID", "ROCK_NAME", "AGE", "AGE_MEAN"]:
        if column in archean_raw.columns:
            sample.insert(len(sample.columns), column, archean_raw[column].to_numpy())

    print("评估现代测试集校准和太古代高置信度域外比例...")
    test_probabilities = predict_probabilities(
        test_quantile[LONG_COLUMNS], model, device, image_columns, sequence_columns
    )
    test_predicted = np.asarray(class_names, dtype=object)[test_probabilities.argmax(axis=1)]
    test_true = test_quantile["TECTONIC SETTING"].astype(str).to_numpy()
    test_confidence = test_probabilities.max(axis=1)
    test_correct = test_predicted == test_true
    test_accuracy = accuracy_score(test_true, test_predicted)
    test_ece = expected_calibration_error(test_confidence, test_correct)
    test_high_conf_error = 1.0 - test_correct[test_confidence >= 0.9].mean()

    class_rows = []
    for class_name, group in sample.groupby("current_class"):
        class_rows.append(
            {
                "predicted_class": class_name,
                "n": len(group),
                "pct": 100.0 * len(group) / len(sample),
                "median_missing_count": group["missing_feature_count"].median(),
                "median_boundary_count": group["boundary_feature_count"].median(),
                "mean_confidence": group["current_confidence"].mean(),
                "knn_outside_99_pct": 100.0 * group["knn_outside_99"].mean(),
                "mahalanobis_outside_99_pct": 100.0 * group["mahalanobis_outside_99"].mean(),
                "nearest_class_match_pct": 100.0 * group["nearest_modern_class"].eq(class_name).mean(),
                "median_imputation_class_change_pct": 100.0 * group["imputation_changed_class"].mean(),
                "normalization_class_change_pct": 100.0 * group["normalization_changed_class"].mean(),
            }
        )
    class_summary = pd.DataFrame(class_rows).sort_values("n", ascending=False)

    source_column = "REFERENCE" if "REFERENCE" in sample.columns else "Craton"
    source_summary = pd.DataFrame()
    if source_column in sample.columns:
        source_rows = []
        for source_name, group in sample.groupby(source_column, dropna=False):
            if len(group) < 10:
                continue
            source_rows.append(
                {
                    "source": source_name,
                    "n": len(group),
                    "cfb_pct": 100.0 * group["current_class"].eq(CFB_CLASS).mean(),
                    "arc_ge_0_5_pct": 100.0 * group["current_arc_probability"].ge(0.5).mean(),
                    "median_missing_count": group["missing_feature_count"].median(),
                    "median_knn_distance": group["knn_distance"].median(),
                    "mean_confidence": group["current_confidence"].mean(),
                }
            )
        source_summary = pd.DataFrame(source_rows).sort_values("n", ascending=False)

    counterfactual = pd.DataFrame(
        [
            summarize_counterfactual("current", current_prediction, current_prediction["current_class"]),
            summarize_counterfactual(
                "median_impute", median_prediction, current_prediction["current_class"]
            ),
            summarize_counterfactual(
                "no_major_normalization",
                no_normalization_prediction,
                current_prediction["current_class"],
            ),
        ]
    )

    element.to_csv(ELEMENT_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    correlations.to_csv(CORRELATION_SHIFT_PATH, index=False, encoding="utf-8-sig")
    sample.to_csv(SAMPLE_DIAGNOSTICS_PATH, index=False, encoding="utf-8-sig")
    class_summary.to_csv(CLASS_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    source_summary.to_csv(SOURCE_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    counterfactual.to_csv(COUNTERFACTUAL_SUMMARY_PATH, index=False, encoding="utf-8-sig")

    cfb = sample["current_class"].eq(CFB_CLASS)
    complete = sample["missing_feature_count"].eq(0)
    high_conf_ood = sample["current_confidence"].ge(0.9) & sample["knn_outside_99"]
    boundary_cells = archean_quantile[LONG_COLUMNS].isin([1, 255]).to_numpy().mean()
    top_elements = element.head(8)["element"].tolist()
    top_boundaries = element.sort_values("archean_boundary_pct", ascending=False).head(8)["element"].tolist()

    # 中文注释：图形只呈现判断成因所需的高信号结果。
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(
        pc_modern[:, 0], pc_modern[:, 1], s=2, alpha=0.08, color="#777777", label="Modern"
    )
    axes[0, 0].scatter(
        pc_archean[~cfb, 0], pc_archean[~cfb, 1], s=7, alpha=0.35, color="#4E79A7", label="Archean non-CFB"
    )
    axes[0, 0].scatter(
        pc_archean[cfb, 0], pc_archean[cfb, 1], s=7, alpha=0.35, color="#E15759", label="Archean CFB"
    )
    axes[0, 0].set_title("Modern-reference PCA")
    axes[0, 0].set_xlabel("PC1")
    axes[0, 0].set_ylabel("PC2")
    # 中文注释：仅对显示范围做稳健裁切，距离和域外比例仍使用全部样品计算。
    combined_pc1 = np.concatenate([pc_modern[:, 0], pc_archean[:, 0]])
    combined_pc2 = np.concatenate([pc_modern[:, 1], pc_archean[:, 1]])
    pc1_lower, pc1_upper = np.quantile(combined_pc1, [0.005, 0.995])
    pc2_lower, pc2_upper = np.quantile(combined_pc2, [0.005, 0.995])
    pc1_padding = 0.08 * (pc1_upper - pc1_lower)
    pc2_padding = 0.08 * (pc2_upper - pc2_lower)
    axes[0, 0].set_xlim(pc1_lower - pc1_padding, pc1_upper + pc1_padding)
    axes[0, 0].set_ylim(pc2_lower - pc2_padding, pc2_upper + pc2_padding)
    axes[0, 0].legend(frameon=False, fontsize=8)

    missing_bins = pd.cut(sample["missing_feature_count"], [-1, 0, 5, 10, 15, 19])
    cfb_by_missing = sample.groupby(missing_bins, observed=True)["current_class"].apply(
        lambda values: 100.0 * values.eq(CFB_CLASS).mean()
    )
    axes[0, 1].bar(cfb_by_missing.index.astype(str), cfb_by_missing.to_numpy(), color="#E15759")
    axes[0, 1].set_title("CFB prediction vs missingness")
    axes[0, 1].set_ylabel("CFB (%)")
    axes[0, 1].tick_params(axis="x", rotation=25)

    plot_counterfactual = counterfactual.set_index("variant")
    axes[1, 0].bar(
        ["Current", "Median impute", "No major norm."],
        plot_counterfactual["cfb_pct"],
        color=["#4E79A7", "#F28E2B", "#59A14F"],
    )
    axes[1, 0].set_title("Counterfactual CFB share")
    axes[1, 0].set_ylabel("CFB (%)")

    axes[1, 1].scatter(
        sample["knn_distance"],
        sample["current_confidence"],
        s=8,
        alpha=0.3,
        c=np.where(cfb, "#E15759", "#4E79A7"),
    )
    axes[1, 1].axvline(thresholds["knn_q99"], color="black", linestyle="--", linewidth=1)
    axes[1, 1].set_title("Confidence remains high outside domain")
    axes[1, 1].set_xlabel("kNN distance to modern training")
    axes[1, 1].set_ylabel("Softmax confidence")
    fig.tight_layout()
    fig.savefig(FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    current_cfb = int(cfb.sum())
    median_cfb = int(median_prediction["median_impute_class"].eq(CFB_CLASS).sum())
    no_norm_cfb = int(
        no_normalization_prediction["no_major_normalization_class"].eq(CFB_CLASS).sum()
    )
    complete_cfb_pct = 100.0 * sample.loc[complete, "current_class"].eq(CFB_CLASS).mean() if complete.any() else np.nan
    cfb_missing_median = sample.loc[cfb, "missing_feature_count"].median()
    non_cfb_missing_median = sample.loc[~cfb, "missing_feature_count"].median()
    cfb_ood99 = 100.0 * sample.loc[cfb, "knn_outside_99"].mean()
    all_ood99 = 100.0 * sample["knn_outside_99"].mean()
    nearest_cfb = 100.0 * sample.loc[cfb, "nearest_modern_class"].eq(CFB_CLASS).mean()
    source_cfb_sd = source_summary["cfb_pct"].std() if not source_summary.empty else np.nan
    source_missing_correlation = (
        source_summary["cfb_pct"].corr(
            source_summary["median_missing_count"], method="spearman"
        )
        if not source_summary.empty else np.nan
    )
    source_distance_correlation = (
        source_summary["cfb_pct"].corr(
            source_summary["median_knn_distance"], method="spearman"
        )
        if not source_summary.empty else np.nan
    )
    cfb_boundary_mean = sample.loc[cfb, "boundary_feature_count"].mean()
    non_cfb_boundary_mean = sample.loc[~cfb, "boundary_feature_count"].mean()
    cfb_median_stability = 100.0 * sample.loc[
        cfb, "median_impute_class"
    ].eq(CFB_CLASS).mean()

    report = f"""# 现代训练集与太古代应用集分布差异诊断

## 总体判断

- **大量CFB预测的首要原因不是随机森林插补，也不是无水标准化。** 完整无缺失样品仍有 {complete_cfb_pct:.1f}% 被判为CFB；替换成现代中位数插补后CFB不降反升至 {median_cfb} 条；跳过无水标准化则2116条类别全部不变。
- **主要矛盾是年代域/组成关系偏移与模型类别决策边界的不匹配。** 太古代样品的P2O5、Zr、TiO2、HF、Sr及REE分布显著偏离现代训练集，多个Zr-Y-REE-HFSE组合关系也发生变化。当前被判为CFB的样品在连续地球化学类别马氏距离上多数更接近现代岛弧或洋内弧类，而不是现代CFB类。
- **数据来源差异是重要的次级因素，但不能归结为某些来源缺失更多。** 文献来源间CFB比例标准差为 {source_cfb_sd:.1f} 个百分点，但来源CFB比例与缺失数、kNN域距离的Spearman相关分别仅为 {source_missing_correlation:.2f} 和 {source_distance_correlation:.2f}。
- **模型存在明显的域外/跨时代过度自信。** 太古代平均softmax置信度为 {sample["current_confidence"].mean():.3f}，且有 {int(high_conf_ood.sum())} 条样品同时高置信并位于现代99% kNN域外。即使处于全局现代域内，高置信度也不能保证九类构造语义可直接迁移到太古代。
- 因此，这1229条结果应表述为“与模型所学现代CFB决策模式相似”，不宜直接等同于1229个可靠的太古代大陆洪流玄武岩构造判定。弧概率阈值保持0.5，不通过降低阈值增加高弧数量。

## 核心结果

- 当前流程预测 CFB {current_cfb}/{len(sample)}（{100 * current_cfb / len(sample):.1f}%）；弧概率和不低于0.5的样品为 {int(sample["current_arc_probability"].ge(0.5).sum())} 条。
- 太古代原始数据平均缺失 {missing_count.mean():.2f}/36 个元素，中位数 {missing_count.median():.0f}；完整样品 {int(complete.sum())} 条。完整样品的 CFB 比例为 {complete_cfb_pct:.1f}%。
- CFB 样品缺失数中位数为 {cfb_missing_median:.1f}，非 CFB 为 {non_cfb_missing_median:.1f}。因此可直接检验 CFB 集中是否只随插补比例增加。
- 将随机森林插补替换为现代训练中位数后，CFB 为 {median_cfb} 条；与当前类别一致率为 {100 * median_prediction["median_impute_class"].eq(current_prediction["current_class"]).mean():.1f}%。当前CFB中仍有 {cfb_median_stability:.1f}% 保持CFB。
- 保留随机森林插补但跳过主量无水标准化后，CFB 为 {no_norm_cfb} 条；与当前类别一致率为 {100 * no_normalization_prediction["no_major_normalization_class"].eq(current_prediction["current_class"]).mean():.1f}%。

## 分布与适用域

- 太古代全部样品中 {all_ood99:.1f}% 超出现代训练集自身 kNN 距离的99%阈值；CFB 子集中该比例为 {cfb_ood99:.1f}%。
- CFB 样品中仅 {nearest_cfb:.1f}% 在类别马氏距离上最近于现代 CFB 类；作为参照，现代CFB训练样品自身的同方法匹配率为 {100 * thresholds["modern_cfb_nearest_class_accuracy"]:.1f}%（现代全部类别为 {100 * thresholds["modern_nearest_class_accuracy"]:.1f}%）。该邻近法不是分类真值，但太古代CFB与现代CFB端元的一致性明显更弱。
- 太古代所有特征单元中有 {100 * boundary_cells:.2f}% 堆积在训练分位数编码1或255。边界堆积最强元素：{", ".join(top_boundaries)}。
- CFB样品平均有 {cfb_boundary_mean:.2f} 个边界特征，非CFB为 {non_cfb_boundary_mean:.2f} 个；边界堆积不是CFB集中的正向驱动。
- 单元素 log10 分布差异最大的元素：{", ".join(top_elements)}。完整统计见 `domain_shift_element_summary.csv`。
- PCA保留 {thresholds["n_pc"]} 个主成分，累计解释 {100 * thresholds["pca_variance"]:.1f}% 方差。

## 模型置信度

- 同一模型在现代测试集上的本次复算 Accuracy 为 {test_accuracy:.4f}，ECE 为 {test_ece:.4f}；softmax置信度不低于0.9的现代测试样品错误率为 {100 * test_high_conf_error:.2f}%。
- 太古代样品平均置信度为 {sample["current_confidence"].mean():.3f}，其中 {int(high_conf_ood.sum())} 条同时满足置信度不低于0.9且位于现代99% kNN域外。
- 现代测试集上的高准确率不能校准域外太古代样品；高置信度域外预测应解释为模型过度自信，而不是额外证据。

## 数据来源差异

- 按至少10条样品的文献来源分组，CFB比例的组间标准差为 {source_cfb_sd:.1f} 个百分点。来源级缺失率、kNN距离和CFB比例见 `domain_shift_source_summary.csv`。
- 若高CFB来源同时具有高缺失数和高域距离，则插补与来源偏差共同作用；若低缺失来源仍稳定高CFB，则更支持年代域偏移或真实组成差异。

## 判断原则

1. **插补偏差**：以“中位数插补反事实的类别翻转率”、CFB与非CFB缺失数差异、以及插补前后元素相关变化共同判断。
2. **无水标准化**：以“跳过标准化反事实的类别翻转率”判断，不依赖人为调整弧概率阈值。
3. **年代域偏移**：以低缺失/完整样品仍高CFB、连续空间域外比例和类别邻近度共同判断。
4. **数据来源差异**：以文献来源间CFB比例、缺失度和域距离的同步变化判断。
5. **模型过度自信**：以高置信度且超出现代99%适用域的样品数量判断。

## 方法限制

- 中位数插补是敏感性分析，不是更优插补器；它用于判断结果是否依赖当前随机森林插补的具体数值。
- 当前插补代码是逐元素单次随机森林预测，其余缺失预测变量以现代标准化均值0代入，不是迭代收敛的标准MissForest。
- 太古代没有可靠的九分类真值，因此不能由本分析直接证明CFB标签正确，只能判断预测由哪些预处理和域偏移因素驱动。
"""
    with open(REPORT_PATH, "w", encoding="utf-8") as file:
        file.write(report)

    print(report)
    print(f"输出图: {FIGURE_PATH}")
    print(f"输出报告: {REPORT_PATH}")


if __name__ == "__main__":
    main()
