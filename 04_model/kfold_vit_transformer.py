"""
5 折交叉验证训练入口。

流程：
1. 读取已经固定划分出的 80% 原始训练集。
2. 在该训练集内部执行 StratifiedKFold。
3. 每一折只使用折内训练数据拟合 IQR、分类别 MissForest 和分位数边界。
4. 折内验证数据只做 transform，不参与任何预处理参数拟合。
5. 每折使用两个随机种子训练主模型，最终汇总全部折和种子的指标。

说明：
- 固定的 20% 最终测试集不参与本脚本。
- 本脚本沿用项目当前“按真实构造类别选择插补器”的规则。
- 主量无水归一化是逐行计算，不需要拟合全局参数。
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# =============================================================
# 固定配置
# =============================================================

# 路径直接写完整地址，避免运行目录变化影响文件定位。
TRAIN_RAW_FILE = Path(r"E:\program\python\basalt_tectonic_discrimination\data\04_split\01_basalt_number_year_train.csv")
IMPUTATION_MODULE_FILE = Path(r"E:\program\python\basalt_tectonic_discrimination\02_imputation\imputation_train_predict.py")
MAJOR_NORMALIZATION_MODULE_FILE = Path(r"E:\program\python\basalt_tectonic_discrimination\03_normalization\normalize_major_elements.py")
QUANTILE_NORMALIZATION_MODULE_FILE = Path(r"E:\program\python\basalt_tectonic_discrimination\03_normalization\normalize.py")
MODEL_MODULE_FILE = Path(r"E:\program\python\basalt_tectonic_discrimination\04_model\ablation_v4_vit_transformer.py")

# 每一折的权重输出目录均使用完整路径。
FOLD_OUTPUT_DIRS = [
    Path(r"E:\program\python\basalt_tectonic_discrimination\data\models\kfold\fold_1"),
    Path(r"E:\program\python\basalt_tectonic_discrimination\data\models\kfold\fold_2"),
    Path(r"E:\program\python\basalt_tectonic_discrimination\data\models\kfold\fold_3"),
    Path(r"E:\program\python\basalt_tectonic_discrimination\data\models\kfold\fold_4"),
    Path(r"E:\program\python\basalt_tectonic_discrimination\data\models\kfold\fold_5"),
]
PER_RUN_RESULT_FILE = Path(r"E:\program\python\basalt_tectonic_discrimination\data\models\kfold\kfold_per_run_results.csv")
SUMMARY_RESULT_FILE = Path(r"E:\program\python\basalt_tectonic_discrimination\data\models\kfold\kfold_summary.csv")

LABEL_COLUMN = "TECTONIC SETTING"
N_SPLITS = 5
CV_SPLIT_SEED = 32
TRAIN_SEEDS = [42, 123]
EPOCHS = 200
MIXUP_ALPHA = 0
IQR_THRESHOLD = 6

# 默认只跑主模型。需要做完整消融时，可在这里增加 Abl-1、Abl-2 等编号。
EXPERIMENTS_TO_RUN = [
    "Full",   # ViT + Transformer 双流
    "Abl-1",  # 仅 ViT 矩阵分支
    "Abl-2",  # 仅 Transformer 序列分支
    "Abl-3",  # 双流但没有位置编码
    "Cmp-1",  # CNN-BiLSTM
    "Cmp-2",  # CNN-ViT-Transformer
    "Cmp-3",  # 仅 CNN
]
# EXPERIMENTS_TO_RUN = ["Full"]
COLUMN_ORDER_SCHEME = "v1"


def load_module(module_name, module_file):
    """从固定文件地址加载现有模块。"""
    spec = importlib.util.spec_from_file_location(module_name, module_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块: {module_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_input_data(data):
    """检查 5 折分层划分所需的标签和样本数。"""
    if LABEL_COLUMN not in data.columns:
        raise KeyError(f"训练数据缺少标签列: {LABEL_COLUMN}")

    missing_labels = int(data[LABEL_COLUMN].isna().sum())
    if missing_labels > 0:
        raise ValueError(f"训练数据存在 {missing_labels} 个空标签，无法进行分层 5 折")

    class_counts = data[LABEL_COLUMN].value_counts()
    too_small = class_counts[class_counts < N_SPLITS]
    if not too_small.empty:
        raise ValueError(
            "以下类别的样本数少于 5，无法执行 StratifiedKFold:\n"
            f"{too_small.to_string()}"
        )


def clean_training_fold_by_class(data, chemical_columns):
    """
    仅对折内训练数据按类别执行 IQR 清洗。

    每个类别、每个化学特征都只使用当前训练折计算上下界；
    验证折不会参与边界计算，也不会删除异常值。
    """
    cleaned_parts = []
    cleaning_rows = []

    for setting, group in data.groupby(LABEL_COLUMN, sort=False):
        group = group.copy()
        row_outliers = pd.Series(False, index=group.index)

        for column in chemical_columns:
            values = pd.to_numeric(group[column], errors="coerce")
            valid_values = values.dropna()
            if valid_values.empty:
                continue

            q1 = valid_values.quantile(0.25)
            q3 = valid_values.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - IQR_THRESHOLD * iqr
            upper_bound = q3 + IQR_THRESHOLD * iqr
            row_outliers |= (values < lower_bound) | (values > upper_bound)

        cleaned_group = group.loc[~row_outliers].copy()
        cleaned_parts.append(cleaned_group)
        cleaning_rows.append(
            {
                "tectonic_setting": setting,
                "before_count": len(group),
                "after_count": len(cleaned_group),
                "removed_count": int(row_outliers.sum()),
            }
        )

    cleaned_data = pd.concat(cleaned_parts, ignore_index=True)
    cleaning_stats = pd.DataFrame(cleaning_rows)
    return cleaned_data, cleaning_stats


def fit_and_impute_training_fold(data, imputation_module):
    """
    按类别拟合 StandardScaler 和 MissForest，并插补折内训练数据。

    返回插补后的训练数据以及每个类别对应的插补模型。
    """
    imputed_parts = []
    trained_models = {}

    for setting, group in data.groupby(LABEL_COLUMN, sort=False):
        chemical_data = imputation_module.preprocess_chemical_data(
            group,
            imputation_module.CHEMICAL_COLUMNS,
        )
        if chemical_data.empty:
            raise ValueError(f"类别 {setting} 清洗后没有可用于插补的训练样本")

        scaler = StandardScaler()
        scaled_data = pd.DataFrame(
            scaler.fit_transform(chemical_data),
            columns=chemical_data.columns,
            index=chemical_data.index,
        )
        imputers, column_order = imputation_module.missforest_fit(scaled_data)
        scaled_imputed = imputation_module.missforest_transform(
            scaled_data,
            imputers,
            column_order,
        )
        original_scale = pd.DataFrame(
            scaler.inverse_transform(scaled_imputed),
            columns=chemical_data.columns,
            index=chemical_data.index,
        )

        preserved = group.loc[chemical_data.index, [LABEL_COLUMN]].copy()
        imputed_parts.append(
            pd.concat(
                [
                    preserved.reset_index(drop=True),
                    original_scale.reset_index(drop=True),
                ],
                axis=1,
            )
        )
        trained_models[setting] = {
            "scaler": scaler,
            "imputers": imputers,
            "column_order": column_order,
        }

    return pd.concat(imputed_parts, ignore_index=True), trained_models


def impute_validation_fold(data, trained_models, imputation_module):
    """使用当前折训练好的分类别插补器转换验证折。"""
    imputed_parts = []

    for setting, group in data.groupby(LABEL_COLUMN, sort=False):
        if setting not in trained_models:
            raise ValueError(f"验证折类别 {setting} 在当前训练折中没有对应插补器")

        chemical_data = imputation_module.preprocess_chemical_data(
            group,
            imputation_module.CHEMICAL_COLUMNS,
        )
        bundle = trained_models[setting]
        scaled_data = pd.DataFrame(
            bundle["scaler"].transform(chemical_data),
            columns=chemical_data.columns,
            index=chemical_data.index,
        )
        scaled_imputed = imputation_module.missforest_transform(
            scaled_data,
            bundle["imputers"],
            bundle["column_order"],
        )
        original_scale = pd.DataFrame(
            bundle["scaler"].inverse_transform(scaled_imputed),
            columns=chemical_data.columns,
            index=chemical_data.index,
        )

        preserved = group.loc[chemical_data.index, [LABEL_COLUMN]].copy()
        imputed_parts.append(
            pd.concat(
                [
                    preserved.reset_index(drop=True),
                    original_scale.reset_index(drop=True),
                ],
                axis=1,
            )
        )

    return pd.concat(imputed_parts, ignore_index=True)


def normalize_fold(
    train_data,
    validation_data,
    major_normalization_module,
    quantile_normalization_module,
):
    """
    执行主量无水归一化，并仅使用折内训练数据拟合分位数边界。
    """
    train_major = major_normalization_module.normalize_major_elements(train_data)
    validation_major = major_normalization_module.normalize_major_elements(
        validation_data
    )

    feature_columns = quantile_normalization_module.COLUMNS_TO_EXTRACT
    quantile_params = quantile_normalization_module.fit_quantile_boundaries(
        train_major,
        feature_columns,
    )
    train_normalized = quantile_normalization_module.apply_quantile_transform(
        train_major,
        feature_columns,
        quantile_params,
    )
    validation_normalized = quantile_normalization_module.apply_quantile_transform(
        validation_major,
        feature_columns,
        quantile_params,
    )
    return train_normalized, validation_normalized


def prepare_model_arrays(
    train_data,
    validation_data,
    image_columns,
    sequence_columns,
    label_order,
):
    """把归一化后的表格转换为现有双流模型需要的数组。"""
    label_to_index = {label: index for index, label in enumerate(label_order)}

    train_image_2d = train_data[image_columns].to_numpy(dtype=np.float32) / 255.0
    validation_image_2d = (
        validation_data[image_columns].to_numpy(dtype=np.float32) / 255.0
    )
    train_sequence_2d = (
        train_data[sequence_columns].to_numpy(dtype=np.float32) / 255.0
    )
    validation_sequence_2d = (
        validation_data[sequence_columns].to_numpy(dtype=np.float32) / 255.0
    )

    train_labels = train_data[LABEL_COLUMN].map(label_to_index)
    validation_labels = validation_data[LABEL_COLUMN].map(label_to_index)
    if train_labels.isna().any() or validation_labels.isna().any():
        raise ValueError("折内数据出现未注册的构造环境标签")

    return (
        train_image_2d.reshape(-1, 1, 6, 6),
        train_sequence_2d[:, :, np.newaxis],
        train_labels.to_numpy(dtype=np.int64),
        validation_image_2d.reshape(-1, 1, 6, 6),
        validation_sequence_2d[:, :, np.newaxis],
        validation_labels.to_numpy(dtype=np.int64),
    )


def build_experiments(model_module, num_classes):
    """按配置创建需要运行的模型。"""
    all_experiments = {
        "Full": (
            "Full Model\n(ViT+Transformer)",
            lambda: model_module.ViT_Transformer_DualStream(
                num_classes=num_classes
            ),
        ),
        "Abl-1": (
            "Abl-1\nViT Only (Matrix)",
            lambda: model_module.Ablation_ViT_Only(num_classes=num_classes),
        ),
        "Abl-2": (
            "Abl-2\nTransformer Only (Seq)",
            lambda: model_module.Ablation_Transformer_Only(
                num_classes=num_classes
            ),
        ),
        "Abl-3": (
            "Abl-3\nw/o Pos Encoding",
            lambda: model_module.Ablation_NoPositionalEncoding(
                num_classes=num_classes
            ),
        ),
        "Cmp-1": (
            "Cmp-1\nCNN-BiLSTM (EMSPN)",
            lambda: model_module.CNN_BiLSTM(num_classes=num_classes),
        ),
        "Cmp-2": (
            "Cmp-2\nCNN-ViT-Transformer",
            lambda: model_module.CNN_ViT_Transformer(num_classes=num_classes),
        ),
        "Cmp-3": (
            "Cmp-3\nCNN Only",
            lambda: model_module.Baseline_CNN_Only(num_classes=num_classes),
        ),
    }

    invalid_names = [
        name for name in EXPERIMENTS_TO_RUN if name not in all_experiments
    ]
    if invalid_names:
        raise ValueError(f"未注册的实验编号: {invalid_names}")
    return [all_experiments[name] for name in EXPERIMENTS_TO_RUN]


def collect_run_rows(fold_number, result):
    """展开当前折的每个随机种子结果。"""
    rows = []
    for seed_result in result["per_seed_results"]:
        rows.append(
            {
                "fold": fold_number,
                "seed": seed_result["seed"],
                "experiment": result["model_name"].replace("\n", " "),
                "accuracy": seed_result["accuracy"],
                "precision": seed_result["precision"],
                "recall": seed_result["recall"],
                "weighted_f1": seed_result["f1_score"],
                "macro_f1": seed_result["macro_f1"],
                "mAP": seed_result["mAP"],
                "validation_loss": seed_result["val_loss"],
                "best_epoch": seed_result["best_acc_epoch"],
            }
        )
    return rows


def summarize_results(per_run_results):
    """按实验汇总 5 折乘 2 个种子的均值和样本标准差。"""
    metric_columns = [
        "accuracy",
        "precision",
        "recall",
        "weighted_f1",
        "macro_f1",
        "mAP",
        "validation_loss",
        "best_epoch",
    ]
    summary = (
        per_run_results.groupby("experiment")[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        column
        if isinstance(column, str)
        else "_".join(part for part in column if part)
        for column in summary.columns
    ]
    return summary


def main():
    """运行完整的 5 折预处理与模型训练。"""
    imputation_module = load_module(
        "kfold_imputation",
        IMPUTATION_MODULE_FILE,
    )
    major_normalization_module = load_module(
        "kfold_major_normalization",
        MAJOR_NORMALIZATION_MODULE_FILE,
    )
    quantile_normalization_module = load_module(
        "kfold_quantile_normalization",
        QUANTILE_NORMALIZATION_MODULE_FILE,
    )
    model_module = load_module(
        "kfold_model",
        MODEL_MODULE_FILE,
    )

    raw_train_data = pd.read_csv(TRAIN_RAW_FILE, low_memory=False)
    validate_input_data(raw_train_data)
    label_order = sorted(raw_train_data[LABEL_COLUMN].unique().tolist())
    num_classes = len(label_order)

    if COLUMN_ORDER_SCHEME == "v2":
        image_columns = model_module.IMAGE_COLUMNS_V2
        sequence_columns = model_module.SEQUENCE_COLUMNS_V2
    else:
        image_columns = model_module.ORIGINAL_IMAGE_COLUMNS
        sequence_columns = model_module.COLUMNS_ELECTRODE_ORDER_V1

    experiments = build_experiments(model_module, num_classes)
    splitter = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=CV_SPLIT_SEED,
    )

    all_run_rows = []
    split_iterator = splitter.split(
        raw_train_data,
        raw_train_data[LABEL_COLUMN],
    )

    for fold_number, (train_indices, validation_indices) in enumerate(
        split_iterator,
        start=1,
    ):
        print("\n" + "=" * 75)
        print(f"开始第 {fold_number}/{N_SPLITS} 折")
        print("=" * 75)

        fold_train_raw = raw_train_data.iloc[train_indices].copy()
        fold_validation_raw = raw_train_data.iloc[validation_indices].copy()
        print(
            f"原始折内训练样本: {len(fold_train_raw)}，"
            f"验证样本: {len(fold_validation_raw)}"
        )

        fold_train_clean, cleaning_stats = clean_training_fold_by_class(
            fold_train_raw,
            imputation_module.CHEMICAL_COLUMNS,
        )
        print("折内训练数据 IQR 清洗统计:")
        print(cleaning_stats.to_string(index=False))

        fold_train_imputed, trained_imputers = fit_and_impute_training_fold(
            fold_train_clean,
            imputation_module,
        )
        fold_validation_imputed = impute_validation_fold(
            fold_validation_raw,
            trained_imputers,
            imputation_module,
        )
        fold_train_normalized, fold_validation_normalized = normalize_fold(
            fold_train_imputed,
            fold_validation_imputed,
            major_normalization_module,
            quantile_normalization_module,
        )

        (
            train_image,
            train_sequence,
            train_labels,
            validation_image,
            validation_sequence,
            validation_labels,
        ) = prepare_model_arrays(
            fold_train_normalized,
            fold_validation_normalized,
            image_columns,
            sequence_columns,
            label_order,
        )
        print(
            f"模型输入训练样本: {len(train_labels)}，"
            f"验证样本: {len(validation_labels)}"
        )

        fold_output_dir = FOLD_OUTPUT_DIRS[fold_number - 1]
        fold_output_dir.mkdir(parents=True, exist_ok=True)
        for experiment_name, model_factory in experiments:
            result = model_module.run_experiment_multi_seed(
                model_factory,
                experiment_name,
                train_image,
                train_sequence,
                train_labels,
                validation_image,
                validation_sequence,
                validation_labels,
                num_classes,
                model_module.device,
                epochs=EPOCHS,
                mixup_alpha=MIXUP_ALPHA,
                seeds=TRAIN_SEEDS,
                output_dir=str(fold_output_dir),
                save_per_seed_weights=True,
            )
            all_run_rows.extend(collect_run_rows(fold_number, result))

        current_results = pd.DataFrame(all_run_rows)
        PER_RUN_RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)
        current_results.to_csv(
            PER_RUN_RESULT_FILE,
            index=False,
            encoding="utf-8-sig",
        )

    per_run_results = pd.DataFrame(all_run_rows)
    summary = summarize_results(per_run_results)
    summary.to_csv(
        SUMMARY_RESULT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 75)
    print("5 折交叉验证完成")
    print(f"逐次结果: {PER_RUN_RESULT_FILE}")
    print(f"汇总结果: {SUMMARY_RESULT_FILE}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
