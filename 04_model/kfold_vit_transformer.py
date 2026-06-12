"""
5 折交叉验证训练入口（与主流程方法学一致）。

流程：
1. 读取已经固定划分出的 80% 原始训练集。
2. 在该训练集内部执行 StratifiedKFold。
3. 每一折只使用折内训练数据拟合全局 StandardScaler、全局 MissForest、
   SMOTE 和分位数边界；同时在插补前记录原始缺失 mask。
4. 折内验证数据只做 transform，不参与任何预处理参数拟合，也不做 SMOTE。
5. 每折使用两个随机种子训练模型（显式缺失编码），最终汇总全部折和种子的指标。

说明：
- 固定的 20% 最终测试集不参与本脚本。
- 与主流程一致：全局插补（不按类别）、仅训练折选择性 SMOTE、
  分位数边界从 SMOTE 前的折内训练数据拟合、模型接收数值+缺失 mask 双通道。
- SMOTE 合成样本在已插补的完整特征空间生成，其缺失 mask 统一为 0。
"""

import importlib.util
import sys
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# =============================================================
# 固定配置
# =============================================================

# 中文注释：以当前脚本所在位置定位项目根目录，换电脑或盘符后无需修改路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_RAW_FILE = PROJECT_ROOT / Path("data/04_split/01_basalt_number_year_train.csv")
IMPUTATION_MODULE_FILE = PROJECT_ROOT / Path("02_imputation/imputation_train_predict.py")
MAJOR_NORMALIZATION_MODULE_FILE = PROJECT_ROOT / Path("03_normalization/normalize_major_elements.py")
QUANTILE_NORMALIZATION_MODULE_FILE = PROJECT_ROOT / Path("03_normalization/normalize.py")
SMOTE_MODULE_FILE = PROJECT_ROOT / Path("03_normalization/selective_smote.py")
MODEL_MODULE_FILE = PROJECT_ROOT / Path("04_model/ablation_v4_vit_transformer.py")

# 每一折的权重输出目录均相对于项目根目录。
FOLD_OUTPUT_DIRS = [
    PROJECT_ROOT / Path("data/models/kfold/fold_1"),
    PROJECT_ROOT / Path("data/models/kfold/fold_2"),
    PROJECT_ROOT / Path("data/models/kfold/fold_3"),
    PROJECT_ROOT / Path("data/models/kfold/fold_4"),
    PROJECT_ROOT / Path("data/models/kfold/fold_5"),
]
PER_RUN_RESULT_FILE = PROJECT_ROOT / Path("data/models/kfold/kfold_per_run_results.csv")
SUMMARY_RESULT_FILE = PROJECT_ROOT / Path("data/models/kfold/kfold_summary.csv")
LOG_OUTPUT_DIR = PROJECT_ROOT / Path("data/models/kfold/logs")

LABEL_COLUMN = "TECTONIC SETTING"
N_SPLITS = 5
CV_SPLIT_SEED = 32
TRAIN_SEEDS = [42, 123]
EPOCHS = 200
MIXUP_ALPHA = 0
PRINT_EVERY = 1

# 中文注释：与主流程一致，模型显式接收数值+缺失mask双通道，
# 损失为普通交叉熵，类别不平衡完全交给 SMOTE。
USE_MISSING_MASK = True
SMOTE_RANDOM_STATE = 42
SMOTE_K_NEIGHBORS = 5

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


class TeeStream:
    """将终端输出同步写入日志文件，并在每次写入后立即刷新。"""

    def __init__(self, terminal_stream, log_stream):
        self.terminal_stream = terminal_stream
        self.log_stream = log_stream

    def write(self, message):
        self.terminal_stream.write(message)
        self.log_stream.write(message)
        self.flush()
        return len(message)

    def flush(self):
        self.terminal_stream.flush()
        self.log_stream.flush()

    def isatty(self):
        """保持 tqdm 等组件对终端能力的判断。"""
        return self.terminal_stream.isatty()

    @property
    def encoding(self):
        return getattr(self.terminal_stream, "encoding", "utf-8")


def create_run_log():
    """每次启动创建一个带时间戳的新日志文件。"""
    LOG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_OUTPUT_DIR / f"kfold_training_{timestamp}.log"
    log_stream = open(log_file, "a", encoding="utf-8", buffering=1)
    return log_file, log_stream


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


def fit_and_impute_training_fold(data, imputation_module):
    """
    用折内训练数据拟合全局 StandardScaler 和全局 MissForest。

    与主流程一致：不按类别分组、TECTONIC SETTING 不参与拟合。
    返回插补后的训练表（标签 + 36 元素）、原始缺失 mask 和全局插补模型。
    """
    chemical_columns = imputation_module.CHEMICAL_COLUMNS
    chemical_data = imputation_module.preprocess_chemical_data(
        data,
        chemical_columns,
    )
    if chemical_data.empty:
        raise ValueError("折内训练数据没有可用于插补的样本")

    # 中文注释：mask 记录插补前的原始缺失状态（1=缺失，0=实测）。
    missing_mask = chemical_data.isna().astype(np.uint8)

    scaler = StandardScaler()
    scaled_data = pd.DataFrame(
        scaler.fit_transform(chemical_data),
        columns=chemical_data.columns,
        index=chemical_data.index,
    )
    print(
        f"  [全局MissForest] 训练 {len(chemical_columns)} 个随机森林"
        f"（{len(chemical_data)} 个折内训练样本）",
        flush=True,
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
    ).clip(lower=0.0)

    preserved = data.loc[chemical_data.index, [LABEL_COLUMN]].copy()
    imputed = pd.concat(
        [
            preserved.reset_index(drop=True),
            original_scale.reset_index(drop=True),
        ],
        axis=1,
    )
    global_model = {
        "scaler": scaler,
        "imputers": imputers,
        "column_order": column_order,
    }
    return imputed, missing_mask.reset_index(drop=True), global_model


def impute_validation_fold(data, global_model, imputation_module):
    """使用当前折训练好的全局插补器转换验证折（只 transform）。"""
    chemical_columns = imputation_module.CHEMICAL_COLUMNS
    chemical_data = imputation_module.preprocess_chemical_data(
        data,
        chemical_columns,
    )
    missing_mask = chemical_data.isna().astype(np.uint8)

    print(
        f"  [全局MissForest] 插补验证折（{len(chemical_data)} 个样本）",
        flush=True,
    )
    scaled_data = pd.DataFrame(
        global_model["scaler"].transform(chemical_data),
        columns=chemical_data.columns,
        index=chemical_data.index,
    )
    scaled_imputed = imputation_module.missforest_transform(
        scaled_data,
        global_model["imputers"],
        global_model["column_order"],
    )
    original_scale = pd.DataFrame(
        global_model["scaler"].inverse_transform(scaled_imputed),
        columns=chemical_data.columns,
        index=chemical_data.index,
    ).clip(lower=0.0)

    preserved = data.loc[chemical_data.index, [LABEL_COLUMN]].copy()
    imputed = pd.concat(
        [
            preserved.reset_index(drop=True),
            original_scale.reset_index(drop=True),
        ],
        axis=1,
    )
    return imputed, missing_mask.reset_index(drop=True)


def smote_training_fold(train_major, smote_module, full_train_count):
    """
    仅对折内训练数据执行选择性 SMOTE。

    目标数量按折内训练样本占完整训练集的比例缩放，
    使各折的类别平衡程度与主流程（全训练集补到 3000）一致。
    返回 SMOTE 后的训练表；新增合成行追加在原始行之后。
    """
    feature_columns = smote_module.FEATURE_COLUMNS
    fold_fraction = len(train_major) / float(full_train_count)
    class_targets = {
        label: int(round(target * fold_fraction))
        for label, target in smote_module.CLASS_TARGETS.items()
    }

    labels = train_major[LABEL_COLUMN].astype(str)
    counts = Counter(labels)
    sampling_strategy = {
        label: target
        for label, target in class_targets.items()
        if counts.get(label, 0) < target
    }
    print(f"  [SMOTE] 折内目标: {class_targets}")
    if not sampling_strategy:
        print("  [SMOTE] 所有目标类别均已达到设定数量，跳过过采样")
        return train_major.copy()

    features = train_major[feature_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(features.to_numpy()).all():
        raise ValueError("SMOTE 输入仍有缺失值或无穷值，请检查折内插补结果")

    # 近邻计算前做标准化，避免 wt.% 与 ppm 的量纲差异主导距离。
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    smallest_target_class = min(
        counts[label] for label in sampling_strategy
    )
    k_neighbors = min(SMOTE_K_NEIGHBORS, smallest_target_class - 1)

    sampler = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=SMOTE_RANDOM_STATE,
        k_neighbors=k_neighbors,
    )
    resampled_scaled, resampled_labels = sampler.fit_resample(
        scaled_features,
        labels,
    )
    resampled_features = scaler.inverse_transform(resampled_scaled)
    resampled_features = np.clip(resampled_features, 0.0, None)

    result = pd.DataFrame(resampled_features, columns=feature_columns)
    result[LABEL_COLUMN] = resampled_labels
    result = result[[LABEL_COLUMN] + feature_columns]
    print(
        f"  [SMOTE] 折内训练样本: {len(train_major)} -> {len(result)}"
    )
    return result


def normalize_fold(
    fit_data,
    train_data,
    validation_data,
    quantile_normalization_module,
):
    """仅使用 SMOTE 前的折内训练数据拟合分位数边界。"""
    feature_columns = quantile_normalization_module.COLUMNS_TO_EXTRACT
    quantile_params = quantile_normalization_module.fit_quantile_boundaries(
        fit_data,
        feature_columns,
    )
    train_normalized = quantile_normalization_module.apply_quantile_transform(
        train_data,
        feature_columns,
        quantile_params,
        "折内训练集",
    )
    validation_normalized = quantile_normalization_module.apply_quantile_transform(
        validation_data,
        feature_columns,
        quantile_params,
        "折内验证集",
    )
    return train_normalized, validation_normalized


def prepare_model_arrays(
    train_data,
    validation_data,
    train_mask,
    validation_mask,
    image_columns,
    sequence_columns,
    label_order,
):
    """把归一化后的表格转换为双流模型需要的数组（数值 + 缺失 mask）。"""
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

    train_image = train_image_2d.reshape(-1, 1, 6, 6)
    validation_image = validation_image_2d.reshape(-1, 1, 6, 6)
    train_sequence = train_sequence_2d[:, :, np.newaxis]
    validation_sequence = validation_sequence_2d[:, :, np.newaxis]

    if USE_MISSING_MASK:
        # 中文注释：SMOTE 合成行追加在原始行之后，其原始缺失 mask 统一为 0。
        synthetic_count = len(train_data) - len(train_mask)
        if synthetic_count < 0:
            raise ValueError("SMOTE 后训练行数小于原始 mask 行数")
        synthetic_mask = pd.DataFrame(
            0,
            index=np.arange(synthetic_count),
            columns=train_mask.columns,
            dtype=np.uint8,
        )
        full_train_mask = pd.concat(
            [train_mask, synthetic_mask],
            ignore_index=True,
        )

        def mask_arrays(mask_frame, columns):
            values = mask_frame[columns].to_numpy(dtype=np.float32)
            return values

        train_img_mask = mask_arrays(full_train_mask, image_columns).reshape(-1, 1, 6, 6)
        val_img_mask = mask_arrays(validation_mask, image_columns).reshape(-1, 1, 6, 6)
        train_seq_mask = mask_arrays(full_train_mask, sequence_columns)[:, :, np.newaxis]
        val_seq_mask = mask_arrays(validation_mask, sequence_columns)[:, :, np.newaxis]

        train_image = np.concatenate([train_image, train_img_mask], axis=1)
        validation_image = np.concatenate([validation_image, val_img_mask], axis=1)
        train_sequence = np.concatenate([train_sequence, train_seq_mask], axis=2)
        validation_sequence = np.concatenate(
            [validation_sequence, val_seq_mask], axis=2
        )

    return (
        train_image.astype(np.float32),
        train_sequence.astype(np.float32),
        train_labels.to_numpy(dtype=np.int64),
        validation_image.astype(np.float32),
        validation_sequence.astype(np.float32),
        validation_labels.to_numpy(dtype=np.int64),
    )


def build_experiments(model_module, num_classes):
    """按配置创建需要运行的模型（与主流程一致开启缺失编码）。"""
    all_experiments = {
        "Full": (
            "Full Model\n(ViT+Transformer)",
            lambda: model_module.ViT_Transformer_DualStream(
                num_classes=num_classes,
                use_missing_mask=USE_MISSING_MASK,
            ),
        ),
        "Abl-1": (
            "Abl-1\nViT Only (Matrix)",
            lambda: model_module.Ablation_ViT_Only(
                num_classes=num_classes,
                use_missing_mask=USE_MISSING_MASK,
            ),
        ),
        "Abl-2": (
            "Abl-2\nTransformer Only (Seq)",
            lambda: model_module.Ablation_Transformer_Only(
                num_classes=num_classes,
                use_missing_mask=USE_MISSING_MASK,
            ),
        ),
        "Abl-3": (
            "Abl-3\nw/o Pos Encoding",
            lambda: model_module.Ablation_NoPositionalEncoding(
                num_classes=num_classes,
                use_missing_mask=USE_MISSING_MASK,
            ),
        ),
        "Cmp-1": (
            "Cmp-1\nCNN-BiLSTM (EMSPN)",
            lambda: model_module.CNN_BiLSTM(
                num_classes=num_classes,
                use_missing_mask=USE_MISSING_MASK,
            ),
        ),
        "Cmp-2": (
            "Cmp-2\nCNN-ViT-Transformer",
            lambda: model_module.CNN_ViT_Transformer(
                num_classes=num_classes,
                use_missing_mask=USE_MISSING_MASK,
            ),
        ),
        "Cmp-3": (
            "Cmp-3\nCNN Only",
            lambda: model_module.Baseline_CNN_Only(
                num_classes=num_classes,
                use_missing_mask=USE_MISSING_MASK,
            ),
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
    smote_module = load_module(
        "kfold_smote",
        SMOTE_MODULE_FILE,
    )
    model_module = load_module(
        "kfold_model",
        MODEL_MODULE_FILE,
    )
    # 中文注释：降低训练日志打印间隔，便于在日志文件中跟踪每个 epoch。
    model_module.PRINT_EVERY = PRINT_EVERY

    raw_train_data = pd.read_csv(TRAIN_RAW_FILE, low_memory=False)
    validate_input_data(raw_train_data)
    label_order = sorted(raw_train_data[LABEL_COLUMN].unique().tolist())
    num_classes = len(label_order)
    full_train_count = len(raw_train_data)

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

        # ① 全局插补（折内训练 fit，验证 transform）+ 原始缺失 mask
        fold_train_imputed, fold_train_mask, fold_global_model = (
            fit_and_impute_training_fold(fold_train_raw, imputation_module)
        )
        fold_validation_imputed, fold_validation_mask = impute_validation_fold(
            fold_validation_raw,
            fold_global_model,
            imputation_module,
        )

        # ② 主量无水标准化（逐行计算，不存在拟合参数）
        fold_train_major = major_normalization_module.normalize_major_elements(
            fold_train_imputed
        )
        fold_validation_major = major_normalization_module.normalize_major_elements(
            fold_validation_imputed
        )

        # ③ 仅折内训练数据执行选择性 SMOTE
        fold_train_smote = smote_training_fold(
            fold_train_major,
            smote_module,
            full_train_count,
        )

        # ④ 分位数边界从 SMOTE 前的折内训练数据拟合
        fold_train_normalized, fold_validation_normalized = normalize_fold(
            fold_train_major,
            fold_train_smote,
            fold_validation_major,
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
            fold_train_mask,
            fold_validation_mask,
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
    # 中文注释：每次运行创建独立日志，同时保留终端和工作流界面的实时输出。
    run_log_file, run_log_stream = create_run_log()
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, run_log_stream)
    sys.stderr = TeeStream(original_stderr, run_log_stream)

    try:
        print("=" * 75, flush=True)
        print(f"5 折训练日志文件: {run_log_file}", flush=True)
        print(f"启动时间: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
        print("=" * 75, flush=True)
        main()
        print(f"结束时间: {datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    except Exception:
        print("\n5 折训练发生异常，完整堆栈如下:", file=sys.stderr, flush=True)
        traceback.print_exc(file=sys.stderr)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        run_log_stream.close()
