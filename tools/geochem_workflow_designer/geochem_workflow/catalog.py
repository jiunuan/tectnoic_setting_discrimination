from __future__ import annotations

from copy import deepcopy


FEATURE_COLUMNS = "\n".join(
    [
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
)

MAJOR_COLUMNS = "\n".join(
    [
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
    ]
)


NODE_TEMPLATES = [
    {
        "type": "note",
        "label": "说明卡片",
        "category": "Layout",
        "runnable": False,
        "description": "Only for annotation and visual grouping.",
        "params": [
            {
                "key": "content",
                "label": "Content",
                "type": "textarea",
                "default": "Drag operators here and connect the pipeline.",
            }
        ],
    },
    {
        "type": "command_task",
        "label": "脚本桥接",
        "category": "Bridge",
        "runnable": True,
        "description": "Run an existing Python script or shell command from your current project.",
        "params": [
            {"key": "label", "label": "Step Label", "type": "text", "default": ""},
            {
                "key": "command",
                "label": "Command",
                "type": "textarea",
                "default": '"{python_exe}" "{repo_root}\\data_preprocess\\combine_list.py"',
            },
            {
                "key": "workdir",
                "label": "Working Dir",
                "type": "text",
                "default": "{repo_root}",
            },
        ],
    },
    {
        "type": "rebuild_refined_basalt_labels",
        "label": "合并海洋/陆地标签",
        "category": "Bridge",
        "runnable": True,
        "description": "Run rebuild_refined_basalt_from_revised_labels.py with editable input/output paths.",
        "params": [
            {
                "key": "sea_path",
                "label": "Sea CSV",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/03_sea_unknown_review_working/03-2_intra_oceanic_arc_candidates_labeled_revised.csv",
            },
            {
                "key": "land_path",
                "label": "Land CSV",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/06_land_unknown_review_working/06-5_convergent_margin_unknown_location_type_subset_labeled_revised.csv",
            },
            {
                "key": "merged_raw_path",
                "label": "Merged Raw CSV",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/georoc/merged_raw/basalt_2025.csv",
            },
            {
                "key": "convergent_output_path",
                "label": "Convergent Output",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/06-6_convergent_margin_final_labeled_revised.csv",
            },
            {
                "key": "basalt_output_path",
                "label": "Refined Basalt Output",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/08_basalt_refined_tectonic_settings/08_basalt_refined_tectonic_settings_revised.csv",
            },
            {
                "key": "workdir",
                "label": "Working Dir",
                "type": "text",
                "default": "{repo_root}",
            },
        ],
    },
    {
        "type": "cnn_bilstm_train",
        "label": "CNN-BiLSTM 训练",
        "category": "Model",
        "runnable": True,
        "description": "Run cnn_bilstm_presplit.py with editable training hyperparameters and output names.",
        "params": [
            {
                "key": "train_file",
                "label": "Train CSV",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/basalt_combine/dataset_split_correct/12_normalize_basalt_train_7000.csv",
            },
            {
                "key": "test_file",
                "label": "Test CSV",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/basalt_combine/dataset_split_correct/13_normalize_basalt_test_7000.csv",
            },
            {
                "key": "output_dir",
                "label": "Output Dir",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/basalt_combine/dataset_split_correct/CNN_BiLSTM",
            },
            {"key": "seed", "label": "Random Seed", "type": "number", "default": 42},
            {"key": "batch_size", "label": "Batch Size", "type": "number", "default": 64},
            {"key": "epochs", "label": "Epochs", "type": "number", "default": 200},
            {"key": "lr", "label": "Learning Rate", "type": "number", "default": 0.0001},
            {"key": "weight_decay", "label": "Weight Decay", "type": "number", "default": 0.03},
            {"key": "scheduler_patience", "label": "Scheduler Patience", "type": "number", "default": 15},
            {"key": "early_stopping_patience", "label": "Early Stopping Patience", "type": "number", "default": 40},
            {"key": "mixup_alpha", "label": "MixUp Alpha", "type": "number", "default": 0.4},
            {"key": "label_smoothing", "label": "Label Smoothing", "type": "number", "default": 0.2},
            {"key": "file_stem", "label": "File Stem", "type": "text", "default": "basalt"},
            {"key": "model_name", "label": "Model Filename", "type": "text", "default": "cnn_bilstm_best.pth"},
            {"key": "results_name", "label": "Results CSV", "type": "text", "default": "CNN_BiLSTM_results.csv"},
            {"key": "workdir", "label": "Working Dir", "type": "text", "default": "{repo_root}"},
        ],
    },
    {
        "type": "extract_georoc_filter",
        "label": "GEOROC 筛选",
        "category": "Bridge",
        "runnable": True,
        "description": "Run extract_georoc.py with editable input and output paths.",
        "params": [
            {
                "key": "file_path",
                "label": "Input CSV",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/08_basalt_refined_tectonic_settings/08_basalt_refined_tectonic_settings_revised.csv",
            },
            {
                "key": "output_path",
                "label": "Output CSV",
                "type": "text",
                "default": "{repo_root}/data/rockType/basalt/georoc/processed/detail_filtered_with_metadata/08_basalt_refined_tectonic_settings_origin_filtered.csv",
            },
            {
                "key": "workdir",
                "label": "Working Dir",
                "type": "text",
                "default": "{repo_root}",
            },
        ],
    },
    {
        "type": "merge_georoc_petdb",
        "label": "GEOROC+PetDB 合并",
        "category": "Bridge",
        "runnable": True,
        "description": "Merge the prepared GEOROC and PetDB CSV files into one combined dataset.",
        "params": [
            {
                "key": "georoc_input_path",
                "label": "GEOROC Input CSV",
                "type": "text",
                "default": "{repo_root}\\data\\rockType\\basalt\\georoc\\tectonic_split\\08_basalt_refined_tectonic_settings\\09_basalt_refined_tectonic_settings_corrected_filtered.csv",
            },
            {
                "key": "petdb_input_path",
                "label": "PetDB Input CSV",
                "type": "text",
                "default": "{repo_root}\\data\\rockType\\basalt\\petDB\\processed\\petDB.csv",
            },
            {
                "key": "output_path",
                "label": "Output CSV",
                "type": "text",
                "default": "{repo_root}\\data\\rockType\\basalt\\basalt_combine\\01_basalt_number_year_correct.csv",
            },
        ],
    },
    {
        "type": "stratified_split",
        "label": "分层切分",
        "category": "Dataset",
        "runnable": True,
        "description": "Split the merged dataset into train/test while preserving label ratios.",
        "params": [
            {
                "key": "input_path",
                "label": "Input CSV",
                "type": "text",
                "default": "{repo_root}\\data\\rockType\\basalt\\basalt_combine\\01_basalt_number_year_correct.csv",
            },
            {"key": "label_column", "label": "Label Column", "type": "text", "default": "TECTONIC SETTING"},
            {"key": "test_size", "label": "Test Size", "type": "number", "default": 0.2},
            {"key": "random_state", "label": "Random Seed", "type": "number", "default": 42},
            {
                "key": "output_dir",
                "label": "Output Dir",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\split",
            },
            {"key": "output_prefix", "label": "Output Prefix", "type": "text", "default": "modern_basalt"},
        ],
    },
    {
        "type": "iqr_filter",
        "label": "IQR 异常值剔除",
        "category": "Clean",
        "runnable": True,
        "description": "Remove rows whose selected features are outside the IQR threshold.",
        "params": [
            {
                "key": "input_path",
                "label": "Input CSV",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\split\\modern_basalt_train.csv",
            },
            {
                "key": "output_path",
                "label": "Output CSV",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\train\\modern_basalt_train_iqr.csv",
            },
            {
                "key": "feature_columns",
                "label": "Feature Columns",
                "type": "textarea",
                "default": FEATURE_COLUMNS,
            },
            {"key": "threshold", "label": "IQR Multiplier", "type": "number", "default": 6},
        ],
    },
    {
        "type": "missforest_train_test",
        "label": "MissForest 训练+应用",
        "category": "Impute",
        "runnable": True,
        "description": "Fit MissForest only on train and apply the fitted models to train/test.",
        "params": [
            {
                "key": "train_input_path",
                "label": "Train Input",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\train\\modern_basalt_train_iqr.csv",
            },
            {
                "key": "test_input_path",
                "label": "Test Input",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\split\\modern_basalt_test.csv",
            },
            {
                "key": "train_output_path",
                "label": "Train Output",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\train\\modern_basalt_train_imputed.csv",
            },
            {
                "key": "test_output_path",
                "label": "Test Output",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\test\\modern_basalt_test_imputed.csv",
            },
            {
                "key": "model_output_path",
                "label": "Model Bundle",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\models\\missforest_bundle.pkl",
            },
            {
                "key": "feature_columns",
                "label": "Feature Columns",
                "type": "textarea",
                "default": FEATURE_COLUMNS,
            },
            {"key": "n_estimators", "label": "RF Trees", "type": "number", "default": 300},
            {"key": "random_state", "label": "Random Seed", "type": "number", "default": 42},
        ],
    },
    {
        "type": "smote_balance",
        "label": "SMOTE 过采样",
        "category": "Balance",
        "runnable": True,
        "description": "Oversample selected training classes to explicit targets; test data is never sampled.",
        "params": [
            {
                "key": "input_path",
                "label": "Input CSV",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\train\\modern_basalt_train_imputed.csv",
            },
            {
                "key": "output_path",
                "label": "Output CSV",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\train\\modern_basalt_train_smote_7000.csv",
            },
            {"key": "label_column", "label": "Label Column", "type": "text", "default": "TECTONIC SETTING"},
            {"key": "target_count", "label": "Target Per Class", "type": "number", "default": 7000},
            {
                "key": "class_targets",
                "label": "Selected Class Targets (LABEL=COUNT)",
                "type": "textarea",
                "default": "Island arc=2200\nBACK-ARC_BASIN=2200\nIntra-oceanic arc=2200\nOCEANIC PLATEAU=2200",
            },
            {
                "key": "feature_columns",
                "label": "Feature Columns",
                "type": "textarea",
                "default": FEATURE_COLUMNS,
            },
            {"key": "k_neighbors", "label": "K Neighbors", "type": "number", "default": 5},
            {"key": "random_state", "label": "Random Seed", "type": "number", "default": 42},
        ],
    },
    {
        "type": "anhydrous_normalize",
        "label": "无水归一化",
        "category": "Normalize",
        "runnable": True,
        "description": "Normalize 10 major oxides row-wise to 100 wt%.",
        "params": [
            {
                "key": "input_path",
                "label": "Input CSV",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\train\\modern_basalt_train_smote_7000.csv",
            },
            {
                "key": "output_path",
                "label": "Output CSV",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\train\\modern_basalt_train_anhydrous.csv",
            },
            {
                "key": "major_columns",
                "label": "Major Columns",
                "type": "textarea",
                "default": MAJOR_COLUMNS,
            },
        ],
    },
    {
        "type": "quantile_fit",
        "label": "分位数拟合",
        "category": "Normalize",
        "runnable": True,
        "description": "Fit train-only quantile bins and optionally transform the train CSV.",
        "params": [
            {
                "key": "input_path",
                "label": "Train Input",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\train\\modern_basalt_train_anhydrous.csv",
            },
            {
                "key": "params_output_path",
                "label": "Params JSON",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\models\\quantile_bins_255.json",
            },
            {
                "key": "transformed_output_path",
                "label": "Train Output",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\final\\modern_basalt_train_final_1_255.csv",
            },
            {
                "key": "feature_columns",
                "label": "Feature Columns",
                "type": "textarea",
                "default": FEATURE_COLUMNS,
            },
            {"key": "bins", "label": "Bin Count", "type": "number", "default": 255},
        ],
    },
    {
        "type": "quantile_apply",
        "label": "分位数应用",
        "category": "Normalize",
        "runnable": True,
        "description": "Apply train-fitted quantile bins to the test CSV.",
        "params": [
            {
                "key": "input_path",
                "label": "Input CSV",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\test\\modern_basalt_test_anhydrous.csv",
            },
            {
                "key": "params_input_path",
                "label": "Params JSON",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\models\\quantile_bins_255.json",
            },
            {
                "key": "output_path",
                "label": "Output CSV",
                "type": "text",
                "default": "{artifact_root}\\modern_basalt_safe\\final\\modern_basalt_test_final_1_255.csv",
            },
            {
                "key": "feature_columns",
                "label": "Feature Columns",
                "type": "textarea",
                "default": FEATURE_COLUMNS,
            },
        ],
    },
]


NODE_TEMPLATES_BY_TYPE = {template["type"]: template for template in NODE_TEMPLATES}


def build_template_defaults(node_type: str) -> dict:
    template = NODE_TEMPLATES_BY_TYPE[node_type]
    return {item["key"]: deepcopy(item.get("default")) for item in template["params"]}
