from __future__ import annotations


def _node(node_id: str, node_type: str, title: str, x: int, y: int, params: dict | None = None) -> dict:
    return {
        "id": node_id,
        "type": node_type,
        "title": title,
        "position": {"x": x, "y": y},
        "params": params or {},
    }


def build_default_workflow() -> dict:
    nodes = [
        _node(
            "n1",
            "note",
            "目录组织建议",
            40,
            40,
            {
                "content": "默认建议继续沿用 E:\\program\\CNNtest 下的既有输入/输出目录。\n原因：当前多数研究脚本仍是硬编码路径，移动到 workflow 程序目录会增加维护成本。\nworkflow 程序目录只保存流程定义、helper 脚本和运行日志。"
            },
        ),
        _node(
            "n2",
            "rebuild_refined_basalt_labels",
            "1. 合并海洋/陆地标签",
            360,
            40,
            {
                "sea_path": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/03_sea_unknown_review_working/03-2_intra_oceanic_arc_candidates_labeled_revised.csv",
                "land_path": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/06_land_unknown_review_working/06-5_convergent_margin_unknown_location_type_subset_labeled_revised.csv",
                "merged_raw_path": "{repo_root}/data/rockType/basalt/georoc/merged_raw/basalt_2025.csv",
                "convergent_output_path": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/06-6_convergent_margin_final_labeled_revised.csv",
                "basalt_output_path": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/08_basalt_refined_tectonic_settings/08_basalt_refined_tectonic_settings_revised.csv",
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n3",
            "extract_georoc_filter",
            "2. GEOROC 筛选",
            640,
            40,
            {
                "file_path": "{repo_root}/data/rockType/basalt/georoc/tectonic_split/08_basalt_refined_tectonic_settings/08_basalt_refined_tectonic_settings_revised.csv",
                "output_path": "{repo_root}/data/rockType/basalt/georoc/processed/detail_filtered_with_metadata/08_basalt_refined_tectonic_settings_origin_filtered.csv",
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n4",
            "note",
            "PetDB 输入",
            640,
            240,
            {
                "content": "当前流程直接复用：\nE:\\program\\CNNtest\\data\\rockType\\basalt\\petDB\\processed\\petDB.csv"
            },
        ),
        _node(
            "n5",
            "merge_georoc_petdb",
            "3. GEOROC+PetDB 合并",
            920,
            110,
            {
                "georoc_input_path": "{repo_root}\\data\\rockType\\basalt\\georoc\\tectonic_split\\08_basalt_refined_tectonic_settings\\09_basalt_refined_tectonic_settings_corrected_filtered.csv",
                "petdb_input_path": "{repo_root}\\data\\rockType\\basalt\\petDB\\processed\\petDB.csv",
                "output_path": "{repo_root}\\data\\rockType\\basalt\\basalt_combine\\01_basalt_number_year_correct.csv",
            },
        ),
        _node(
            "n6",
            "command_task",
            "4. 训练/测试拆分",
            1200,
            110,
            {
                "label": "split_train_test.py",
                "command": '"{python_exe}" "{repo_root}\\data_preprocess\\split_train_test.py"',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n7",
            "command_task",
            "5. 训练集按构造环境拆分",
            1480,
            40,
            {
                "label": "split_type.py",
                "command": '"{python_exe}" "{repo_root}\\data_preprocess\\split_type.py"',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n8",
            "command_task",
            "6. split 清洗",
            1760,
            40,
            {
                "label": "Outlier_Detection_Basalt.py",
                "command": '"{python_exe}" "{repo_root}\\data_interpolation\\Outlier_Detection_Basalt.py"',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n9",
            "command_task",
            "7. 训练集 MissForest",
            2040,
            40,
            {
                "label": "imputation_train.py",
                "command": '"{python_exe}" "{repo_root}\\data_interpolation\\imputation_train.py"',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n10",
            "command_task",
            "8. 测试集插补",
            1480,
            260,
            {
                "label": "imputation_predict.py",
                "command": '"{python_exe}" "{repo_root}\\data_interpolation\\imputation_predict.py"',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n11",
            "command_task",
            "9. 合并训练集插补结果",
            2320,
            40,
            {
                "label": "combine_all_batch.py",
                "command": '"{python_exe}" "{repo_root}\\data_preprocess\\combine_all_batch.py"',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n12",
            "command_task",
            "10. 主量无水归一 train",
            2600,
            40,
            {
                "label": "normalize_major_elements.py train",
                "command": '"{python_exe}" "{workflow_root}\\script_helpers\\run_with_mode.py" --script "{repo_root}\\data_preprocess\\data_normalize\\normalize_major_elements.py" --mode train',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n13",
            "command_task",
            "10. 主量无水归一 predict",
            2320,
            260,
            {
                "label": "normalize_major_elements.py predict",
                "command": '"{python_exe}" "{workflow_root}\\script_helpers\\run_with_mode.py" --script "{repo_root}\\data_preprocess\\data_normalize\\normalize_major_elements.py" --mode predict',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n14",
            "command_task",
            "11. BorderlineSMOTE",
            2880,
            40,
            {
                "label": "BorderlineSMOTE1.py",
                "command": '"{python_exe}" "{repo_root}\\data_balance\\BorderlineSMOTE1.py"',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n15",
            "command_task",
            "12. 分位数归一 train",
            3160,
            40,
            {
                "label": "normalize.py train",
                "command": '"{python_exe}" "{workflow_root}\\script_helpers\\run_with_mode.py" --script "{repo_root}\\data_preprocess\\data_normalize\\normalize.py" --mode train',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n16",
            "command_task",
            "12. 分位数归一 predict",
            2880,
            260,
            {
                "label": "normalize.py predict",
                "command": '"{python_exe}" "{workflow_root}\\script_helpers\\run_with_mode.py" --script "{repo_root}\\data_preprocess\\data_normalize\\normalize.py" --mode predict',
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n17",
            "cnn_bilstm_train",
            "13. CNN-BiLSTM 训练",
            3440,
            110,
            {
                "train_file": "{repo_root}/data/rockType/basalt/basalt_combine/dataset_split_correct/12_normalize_basalt_train_7000.csv",
                "test_file": "{repo_root}/data/rockType/basalt/basalt_combine/dataset_split_correct/13_normalize_basalt_test_7000.csv",
                "output_dir": "{repo_root}/data/rockType/basalt/basalt_combine/dataset_split_correct/CNN_BiLSTM",
                "seed": 42,
                "batch_size": 64,
                "epochs": 200,
                "lr": 0.0001,
                "weight_decay": 0.03,
                "scheduler_patience": 15,
                "early_stopping_patience": 40,
                "mixup_alpha": 0.4,
                "label_smoothing": 0.2,
                "file_stem": "basalt",
                "model_name": "cnn_bilstm_best.pth",
                "results_name": "CNN_BiLSTM_results.csv",
                "workdir": "{repo_root}",
            },
        ),
        _node(
            "n18",
            "note",
            "结果输出",
            3720,
            110,
            {
                "content": "最终训练结果默认写入：\nE:\\program\\CNNtest\\data\\rockType\\basalt\\basalt_combine\\dataset_split_correct\\CNN_BiLSTM\n\n这套默认模板优先保证和现有研究脚本兼容，而不是强行迁移目录。"
            },
        ),
    ]

    edges = [
        {"id": "e1", "source": "n1", "target": "n2"},
        {"id": "e2", "source": "n2", "target": "n3"},
        {"id": "e3", "source": "n3", "target": "n5"},
        {"id": "e4", "source": "n4", "target": "n5"},
        {"id": "e5", "source": "n5", "target": "n6"},
        {"id": "e6", "source": "n6", "target": "n7"},
        {"id": "e7", "source": "n7", "target": "n8"},
        {"id": "e8", "source": "n8", "target": "n9"},
        {"id": "e9", "source": "n6", "target": "n10"},
        {"id": "e9b", "source": "n9", "target": "n10"},
        {"id": "e10", "source": "n9", "target": "n11"},
        {"id": "e11", "source": "n11", "target": "n12"},
        {"id": "e12", "source": "n10", "target": "n13"},
        {"id": "e13", "source": "n12", "target": "n14"},
        {"id": "e14", "source": "n14", "target": "n15"},
        {"id": "e15", "source": "n13", "target": "n16"},
        {"id": "e16", "source": "n15", "target": "n17"},
        {"id": "e17", "source": "n16", "target": "n17"},
        {"id": "e18", "source": "n17", "target": "n18"},
    ]

    return {
        "name": "current_script_chain_cnn_bilstm",
        "description": "Workflow aligned to the current hard-coded CNNtest script chain.",
        "nodes": nodes,
        "edges": edges,
    }
