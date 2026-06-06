"""
合并训练集插补结果
================================================================
``imputation_train_predict.py`` 阶段 1 会为每个构造环境类别分别输出一个
``<类别>_clean_imputed.csv``（位于 ``data/05_imputed/MissForest/``）。

本脚本把这些按类别分文件的训练集插补结果合并为单一训练集
``data/05_imputed/02_basalt_train_imputed.csv``，作为后续主量无水标准化
（``03_normalization/normalize_major_elements.py``）的输入。

注意：测试集插补结果 ``02_basalt_test_imputed.csv`` 已由
``imputation_train_predict.py`` 直接输出，无需合并。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import MISSFOREST_DIR, TRAIN_IMPUTED_CSV


def main():
    files = sorted(MISSFOREST_DIR.glob("*_clean_imputed.csv"))
    if not files:
        raise FileNotFoundError(
            f"未在 {MISSFOREST_DIR} 找到任何 *_clean_imputed.csv，"
            f"请先运行 imputation_train_predict.py 生成各类别插补结果。"
        )
    print(f"发现 {len(files)} 个类别插补文件，开始合并：")
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        print(f"  + {f.name:<50s} {df.shape}")
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)
    TRAIN_IMPUTED_CSV.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(TRAIN_IMPUTED_CSV, index=False)
    print(f"\n合并完成：{TRAIN_IMPUTED_CSV}  shape={merged.shape}")


if __name__ == "__main__":
    main()
