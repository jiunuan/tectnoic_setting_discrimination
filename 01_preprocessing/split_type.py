import pandas as pd
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import TRAIN_RAW_CSV, SPLIT_BY_TYPE_DIR

# 读取训练集（按构造环境拆分的输入）
df = pd.read_csv(TRAIN_RAW_CSV)


# 获取 TECTONIC SETTING 列中的唯一值
tectonic_settings = df['TECTONIC SETTING'].unique()

# 为每个唯一值创建一个新的 CSV 文件
for setting in tectonic_settings:
    # 创建一个只包含当前 tectonic setting 的数据框
    df_setting = df[df['TECTONIC SETTING'] == setting]
    # 创建一个有效的文件名（替换可能导致问题的字符）
    filename = f"{setting.replace(' ', '_').replace('/', '_')}.csv"
    output_base_dir = str(SPLIT_BY_TYPE_DIR) + os.sep
    # 检查路径是否存在，如果不存在则创建
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
        print(f"目录已创建: {output_base_dir}")
    else:
        print(f"目录已存在: {output_base_dir}")
    filename = output_base_dir + filename
    # 将数据框保存为 CSV 文件
    df_setting.to_csv(filename, index=False)

    print(f"已创建文件: {filename}")

print("所有文件已创建完成。")
