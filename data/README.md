# 数据说明（Data Availability）

> **重要**：本目录下的数据文件体量较大（合计约 400 MB，单文件最大 ~78 MB），
> **默认不纳入 Git 版本库**（见根目录 `.gitignore`）。正式发表时，完整数据集将
> 通过 **Zenodo / figshare** 永久存档并分配 DOI；克隆仓库后，请从该存档下载数据
> 并按下方目录结构放置，即可运行全部脚本。
>
> - **Zenodo DOI**：`<待发表后填写，占位>`
> - **下载地址**：`<占位>`

---

## 数据来源

| 来源 | 说明 |
|---|---|
| **GEOROC** | 大陆与海洋玄武岩地球化学数据库（https://georoc.eu） |
| **PetDB** | 海洋与海底岩石地球化学数据库（https://search.earthchem.org/） |
| **汇聚边缘细分** | 汇聚边缘（岛弧 / 陆缘弧 / 弧后盆地等）构造环境的再分类结果，由独立项目 `convergent_margin_reclass`（含 LLM 辅助复核）产出；本仓库**不含该细分代码**，仅以其产出 CSV 作为输入起点。 |
| **太古代应用集** | Liu et al. (2024) 全球太古代玄武岩数据集 + 6 个克拉通案例研究（`archean/`）。 |

---

## 目录结构与各阶段文件含义

数据按训练流程阶段编号组织，与脚本目录 `01_preprocessing` … `07_figures` 对应：

```
data/
├── 00_raw/                         # 原始数据
│   ├── georoc/basalt_2025.csv                  GEOROC 玄武岩原始合并表
│   └── petdb/petDB_recent_downloads_merged.csv PetDB 原始下载合并表
│
├── 01_cm_reclass_input/            # 汇聚边缘细分项目的产出（输入起点）
│   ├── georoc_convergent_margin_core_training_high_confidence.csv   高置信度细分
│   ├── georoc_convergent_margin_expanded_training_reviewed.csv      高+中置信度细分
│   ├── basalt_refined_expanded.csv                                  用细分结果重分类后的 GEOROC 总表
│   └── references_structured.csv                                    参考文献结构化表（筛选 GUI 默认引用，可选）
│
├── 02_filtered/                    # 经筛选规则后的数据
│   ├── basalt_refined_expanded_filtered.csv    GEOROC 筛选结果
│   └── petDB.csv                               PetDB 筛选结果
│
├── 03_combined/                    # GEOROC + PetDB 合并
│   └── 01_basalt_number_year.csv
│
├── 04_split/                       # 训练/测试切分、按构造环境拆分、IQR 去离群
│   ├── 01_basalt_number_year_train.csv / _test.csv
│   ├── split_summary.csv
│   └── preprocess/
│       ├── split/<构造环境>.csv               训练集按构造环境拆分
│       └── clean/<构造环境>_clean.csv         IQR 去离群后
│
├── 05_imputed/                     # 按构造环境类别 MissForest 插补
│   ├── MissForest/<构造环境>_clean_imputed.csv  各类别插补结果
│   ├── 02_basalt_train_imputed.csv             合并后的训练集（merge_imputed_trainset.py 产出）
│   └── 02_basalt_test_imputed.csv              测试集 transform 插补
│
├── 06_normalized/                  # 主量无水标准化 + 分位数分箱归一化
│   ├── 03_basalt_train_major_normalize.csv / 03_basalt_test_major_normalize.csv
│   ├── 05_normalize_basalt_train.csv / 05_normalize_basalt_test.csv  ← 最终喂模型 / SHAP
│   └── quantile_params.json                    训练集分位参数（测试集 / 太古代应用集共用）
│
├── models/                         # 训练产出权重
│   └── Full_Model_(ViT+Transformer)_best_seed.pth   主模型权重（SHAP / 太古代预测加载）
│
└── archean/                        # 太古代应用（no_impute 模式）
    ├── data/                       Liu 2024 全球数据 + 6 克拉通案例 CSV/XLSX/ZIP
    └── outputs/                    （脚本运行时生成：预处理 / 预测 / 图件）
```

---

## 36 个地球化学特征列

模型输入为 36 个主量 + 微量元素（主量 WT%、微量 PPM）：

```
NA2O MGO AL2O3 SIO2 P2O5 K2O CAO TIO2 MNO FEOT   (10 主量氧化物, WT%)
RB V CR CO NI BA SR Y ZR NB LA CE PR ND SM EU GD TB DY HO ER YB LU HF TA TH  (26 微量, PPM)
```

构造为两种排列：**6×6 地化亲缘矩阵**（ViT 分支）与 **按不相容性排序的 36 元素序列**（Transformer 分支）。

## 9 类构造环境标签（`TECTONIC SETTING`）

`SPREADING_CENTER`、`OCEAN_ISLAND`、`CONTINENTAL_RIFT`、`OCEANIC_PLATEAU`、
`CONTINENTAL_FLOOD_BASALT`、`BACK-ARC_BASIN`、`Island_arc`、`Continental_arc`、
`Intra-oceanic_arc`。
