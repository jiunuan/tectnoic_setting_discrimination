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
│   ├── georoc/references_structured.csv        GEOROC 参考文献编号→年份映射（筛选脚本引用）
│   └── petdb/petdbv2_merged.csv                PetDB 2.0 原始下载合并表
│
├── 01_cm_reclass_input/            # 汇聚边缘细分项目的产出（输入起点）
│   ├── georoc_convergent_margin_core_training_high_confidence.csv   高置信度细分
│   ├── georoc_convergent_margin_expanded_training_reviewed.csv      高+中置信度细分
│   └── basalt_refined_expanded.csv                                  用细分结果重分类后的 GEOROC 总表（全岩）
│
├── 02_filtered/                    # 经筛选规则后的数据（extract_georoc / extract_petdb 产出）
│   ├── basalt_refined_expanded_filtered.csv    GEOROC 筛选结果
│   └── petDB.csv                               PetDB 筛选结果
│
├── 03_combined/                    # GEOROC + PetDB 合并
│   └── 01_basalt_number_year.csv
│
├── 04_split/                       # 训练/测试切分（分层 0.2，seed=32）
│   ├── 01_basalt_number_year_train.csv / _test.csv
│   └── split_summary.csv
│
├── 05_imputed/                     # 全局随机森林插补 + 原始缺失 mask
│   ├── 02_basalt_train_imputed.csv             训练集插补结果（训练集 fit）
│   ├── 02_basalt_test_imputed.csv              测试集 transform 插补
│   ├── 03_train_missing_mask.csv               训练集原始缺失 mask（1=缺失，模型第二通道）
│   └── 03_test_missing_mask.csv                测试集原始缺失 mask
│
├── 06_normalized/                  # 主量无水标准化 + 选择性 SMOTE + 分位数分箱归一化
│   ├── 04_basalt_train_major_normalize.csv / 04_basalt_test_major_normalize.csv
│   ├── 05_basalt_train_selected_smote.csv      仅训练集 SMOTE（5 个少数类补到 3000）
│   ├── 06_normalize_basalt_train.csv / 06_normalize_basalt_test.csv  ← 最终喂模型 / SHAP
│   ├── 06_normalize_basalt_train_no_smote.csv  未 SMOTE 真实训练集（对照实验）
│   └── quantile_params.json                    分位参数（从 SMOTE 前训练集拟合，太古代共用）
│
├── models/                         # 训练产出权重
│   └── Full_Model_(ViT+Transformer)_best_seed.pth   主模型权重（SHAP / 太古代预测加载）
│
└── archean/                        # 太古代应用（缺失编码：不插补，数值 0 + mask 1）
    ├── data/                       Liu 2024 全球数据 + 6 克拉通案例 CSV/XLSX/ZIP
    └── outputs/                    （脚本运行时生成）
        ├── extended_archean_pool/  扩展应用集（正式 3,483 条：expanded_archean_basalt_age_nonmissing.csv）
        ├── archean_geodan_final/   正式缺失编码预测（expanded_archean_predictions.csv 等）
        └── distribution_consistency/  适用域 / 域偏移诊断
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
