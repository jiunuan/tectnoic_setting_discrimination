# Geochem Workflow Designer

一个独立的可视化工作流项目，专门为你当前的现代玄武岩地球化学处理流程搭的“算子编排层”。它的目标不是替代现有 `E:\program\CNNtest` 里的研究脚本，而是把这些步骤组织成一个可拖拽、可保存、可重复运行的节点工作流。

## 这个项目能做什么

- 用拖拽节点的方式搭你的现代玄武岩预处理与建模流程
- 保存/加载工作流 JSON
- 按 DAG 顺序执行节点，并在界面里看日志
- 内置以下可执行算子：
  - `分层切分`
  - `IQR 异常值剔除`
  - `无水归一化`
  - `分位数拟合`
  - `分位数应用`
- 内置以下依赖外部环境的高级算子：
  - `MissForest 训练+应用`
  - `SMOTE 过采样`
- 通过 `脚本桥接` 节点调用你当前项目中的现有脚本：
  - GEOROC 汇聚边缘细分
  - GEOROC / PetDB 原始预处理
  - 合并统一数据集
  - CNN-BiLSTM 训练脚本

## 预置流程

默认模板已经按你现在强调的“安全训练/测试流程”铺好了：

1. GEOROC 汇聚边缘细分
2. GEOROC 独立预处理
3. PetDB 独立预处理
4. 合并统一地化数据集
5. `8:2` 分层切分
6. 只对训练集做 `IQR`
7. 只在训练集拟合 `MissForest`，再应用到测试集
8. 只对训练集做 `SMOTE` 到 `7000`
9. 训练集 / 测试集分别做 10 主量元素无水归一化
10. 只在训练集拟合分位数参数，并应用到测试集
11. 输出最终 train/test CSV
12. 送入 CNN-BiLSTM

## 目录结构

```text
geochem_workflow_designer/
├─ app.py
├─ README.md
├─ requirements.txt
├─ artifacts/
├─ runs/
├─ workflows/
└─ geochem_workflow/
   ├─ catalog.py
   ├─ executor.py
   ├─ operations.py
   ├─ presets.py
   ├─ server.py
   ├─ settings.py
   └─ static/
      ├─ app.js
      ├─ index.html
      └─ styles.css
```

## 启动方式

在项目根目录执行：

```powershell
python app.py
```

启动后打开：

```text
http://127.0.0.1:8766
```

## 依赖说明

基础运行只需要：

- `numpy`
- `pandas`

如果你想直接运行内置的 `MissForest` 和 `SMOTE` 节点，还需要：

- `scipy`
- `scikit-learn`
- `imbalanced-learn`

## 当前版本的边界

- 这个版本的前端是原生 HTML/CSS/JS，没有引入 React Flow 一类的大前端框架，所以部署轻，但功能也更偏“研究工作流编排 MVP”。
- `脚本桥接` 节点默认直接调用你当前仓库里的脚本；如果脚本本身还使用硬编码路径，你需要在节点参数里保持上下游路径一致。
- 内置 `SMOTE` 节点输出以 `特征列 + 标签列` 为主；对合成样本无法自然继承的元数据列会留空。

## 推荐你后续继续扩展的方向

1. 把 `GEOROC/PetDB 预处理` 改造成真正的标准算子，而不是脚本桥接。
2. 给节点增加“端口语义”，让上游输出路径自动传递给下游。
3. 增加工作流版本管理和运行快照。
4. 增加“只运行选中节点及其下游”的能力。
5. 增加 CNN-BiLSTM 训练参数面板和结果图回显。
