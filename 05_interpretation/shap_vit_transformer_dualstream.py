"""
ViT-Transformer Dual-Stream 玄武岩构造环境判别 —— SHAP 可解释性分析
=============================================================
模型结构：双分支（无 CNN）
  - ViT 矩阵分支：输入 (B, 1, 6, 6)，特征排列由 COLUMNS_IMG_ORDER 决定
  - 序列 Transformer 分支：输入 (B, 36, 1)，特征排列由 COLUMNS_ELECTRODE_ORDER 决定
    （按不相容性从高到低排列，与训练脚本 SEQUENCE_COLUMNS_V2 一致）

处理策略：
  1. 对两路分支分别用 shap.GradientExplainer 计算梯度 SHAP 值
  2. 按特征名将两路 SHAP 值对齐后相加，得到每个元素的综合重要性
  3. 输出：堆叠条形图（所有类别汇总）、各类别 violin/beeswarm 图

绘图（SCI 出版风格）：
  - plot_stacked_bar          : 竖直堆叠柱状图（总览）
  - plot_overall_beeswarm     : 整体 violin + beeswarm（紧凑版）
  - plot_per_class_beeswarm   : 每个构造环境一张紧凑 violin + beeswarm
  - plot_heatmap              : 类别 × 元素 热力图
"""

import os
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import shap
import matplotlib
matplotlib.use('Agg')  # 无显示器时使用 Agg 后端
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
# ①  路径配置 —— 修改这里即可
#     亦可由 geodan_main_model.py 通过环境变量覆盖：
#       GEODAN_MODEL_PATH    : 模型权重路径
#       GEODAN_COLUMN_SCHEME : 'v1' 或 'v2'
#       GEODAN_OUTPUT_DIR    : 输出根目录（SHAP 子目录在其下）
# ══════════════════════════════════════════════════════════════
import os as _os
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from config.paths import MAIN_MODEL_WEIGHT, TRAIN_NORM_CSV, TEST_NORM_CSV, MODELS_DIR

# 列排列方案：'v1'=元素周期表+电极电势  'v2'=亲缘矩阵+不相容性序列
COLUMN_ORDER_SCHEME = 'v1'

MODEL_PATH = str(MAIN_MODEL_WEIGHT)

TRAIN_FILE  = str(TRAIN_NORM_CSV)
TEST_FILE   = str(TEST_NORM_CSV)

_BASE_OUTPUT = _os.environ.get('GEODAN_OUTPUT_DIR', str(MODELS_DIR))
OUTPUT_DIR  = _os.path.join(_BASE_OUTPUT, 'shap_analysis')

MAX_BACKGROUND   = 500    # 背景数据集样本数（越大越精确，但越慢）
MAX_EXPLAIN      = 1000   # 用于解释的测试集样本数
SHAP_BATCH_SIZE  = 50     # 每批计算的样本数（显存不足可调小）

# ══════════════════════════════════════════════════════════════
# ②  特征列顺序（与 geodan_main_model.py 的 COLUMN_ORDER_SCHEME 保持一致）
# ══════════════════════════════════════════════════════════════

# ── v1: 矩阵分支 — 元素周期表顺序 ──────────────────────────────
_COLUMNS_IMG_V1 = [
    'NA2O(WT%)', 'MGO(WT%)',   'CR(PPM)',    'AL2O3(WT%)', 'SIO2(WT%)',  'P2O5(WT%)',
    'K2O(WT%)',  'CAO(WT%)',   'TIO2(WT%)',  'V(PPM)',     'MNO(WT%)',   'FEOT(WT%)',
    'RB(PPM)',   'SR(PPM)',    'Y(PPM)',     'NB(PPM)',    'CO(PPM)',    'NI(PPM)',
    'BA(PPM)',   'LA(PPM)',    'CE(PPM)',    'PR(PPM)',    'ND(PPM)',    'ZR(PPM)',
    'SM(PPM)',   'EU(PPM)',    'GD(PPM)',    'TB(PPM)',    'DY(PPM)',    'HO(PPM)',
    'TH(PPM)',   'ER(PPM)',    'YB(PPM)',    'LU(PPM)',    'HF(PPM)',    'TA(PPM)',
]

# ── v1: 序列分支 — 电极电势序列 ────────────────────────────────
_COLUMNS_SEQ_V1 = [
    'RB(PPM)',   'K2O(WT%)',  'BA(PPM)',    'SR(PPM)',    'CAO(WT%)',  'NA2O(WT%)',
    'LA(PPM)',   'Y(PPM)',    'MGO(WT%)',   'PR(PPM)',    'CE(PPM)',   'ER(PPM)',
    'HO(PPM)',   'ND(PPM)',   'SM(PPM)',    'DY(PPM)',    'LU(PPM)',   'TB(PPM)',
    'GD(PPM)',   'YB(PPM)',   'EU(PPM)',    'TH(PPM)',    'AL2O3(WT%)','HF(PPM)',
    'ZR(PPM)',   'TIO2(WT%)', 'MNO(WT%)',  'V(PPM)',     'NB(PPM)',   'CR(PPM)',
    'TA(PPM)',   'FEOT(WT%)', 'CO(PPM)',   'NI(PPM)',    'SIO2(WT%)', 'P2O5(WT%)',
]

# ── v2: 矩阵分支 — 地化亲缘分组（6×6）─────────────────────────
_COLUMNS_IMG_V2 = [
    'RB(PPM)',   'BA(PPM)',    'TH(PPM)',    'SR(PPM)',    'K2O(WT%)',   'NA2O(WT%)',
    'LA(PPM)',   'CE(PPM)',    'NB(PPM)',    'TA(PPM)',    'PR(PPM)',    'ND(PPM)',
    'SM(PPM)',   'EU(PPM)',    'ZR(PPM)',    'HF(PPM)',    'GD(PPM)',    'TB(PPM)',
    'DY(PPM)',   'HO(PPM)',    'ER(PPM)',    'YB(PPM)',    'LU(PPM)',    'Y(PPM)',
    'SIO2(WT%)', 'AL2O3(WT%)', 'FEOT(WT%)', 'MGO(WT%)',   'CAO(WT%)',   'TIO2(WT%)',
    'CR(PPM)',   'NI(PPM)',    'CO(PPM)',    'V(PPM)',     'MNO(WT%)',   'P2O5(WT%)',
]

# ── v2: 序列分支 — 不相容性从高到低 ────────────────────────────
_COLUMNS_SEQ_V2 = [
    'RB(PPM)',   'BA(PPM)',   'TH(PPM)',   'K2O(WT%)',  'NA2O(WT%)',
    'NB(PPM)',   'TA(PPM)',   'LA(PPM)',   'CE(PPM)',
    'PR(PPM)',   'SR(PPM)',   'P2O5(WT%)',
    'ND(PPM)',   'SM(PPM)',   'ZR(PPM)',   'HF(PPM)',   'EU(PPM)',
    'TIO2(WT%)', 'AL2O3(WT%)','GD(PPM)',  'TB(PPM)',   'DY(PPM)',
    'HO(PPM)',   'Y(PPM)',    'ER(PPM)',   'YB(PPM)',   'LU(PPM)',
    'CAO(WT%)',  'V(PPM)',    'MNO(WT%)',
    'FEOT(WT%)', 'MGO(WT%)', 'SIO2(WT%)', 'CR(PPM)',   'NI(PPM)',   'CO(PPM)',
]

# 根据方案选择实际列
if COLUMN_ORDER_SCHEME == 'v2':
    COLUMNS_IMG_ORDER       = _COLUMNS_IMG_V2
    COLUMNS_ELECTRODE_ORDER = _COLUMNS_SEQ_V2
else:
    COLUMNS_IMG_ORDER       = _COLUMNS_IMG_V1
    COLUMNS_ELECTRODE_ORDER = _COLUMNS_SEQ_V1

# 36 个特征的标准顺序（用于绘图展示，以图像分支顺序为准）
ALL_FEATURES_COLS = COLUMNS_IMG_ORDER

# 原始列名 → 显示名称映射
COL_DISPLAY = {
    'RB(PPM)':'Rb',    'K2O(WT%)':'K',     'BA(PPM)':'Ba',    'SR(PPM)':'Sr',
    'CAO(WT%)':'Ca',   'NA2O(WT%)':'Na',   'LA(PPM)':'La',    'Y(PPM)':'Y',
    'MGO(WT%)':'Mg',   'PR(PPM)':'Pr',     'CE(PPM)':'Ce',    'ER(PPM)':'Er',
    'HO(PPM)':'Ho',    'ND(PPM)':'Nd',     'SM(PPM)':'Sm',    'DY(PPM)':'Dy',
    'LU(PPM)':'Lu',    'TB(PPM)':'Tb',     'GD(PPM)':'Gd',    'YB(PPM)':'Yb',
    'EU(PPM)':'Eu',    'TH(PPM)':'Th',     'AL2O3(WT%)':'Al', 'HF(PPM)':'Hf',
    'ZR(PPM)':'Zr',    'TIO2(WT%)':'Ti',   'MNO(WT%)':'Mn',   'V(PPM)':'V',
    'NB(PPM)':'Nb',    'CR(PPM)':'Cr',     'TA(PPM)':'Ta',    'FEOT(WT%)':'Fe',
    'CO(PPM)':'Co',    'NI(PPM)':'Ni',     'SIO2(WT%)':'Si',  'P2O5(WT%)':'P',
}

# 构造环境全名 → 缩写
LABEL_MAPPING = {
    'Continental arc':           'CA',
    'Island arc':                'IA',
    'Intra-oceanic arc':         'IOA',
    'BACK-ARC BASIN':            'BAB',
    'BACK-ARC_BASIN':            'BAB',
    'Mid-Oceanic Ridge':         'MOR',
    'SPREADING CENTER':          'MOR',
    'SPREADING_CENTER':          'MOR',
    'OCEANIC PLATEAU':           'OP',
    'OCEAN ISLAND':              'OI',
    'CONTINENTAL FLOOD BASALT':  'CF',
    'CONTINENTAL_RIFT':          'CR',
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# ══════════════════════════════════════════════════════════════
# ③  模型结构定义（与训练代码完全一致，勿改动）
# ══════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim, num_patches):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim) * 0.02)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.ffn(x))
        return x


class ViT_Transformer_DualStream(nn.Module):
    """
    GeoDAN 主模型 v4：ViT + Seq Transformer 双流（无 CNN）。
      矩阵分支: 6×6 周期表矩阵 → Patch Embed → CLS+GAP → ViT Encoder
      序列分支: 36 电极电势序列 → Linear Embed → CLS+PE → Transformer Encoder
      融合:    vit_cls + vit_gap + seq_cls + seq_gap → MLP 分类头
    """
    def __init__(self, num_classes, input_size=6, patch_size=2,
                 embed_dim=96, num_heads=8, transformer_layers=2,
                 ff_dim=192, dropout=0.1):
        super().__init__()
        self.num_patches = (input_size // patch_size) ** 2   # 9 patches
        self.seq_len     = input_size * input_size           # 36 tokens
        self.embed_dim   = embed_dim

        # ── 矩阵分支: ViT ──
        self.patch_embed = PatchEmbedding(1, patch_size, embed_dim, self.num_patches)
        self.vit_cls     = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.vit_cls_pos = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.vit_blocks  = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.vit_norm    = nn.LayerNorm(embed_dim)

        # ── 序列分支: Transformer ──
        self.seq_proj       = nn.Linear(1, embed_dim)
        self.seq_norm       = nn.LayerNorm(embed_dim)
        self.seq_cls        = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.seq_pos_embed  = nn.Parameter(torch.randn(1, self.seq_len + 1, embed_dim) * 0.02)
        self.seq_blocks     = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(transformer_layers)
        ])
        self.seq_final_norm = nn.LayerNorm(embed_dim)

        # ── 融合分类头: 4 路特征 (vit_cls + vit_gap + seq_cls + seq_gap) ──
        head_in = embed_dim * 4
        self.fusion = nn.Sequential(
            nn.Linear(head_in, 192),
            nn.LayerNorm(192), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(192, 96),
            nn.LayerNorm(96),  nn.GELU(), nn.Dropout(dropout),
            nn.Linear(96, num_classes),
        )
        nn.init.trunc_normal_(self.vit_cls,     std=0.02)
        nn.init.trunc_normal_(self.seq_cls,     std=0.02)
        nn.init.trunc_normal_(self.vit_cls_pos, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, x_seq):
        B = x.size(0)

        # ── 矩阵分支前向 ──
        vit_tokens = self.patch_embed(x)                         # (B, 9, D)
        vit_cls    = self.vit_cls.expand(B, -1, -1) + self.vit_cls_pos
        vit_tokens = torch.cat([vit_cls, vit_tokens], dim=1)     # (B, 10, D)
        for blk in self.vit_blocks:
            vit_tokens = blk(vit_tokens)
        vit_tokens  = self.vit_norm(vit_tokens)
        vit_cls_out = vit_tokens[:, 0]
        vit_gap_out = vit_tokens[:, 1:].mean(dim=1)

        # ── 序列分支前向 ──
        seq_tokens = self.seq_norm(self.seq_proj(x_seq))         # (B, 36, D)
        seq_cls    = self.seq_cls.expand(B, -1, -1)
        seq_tokens = torch.cat([seq_cls, seq_tokens], dim=1)     # (B, 37, D)
        seq_tokens = seq_tokens + self.seq_pos_embed
        for blk in self.seq_blocks:
            seq_tokens = blk(seq_tokens)
        seq_tokens   = self.seq_final_norm(seq_tokens)
        seq_cls_out  = seq_tokens[:, 0]
        seq_gap_out  = seq_tokens[:, 1:].mean(dim=1)

        # ── 四路融合 ──
        fused = torch.cat([vit_cls_out, vit_gap_out,
                           seq_cls_out, seq_gap_out], dim=1)
        return self.fusion(fused)


# ══════════════════════════════════════════════════════════════
# ④  SHAP 专用包装器：用 36 个原始元素生成模型需要的双输入
# ══════════════════════════════════════════════════════════════

class TableInputWrapper(nn.Module):
    """
    SHAP 只接收一个 (B, 36) 的表格输入，每一列对应一个原始地化变量。
    wrapper 内部再按训练时的两种顺序重排成 x_img 和 x_seq。
    """
    def __init__(self, model, canonical_cols, img_cols, seq_cols):
        super().__init__()
        self.model = model
        self.img_idx = [canonical_cols.index(c) for c in img_cols]
        self.seq_idx = [canonical_cols.index(c) for c in seq_cols]

    def forward(self, x_36):
        img_flat = x_36[:, self.img_idx]       # (B, 36)
        seq_flat = x_36[:, self.seq_idx]       # (B, 36)
        x_img  = img_flat.reshape(-1, 1, 6, 6)   # (B, 1, 6, 6)
        x_seq  = seq_flat.reshape(-1, 36, 1)     # (B, 36, 1)
        return self.model(x_img, x_seq)


# ══════════════════════════════════════════════════════════════
# ⑤  数据加载
# ══════════════════════════════════════════════════════════════

def load_data(train_path, test_path):
    """读取预分割的训练集与测试集，返回 36 个原始地化变量的表格输入"""
    for enc in ('utf-8', 'ISO-8859-1'):
        try:
            df_train = pd.read_csv(train_path, encoding=enc)
            df_test  = pd.read_csv(test_path,  encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    # SHAP 只看到一个 36 维表格输入；两路模型输入在 wrapper 内部生成。
    X_train_36 = df_train[ALL_FEATURES_COLS].values.astype(np.float32) / 255.0
    X_test_36  = df_test [ALL_FEATURES_COLS].values.astype(np.float32) / 255.0

    # 标签编码（以训练集顺序为准）
    y_train_raw, unique = pd.factorize(df_train['TECTONIC SETTING'])
    label2idx = {lbl: idx for idx, lbl in enumerate(unique)}
    y_test_raw = df_test['TECTONIC SETTING'].map(label2idx).values

    if np.any(pd.isna(y_test_raw)):
        unknown = set(df_test['TECTONIC SETTING'].unique()) - set(label2idx.keys())
        raise ValueError(f'测试集含未知标签: {unknown}')

    y_train = y_train_raw.astype(np.int64)
    y_test  = y_test_raw.astype(np.int64)

    print(f'\n数据加载完成')
    print(f'  训练集: {len(y_train)}  测试集: {len(y_test)}')
    print(f'  类别 ({len(unique)}): {list(unique)}')

    return X_train_36, y_train, X_test_36, y_test, unique


# ══════════════════════════════════════════════════════════════
# ⑥  分层背景数据采样
# ══════════════════════════════════════════════════════════════

def select_background(X_36, y, n_bg):
    """
    分层采样背景数据，确保每个类别均有代表。
    返回 36 维表格 tensor，形状 (n_bg, 36)。
    """
    unique_labels = np.unique(y)
    bg_idx = []
    for lbl in unique_labels:
        lbl_idx = np.where(y == lbl)[0]
        n_lbl = max(1, int(n_bg * len(lbl_idx) / len(y)))
        n_lbl = min(n_lbl, len(lbl_idx))
        bg_idx.extend(np.random.choice(lbl_idx, n_lbl, replace=False))
    bg_idx = list(set(bg_idx))[:n_bg]
    return torch.FloatTensor(X_36[bg_idx]).to(device)


# ══════════════════════════════════════════════════════════════
# ⑦  核心：分批计算 SHAP 值
# ══════════════════════════════════════════════════════════════

def _as_class_list(shap_values):
    """统一不同 SHAP 版本的输出格式：list[n_classes]，每项 shape=(n, 36)。"""
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]

    arr = np.asarray(shap_values)
    if arr.ndim == 3 and arr.shape[1] == len(ALL_FEATURES_COLS):
        # 常见新格式: (n_samples, n_features, n_outputs)
        return [arr[:, :, c] for c in range(arr.shape[2])]
    if arr.ndim == 3 and arr.shape[2] == len(ALL_FEATURES_COLS):
        # 兼容格式: (n_outputs, n_samples, n_features)
        return [arr[c, :, :] for c in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr]

    raise ValueError(f'Unexpected SHAP output shape: {arr.shape}')


def compute_shap_values(wrapped_model, background_tensor,
                        X_36, n_explain, batch_size):
    """
    使用 shap.GradientExplainer 对 36 维原始特征输入分批计算 SHAP 值。

    返回:
        shap_values : list[ndarray(n_explain, 36)]，每个类别一项
    """
    explainer = shap.GradientExplainer(wrapped_model, background_tensor)

    X_to_explain = X_36[:n_explain]
    n = len(X_to_explain)
    n_batches = int(np.ceil(n / batch_size))

    all_batches = []  # list of list[n_classes] arrays shaped (batch, 36)

    for i in tqdm(range(n_batches), desc='计算 SHAP 值'):
        s = i * batch_size
        e = min((i + 1) * batch_size, n)
        batch_tensor = torch.FloatTensor(X_to_explain[s:e]).to(device)
        sv = _as_class_list(explainer.shap_values(batch_tensor))
        all_batches.append(sv)

    # 按类别拼接所有 batch
    n_classes = len(all_batches[0])
    shap_values = [
        np.concatenate([batch[c] for batch in all_batches], axis=0)   # (n, 36)
        for c in range(n_classes)
    ]

    print(f'\nSHAP 计算完成：{n_explain} 个样本，{n_classes} 个类别')
    return shap_values


# ══════════════════════════════════════════════════════════════
# ⑧  绘图函数（SCI 出版风格）
# ══════════════════════════════════════════════════════════════

# 9 类构造环境配色（紫→蓝→青→绿→黄→橙→红，与参考图一致）
_RAINBOW_ANCHORS = [
    '#6A0DAD', '#1E3FBF', '#1E8FD8', '#21C1CB', '#74D89C',
    '#BFE36A', '#F7B24A', '#F26B3A', '#D62018',
]


def _get_class_colors(n_classes):
    """类别数 = 9 时直接用参考色；其它数量在该色带上等距取样。"""
    if n_classes == len(_RAINBOW_ANCHORS):
        return _RAINBOW_ANCHORS
    cmap = LinearSegmentedColormap.from_list('cust_rainbow', _RAINBOW_ANCHORS, N=256)
    return [cmap(i / max(n_classes - 1, 1)) for i in range(n_classes)]


def _set_pub_rcparams():
    """SCI 论文常用排版参数。系统装了 Arial 可改 'font.family' 为 'Arial'。"""
    rcParams.update({
        'font.family':       'DejaVu Sans',
        'font.size':         11,
        'axes.linewidth':    0.9,
        'axes.edgecolor':    '#222222',
        'xtick.major.width': 0.9,
        'ytick.major.width': 0.9,
        'xtick.major.size':  3.5,
        'ytick.major.size':  3.5,
        'xtick.color':       '#222222',
        'ytick.color':       '#222222',
        'axes.labelcolor':   '#111111',
        'axes.titlecolor':   '#111111',
        'ps.fonttype':       42,
    })


# --------------------------------------------------------------
# ⑧-A  总览：竖直堆叠柱状图
# --------------------------------------------------------------
def plot_stacked_bar(merged_shap, unique_labels, output_dir,
                     model_name='GeoDAN',
                     panel_label=None):
    """
    SCI 风格竖直堆叠柱状图（参考 EMSPN SHAP Summary 样式）。
        x 轴 = 元素（按总重要性降序）
        y 轴 = mean(|SHAP value|)
        每根柱按构造环境分段堆叠
    """
    n_classes = merged_shap.shape[0]
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]

    # 每类 / 每特征平均 |SHAP|
    setting_imp = np.abs(merged_shap).mean(axis=1)        # (C, F)
    total_imp   = setting_imp.sum(axis=0)                  # (F,)

    # 按总重要性降序：最重要在最左
    sort_idx = np.argsort(total_imp)[::-1]
    sorted_names = [display_names[i] for i in sort_idx]
    sorted_imp   = setting_imp[:, sort_idx]                # (C, F)

    short_labels = [LABEL_MAPPING.get(lbl, lbl) for lbl in unique_labels]
    colors = _get_class_colors(n_classes)

    _set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(14, 6), dpi=130)
    fig.patch.set_facecolor('white')

    x = np.arange(len(sort_idx))
    bottom = np.zeros(len(sort_idx))
    for c in range(n_classes):
        ax.bar(x, sorted_imp[c], bottom=bottom,
               width=0.78,
               color=colors[c],
               edgecolor='white', linewidth=0.35,
               label=short_labels[c],
               zorder=3)
        bottom += sorted_imp[c]

    # 轴 & 标题
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_names, fontsize=11)
    ax.set_xlim(-0.7, len(sort_idx) - 0.3)
    ax.set_xlabel('Elements', fontsize=13, labelpad=6)

    ax.set_ylim(0, bottom.max() * 1.04)
    ax.set_ylabel('mean(|SHAP value|)', fontsize=13, labelpad=6)

    ax.set_title(f'{model_name}  SHAP Summary', fontsize=13, pad=10)

    if panel_label:
        fig.text(0.012, 0.965, panel_label,
                 fontsize=15, fontweight='bold', va='top')

    # 网格 / 边框
    ax.grid(axis='y', linestyle='--', linewidth=0.45,
            color='#bbbbbb', alpha=0.65, zorder=0)
    ax.set_axisbelow(True)
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_linewidth(0.9)
        ax.spines[s].set_color('#222222')
    ax.tick_params(axis='both', which='major', labelsize=10)

    # 图例
    legend = ax.legend(
        title='Tectonic Settings',
        loc='upper right',
        bbox_to_anchor=(0.998, 0.985),
        fontsize=10, title_fontsize=11,
        frameon=True, framealpha=1.0,
        edgecolor='#888888', facecolor='white',
        handlelength=1.4, handleheight=1.0,
        labelspacing=0.45, borderpad=0.6,
    )
    legend.get_frame().set_linewidth(0.6)
    legend._legend_box.align = 'left'

    plt.tight_layout()
    out_path = os.path.join(output_dir, f'shap_stacked_bar_{model_name}.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    mpl.rcParams.update(mpl.rcParamsDefault)
    print(f'已保存：{out_path}')


# --------------------------------------------------------------
# ⑧-B  自定义紧凑 violin / beeswarm 引擎（per-class & overall 共用）
# --------------------------------------------------------------
def _compact_violin_beeswarm(shap_vals, feat_vals, feature_names, ax,
                             max_points=400):
    """
    在指定 ax 上画紧凑 violin + beeswarm。
        shap_vals : (n_samples, n_features)
        feat_vals : (n_samples, n_features)
    最重要的特征排在顶部。
    返回 cmap（外部据此画 colorbar）。
    """
    n_samp, n_feat = shap_vals.shape

    # 重要性升序 → 顶部最大
    importance = np.abs(shap_vals).mean(axis=0)
    order_for_plot = np.argsort(importance)
    sorted_names = [feature_names[i] for i in order_for_plot]

    cmap = plt.cm.coolwarm

    if n_samp > max_points:
        idx_sub = np.random.choice(n_samp, max_points, replace=False)
    else:
        idx_sub = np.arange(n_samp)

    for y_pos, feat_idx in enumerate(order_for_plot):
        sv = shap_vals[idx_sub, feat_idx]
        fv = feat_vals[idx_sub, feat_idx]

        # 特征值标准化到 [0,1] 用于上色
        fv_min, fv_max = np.nanmin(fv), np.nanmax(fv)
        if fv_max > fv_min:
            fv_norm = (fv - fv_min) / (fv_max - fv_min)
        else:
            fv_norm = np.full_like(fv, 0.5)

        # ---- violin 轮廓（KDE） ----
        try:
            if np.std(sv) > 1e-9:
                kde = gaussian_kde(sv, bw_method=0.35)
                xs = np.linspace(sv.min(), sv.max(), 100)
                d = kde(xs)
                if d.max() > 0:
                    d = d / d.max() * 0.42
                ax.fill_between(xs, y_pos - d, y_pos + d,
                                color='#dddddd', alpha=0.55,
                                linewidth=0, zorder=2)
                ax.plot(xs, y_pos - d, color='#999999', linewidth=0.5, zorder=2.1)
                ax.plot(xs, y_pos + d, color='#999999', linewidth=0.5, zorder=2.1)
        except Exception:
            pass

        # ---- beeswarm：密度感知 jitter ----
        try:
            kde_local = gaussian_kde(sv, bw_method=0.35)
            d2 = kde_local(sv)
            d2 = d2 / d2.max() * 0.40 if d2.max() > 0 else np.full_like(sv, 0.05)
        except Exception:
            d2 = np.full_like(sv, 0.15)
        rand_signs = np.random.uniform(-1, 1, size=len(sv))
        y_off = rand_signs * d2

        ax.scatter(sv, y_pos + y_off,
                   c=cmap(fv_norm), s=8, alpha=0.78,
                   edgecolors='none', zorder=3)

    ax.set_yticks(range(len(order_for_plot)))
    ax.set_yticklabels(sorted_names, fontsize=9)
    ax.set_ylim(-0.7, len(order_for_plot) - 0.3)

    # x 范围留白
    vmin, vmax = shap_vals.min(), shap_vals.max()
    pad = (vmax - vmin) * 0.04 if vmax > vmin else 0.01
    ax.set_xlim(vmin - pad, vmax + pad)
    ax.set_xlabel('SHAP value', fontsize=11, labelpad=5)

    # 0 参考线 / 网格 / 边框
    ax.axvline(0, color='#666666', linewidth=0.7, zorder=1.5)
    ax.grid(axis='x', linestyle='--', linewidth=0.4,
            color='#bbbbbb', alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)
    for s in ['left', 'bottom']:
        ax.spines[s].set_linewidth(0.8)

    return cmap


def _attach_feature_value_colorbar(fig, cmap):
    """右侧加一条 Low → High 色条。"""
    sm = ScalarMappable(norm=Normalize(0, 1), cmap=cmap)
    sm.set_array([])
    cax = fig.add_axes([0.94, 0.18, 0.018, 0.65])
    cb = fig.colorbar(sm, cax=cax)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(['Low', 'High'])
    cb.ax.tick_params(labelsize=9, length=0)
    cb.outline.set_linewidth(0.6)
    cax.set_ylabel('Feature value', fontsize=10,
                   labelpad=6, rotation=270, va='bottom')


# --------------------------------------------------------------
# ⑧-C  Per-class beeswarm
# --------------------------------------------------------------
def plot_per_class_beeswarm(merged_shap, X_explain_img, unique_labels,
                            output_dir,
                            model_name='GeoDAN'):
    """
    每个构造环境一张紧凑 violin + beeswarm。
    高度固定 ~8.2 in，36 特征也不会拉得很长。
    """
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]
    feat_arr = np.asarray(X_explain_img)

    for c, lbl in enumerate(unique_labels):
        short = LABEL_MAPPING.get(lbl, lbl)
        safe  = short.replace(' ', '_').replace('/', '_')

        _set_pub_rcparams()
        fig, ax = plt.subplots(figsize=(7.2, 8.2), dpi=130)
        fig.patch.set_facecolor('white')

        cmap = _compact_violin_beeswarm(
            merged_shap[c], feat_arr, display_names, ax)

        ax.set_title(f'SHAP — {short}', fontsize=12, pad=8)
        _attach_feature_value_colorbar(fig, cmap)

        plt.subplots_adjust(left=0.10, right=0.91, top=0.94, bottom=0.07)

        out_path = os.path.join(output_dir,
                                f'shap_beeswarm_{safe}_{model_name}.png')
        plt.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
        plt.close()
        mpl.rcParams.update(mpl.rcParamsDefault)
        print(f'  已保存：{out_path}')


# --------------------------------------------------------------
# ⑧-D  整体 beeswarm（所有类别取均值）
# --------------------------------------------------------------
def plot_overall_beeswarm(merged_shap, X_explain_img, output_dir,
                          model_name='GeoDAN'):
    """
    所有类别取均值后的整体紧凑 violin + beeswarm。
    """
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]
    feat_arr  = np.asarray(X_explain_img)
    mean_shap = merged_shap.mean(axis=0)   # (n_samples, n_features)

    _set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(7.2, 8.2), dpi=130)
    fig.patch.set_facecolor('white')

    cmap = _compact_violin_beeswarm(
        mean_shap, feat_arr, display_names, ax)

    ax.set_title('SHAP — Overall (Mean of all classes)', fontsize=12, pad=8)
    _attach_feature_value_colorbar(fig, cmap)

    plt.subplots_adjust(left=0.10, right=0.91, top=0.94, bottom=0.07)
    out_path = os.path.join(output_dir,
                            f'shap_beeswarm_overall_{model_name}.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    mpl.rcParams.update(mpl.rcParamsDefault)
    print(f'已保存：{out_path}')


# --------------------------------------------------------------
# ⑧-E  热力图（保留）
# --------------------------------------------------------------
def plot_heatmap(merged_shap, unique_labels, output_dir,
                 model_name='GeoDAN'):
    """
    热力图：行 = 构造环境，列 = 地化元素，
    单元格颜色 = 该类的平均绝对 SHAP 值（越深越重要）。
    """
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]
    short_labels  = [LABEL_MAPPING.get(lbl, lbl) for lbl in unique_labels]

    # (n_classes, 36) 平均绝对 SHAP
    heatmap_data = np.abs(merged_shap).mean(axis=1)

    # 按列（特征）总重要性排序
    col_order = np.argsort(heatmap_data.sum(axis=0))[::-1]
    heatmap_sorted  = heatmap_data[:, col_order]
    feat_sorted = [display_names[i] for i in col_order]

    _set_pub_rcparams()
    fig, ax = plt.subplots(figsize=(18, 5), dpi=130)
    im = ax.imshow(heatmap_sorted, aspect='auto', cmap='YlOrRd')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label('mean(|SHAP value|)', fontsize=11, labelpad=6)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_linewidth(0.6)

    ax.set_xticks(range(len(feat_sorted)))
    ax.set_xticklabels(feat_sorted, rotation=45, ha='right', fontsize=11)
    ax.set_yticks(range(len(short_labels)))
    ax.set_yticklabels(short_labels, fontsize=12)
    ax.set_title(f'{model_name} — SHAP Heatmap by Tectonic Setting',
                 fontsize=13, pad=8)
    for s in ['top', 'right', 'left', 'bottom']:
        ax.spines[s].set_linewidth(0.8)
    plt.tight_layout()

    out_path = os.path.join(output_dir, f'shap_heatmap_{model_name}.png')
    plt.savefig(out_path, dpi=400, bbox_inches='tight', facecolor='white')
    plt.close()
    mpl.rcParams.update(mpl.rcParamsDefault)
    print(f'已保存：{out_path}')


# --------------------------------------------------------------
# ⑧-F  保存 CSV
# --------------------------------------------------------------
def save_shap_csv(merged_shap, unique_labels, output_dir,
                  model_name='GeoDAN'):
    """将每个类别的平均绝对 SHAP 值保存为 CSV，方便后续分析。"""
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]
    rows = []
    for c, lbl in enumerate(unique_labels):
        row = {'tectonic_setting': lbl,
               'short_name': LABEL_MAPPING.get(lbl, lbl)}
        for f_idx, fname in enumerate(display_names):
            row[fname] = float(np.abs(merged_shap[c, :, f_idx]).mean())
        rows.append(row)
    df_out = pd.DataFrame(rows)
    out_path = os.path.join(output_dir, f'shap_mean_abs_{model_name}.csv')
    df_out.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'已保存：{out_path}')
    return df_out


# ══════════════════════════════════════════════════════════════
# ⑨  主程序
# ══════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- 1. 加载数据 ----------
    X_train_36, y_train, X_test_36, y_test, unique_labels = load_data(TRAIN_FILE, TEST_FILE)

    num_classes = len(unique_labels)

    # ---------- 2. 加载模型 ----------
    model = ViT_Transformer_DualStream(num_classes=num_classes).to(device)

    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'\n模型权重加载成功：{MODEL_PATH}')

    # ---------- 3. 包装为 36 维表格输入 ----------
    wrapped = TableInputWrapper(
        model,
        canonical_cols=ALL_FEATURES_COLS,
        img_cols=COLUMNS_IMG_ORDER,
        seq_cols=COLUMNS_ELECTRODE_ORDER,
    ).to(device)
    wrapped.eval()

    # ---------- 4. 准备背景数据 ----------
    print(f'\n准备背景数据（分层采样 {MAX_BACKGROUND} 个样本）...')
    background_tensor = select_background(X_train_36, y_train, MAX_BACKGROUND)
    print(f'  背景数据 shape: {background_tensor.shape}')   # (n_bg, 36)

    # ---------- 5. 选择待解释的测试样本 ----------
    n_explain = min(MAX_EXPLAIN, len(y_test))
    np.random.seed(42)
    explain_idx = []
    for lbl in np.unique(y_test):
        lbl_idx = np.where(y_test == lbl)[0]
        n_lbl = max(1, int(n_explain * len(lbl_idx) / len(y_test)))
        n_lbl = min(n_lbl, len(lbl_idx))
        explain_idx.extend(np.random.choice(lbl_idx, n_lbl, replace=False))
    explain_idx = explain_idx[:n_explain]
    X_exp_36 = X_test_36[explain_idx]
    print(f'\n待解释样本：{len(explain_idx)} 个（来自测试集）')

    # ---------- 6. 计算 SHAP 值 ----------
    shap_values = compute_shap_values(
        wrapped, background_tensor,
        X_exp_36,
        n_explain=len(explain_idx),
        batch_size=SHAP_BATCH_SIZE,
    )

    # ---------- 7. 整理 SHAP ----------
    merged_shap = np.array(shap_values)   # (n_classes, n_samples, 36)

    np.save(os.path.join(OUTPUT_DIR, 'shap_merged.npy'),  merged_shap)
    print(f'  SHAP numpy 数据已保存至 {OUTPUT_DIR}')

    # ---------- 8. 绘图 ----------
    print('\n开始绘图...')

    # 8a. 堆叠条形图（总览）
    plot_stacked_bar(merged_shap, unique_labels, OUTPUT_DIR)

    # 8b. 整体 beeswarm
    plot_overall_beeswarm(merged_shap, X_exp_36, OUTPUT_DIR)

    # 8c. 每类 beeswarm（可能较慢）
    print('\n绘制各构造环境 beeswarm 图...')
    plot_per_class_beeswarm(merged_shap, X_exp_36, unique_labels, OUTPUT_DIR)

    # 8d. 热力图
    plot_heatmap(merged_shap, unique_labels, OUTPUT_DIR)

    # 8e. 保存 CSV
    df_importance = save_shap_csv(merged_shap, unique_labels, OUTPUT_DIR)

    # ---------- 9. 打印 Top-10 最重要特征 ----------
    display_names = [COL_DISPLAY.get(f, f) for f in ALL_FEATURES_COLS]
    total_importance = np.abs(merged_shap).mean(axis=(0, 1))   # (36,)
    top_idx = np.argsort(total_importance)[::-1]
    print('\n' + '=' * 45)
    print('  Top-10 最重要地化元素（综合全部构造环境）')
    print('=' * 45)
    for rank, idx in enumerate(top_idx[:10], 1):
        print(f'  {rank:2d}. {display_names[idx]:4s}  mean|SHAP| = {total_importance[idx]:.6f}')
    print('=' * 45)
    print(f'\nSHAP 分析完成！结果已保存至:\n  {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
