"""
现代训练集 vs 太古代应用集的 PCA 分布一致性分析。

数据流：
- 现代玄武岩：直接使用原始 CSV。
- 太古代玄武岩：直接使用原始 CSV，不做 MissForest 插值，也不读取插值缓存。
- PCA 轴只用现代训练集拟合；太古代样品只投影到固定 PCA 空间。
"""

from __future__ import annotations

# === ?????????? config/paths.py????????????===
import sys as _cfg_sys
from pathlib import Path as _cfg_Path
_cfg_sys.path.insert(0, str(_cfg_Path(__file__).resolve().parents[1]))
from config.paths import (
    ARCHEAN_DIR, TRAIN_NORM_CSV, TEST_NORM_CSV,
    MAIN_MODEL_WEIGHT, QUANTILE_PARAMS_JSON, COMBINED_CSV,
)

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


# =========================
# 路径
# =========================
RAW_MODERN_PATH = COMBINED_CSV
ARCHEAN_RAW_PATH = (ARCHEAN_DIR / "data/archean_basalt.csv")
OUT_ROOT = (ARCHEAN_DIR / "outputs/distribution_consistency")
OUT_NO_IMPUTE = (ARCHEAN_DIR / "outputs/distribution_consistency/pca_no_impute")

# 中文注释：新的 applicability-domain 三联图直接输出到 distribution_consistency 根目录。
AD_FIG_PNG = OUT_ROOT / "pca_applicability_domain.png"
AD_COMPAT_PNG = OUT_ROOT / "pca_v1_overall.png"          # 中文注释：兼容旧文件名，输出同一张三联图。
AD_COVERAGE_CSV = OUT_ROOT / "pca_applicability_coverage.csv"

# 中文注释：保留旧的逐类版 PCA 及统计文件（仍写到 pca_no_impute 子目录，不影响新图）。
PCA_OVERALL_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/pca_no_impute/pca_v1_overall.png")
PCA_PER_CLASS_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/pca_no_impute/pca_v2_per_class.png")
PCA_OVERLAP_SUMMARY_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/pca_no_impute/pca_overlap_summary.csv")
PCA_COORDS_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/pca_no_impute/pca_coords.npz")
PCA_LOADINGS_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/pca_no_impute/pca_loadings.csv")


# =========================
# 元素 / 标签 / 常量
# =========================
MAJORS = ["SIO2", "TIO2", "AL2O3", "FEOT", "MNO", "MGO", "CAO", "NA2O", "K2O", "P2O5"]
TRACES = ["RB", "V", "CR", "CO", "NI", "BA", "SR", "Y", "ZR", "NB",
          "LA", "CE", "PR", "ND", "SM", "EU", "GD", "TB", "DY", "HO",
          "ER", "YB", "LU", "HF", "TA", "TH"]
ALL_ELEMENTS = MAJORS + TRACES

TECTONIC_COL = "TECTONIC SETTING"
CONFIDENCE_COL = "Arc_probability3"
HIGH_CONF_THRESHOLD = 0.60   # 中文注释：高弧亲和性阈值 P_arc >= 0.60。
LOW_CONF_THRESHOLD = 0.30    # 中文注释：低弧亲和性阈值 P_arc < 0.30。
RANDOM_SEED = 42
CHI2_95_DF2 = 5.991464547107979
CHI2_99_DF2 = 9.210340371976294
ELLIPSE_95_SCALE = np.sqrt(CHI2_95_DF2)
ELLIPSE_99_SCALE = np.sqrt(CHI2_99_DF2)

# 中文注释：applicability-domain kNN 距离的参数。
AD_VAR_TARGET = 0.85   # 累计解释方差阈值，决定保留多少个 PC（约 12 个 -> 85% 方差）。
AD_K = 10              # kNN 的 k 值。

# 中文注释：(a) 子图 envelope 椭圆的视觉收紧系数（<1 让圆环更小、更贴主云团）。
ENV_VIS_SCALE = 0.75

# =========================
# 太古代数据预处理：完整度 + 玄武岩主量筛选
# =========================
ARCH_MAX_MISSING = 16            # 36 个元素里缺失（NaN 或 <=0）多于 16 个的样品剔除。
ARCH_SIO2_RANGE = (45.0, 53.0)   # 玄武岩主量窗口（WT%）。
ARCH_MGO_RANGE = (4.5, 12.0)
ARCH_AL2O3_RANGE = (12.0, 19.0)

# =========================
# applicability-domain 三联图配色（低饱和，期刊主文风格）
# =========================
# (a) PCA reference space
C_MODERN_FILL = "#E3E7EA"   # 现代参考背景极浅灰
C_ARCHEAN = "#5F6F7A"       # 太古代蓝灰
C_HIGH_ARC = "#C85A17"      # 高弧亲和性低饱和橙
C_ENV_95 = "#5E666D"        # 95% envelope 深灰
C_ENV_99 = "#747D84"        # 99% envelope 稍浅灰
# (b) applicability-domain distance
C_MODERN_OUTLINE = "#4C555C"  # 现代参考距离 step 轮廓
C_IN_DOMAIN = "#6F8EA3"     # ≤95% 蓝灰
C_MARGINAL = "#E7B36A"      # 95–99% 柔和浅橙
C_OUT_DOMAIN = "#D98686"    # >99% 柔和玫瑰红
C_THRESH = "#30343A"        # 阈值竖线
# (c) quantitative coverage
C_CONNECT = "#B8C0C7"       # dumbbell 连线
C_COV95 = "#31566B"         # ≤95% 覆盖率深蓝灰圆点
C_COV99 = "#A9BFCC"         # ≤99% 覆盖率浅蓝灰方块
C_COV99_EDGE = "#5F7888"    # ≤99% 方块描边
C_NLABEL = "#6F7780"        # 样品数标签灰
# 通用
C_TEXT = "#222222"          # 图内注释主色
C_SPINE = "#9AA0A5"         # 浅深灰边框
C_GRID = "#E9ECEF"          # 极浅网格线

CLASS_ABBREVS = {
    "CONTINENTAL ARC": "CA", "INTRA-OCEANIC ARC": "IOA", "ISLAND ARC": "IA",
    "BACK-ARC_BASIN": "BAB", "SPREADING_CENTER": "MOR", "OCEANIC PLATEAU": "OP",
    "OCEAN ISLAND": "OI", "CONTINENTAL FLOOD BASALT": "CFB", "CONTINENTAL_RIFT": "CR",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.linewidth": 0.9, "axes.labelsize": 12, "axes.titlesize": 12,
    "axes.edgecolor": "#444A4F",
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
    "xtick.color": "#444A4F", "ytick.color": "#444A4F",
    "figure.dpi": 100, "savefig.dpi": 600, "savefig.bbox": "tight",
})


# =========================
# 通用工具
# =========================
def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """去掉列名里的 (WT%) / (PPM) 后缀。"""
    return df.rename(columns={c: c.replace("(WT%)", "").replace("(PPM)", "").strip() for c in df.columns})


def abbr(cls: str) -> str:
    return CLASS_ABBREVS.get(cls, cls)


def extract_elements(df: pd.DataFrame) -> pd.DataFrame:
    """从 df 抽出 36 个元素列，缺失列填 NaN。"""
    out = pd.DataFrame(index=df.index)
    for col in ALL_ELEMENTS:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    return out[ALL_ELEMENTS]


def preprocess_archean(df: pd.DataFrame) -> pd.DataFrame:
    """太古代玄武岩预处理：① 剔除高缺失样品；② 按主量元素范围做玄武岩筛选。

    - 缺失计数：36 个元素里 NaN 或 <=0 的个数（与 to_pca_space 对缺失的处理一致）。
    - 主量窗口：SiO2 / MgO / Al2O3 同时落在给定范围（含边界）。
    返回过滤并 reset_index 后的 df，保留原始列，供后续抽取置信度与元素特征（行索引保持同步）。
    """
    n0 = len(df)
    feats = extract_elements(df)
    miss = (feats.isna() | (feats <= 0)).sum(axis=1)
    df1 = df[miss <= ARCH_MAX_MISSING]
    n1 = len(df1)

    sio2 = pd.to_numeric(df1["SIO2"], errors="coerce")
    mgo = pd.to_numeric(df1["MGO"], errors="coerce")
    al2o3 = pd.to_numeric(df1["AL2O3"], errors="coerce")
    keep_major = (sio2.between(*ARCH_SIO2_RANGE)
                  & mgo.between(*ARCH_MGO_RANGE)
                  & al2o3.between(*ARCH_AL2O3_RANGE))
    df2 = df1[keep_major].reset_index(drop=True)
    n2 = len(df2)

    print("  Archean preprocessing:")
    print(f"    missing <= {ARCH_MAX_MISSING} / {len(ALL_ELEMENTS)}      : "
          f"{n1:,} / {n0:,}  (removed {n0 - n1:,})")
    print(f"    SiO2{ARCH_SIO2_RANGE} MgO{ARCH_MGO_RANGE} Al2O3{ARCH_AL2O3_RANGE}: "
          f"{n2:,} / {n1:,}  (removed {n1 - n2:,})")
    print(f"    kept {n2:,} / {n0:,} archean samples")
    return df2


# =========================
# PCA 输入构造
# =========================
def to_pca_space(train: pd.DataFrame, arch: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """把原始现代/太古代特征转到 log10 + 现代训练集标准化空间，不做插值。"""
    # 中文注释：0 和负值视为缺失；用现代训练集 nanmean/nanstd 标准化。
    # 中文注释：缺失值填到标准化空间的 0，也就是现代训练集均值位置。
    tr, ar = train.copy(), arch.copy()
    tr[tr <= 0], ar[ar <= 0] = np.nan, np.nan
    tlog, alog = np.log10(tr.to_numpy(dtype=float)), np.log10(ar.to_numpy(dtype=float))
    mean = np.nanmean(tlog, axis=0)
    std = np.nanstd(tlog, axis=0)
    std[~np.isfinite(std) | (std == 0)] = 1.0
    return (np.nan_to_num((tlog - mean) / std, nan=0.0),
            np.nan_to_num((alog - mean) / std, nan=0.0))


# =========================
# applicability-domain：多维 PCA 子空间里的 kNN 距离
# =========================
def compute_ad_distances(PC_train_full: np.ndarray, PC_arch_full: np.ndarray,
                         var_ratio: np.ndarray, var_target: float = AD_VAR_TARGET,
                         k: int = AD_K) -> dict:
    """在累计解释 >= var_target 方差的前若干个 PC 子空间里，计算每个样品到现代训练集的 kNN 平均距离。

    - 子空间维度 n_pc 由累计解释方差决定（约 12 个 PC -> 85% 方差）。
    - 各 PC 用现代训练集的标准差白化，使每个保留维度等权（等价于子空间内的马氏距离）。
    - 现代训练集用 leave-one-out（排除自身）得到自身 kNN 距离分布，据此定义 95% / 99% 阈值。
    """
    cum = np.cumsum(var_ratio)
    n_pc = int(np.searchsorted(cum, var_target) + 1)
    n_pc = max(2, min(n_pc, PC_train_full.shape[1]))

    Wt = PC_train_full[:, :n_pc]
    sd = Wt.std(axis=0)
    sd[sd == 0] = 1.0
    Wt = Wt / sd
    Wa = PC_arch_full[:, :n_pc] / sd

    nn = NearestNeighbors(n_neighbors=k + 1).fit(Wt)
    d_tr, _ = nn.kneighbors(Wt)
    dist_modern = d_tr[:, 1:].mean(axis=1)              # 中文注释：丢掉第 0 列（自身距离 0）。
    d_ar, _ = nn.kneighbors(Wa, n_neighbors=k)
    dist_arch = d_ar.mean(axis=1)

    q95 = float(np.quantile(dist_modern, 0.95))
    q99 = float(np.quantile(dist_modern, 0.99))
    return {
        "n_pc": n_pc, "cum_var": float(cum[n_pc - 1]), "k": k,
        "dist_modern": dist_modern, "dist_arch": dist_arch,
        "q95": q95, "q99": q99,
    }


def coverage_rows(dist_arch: np.ndarray, q95: float, q99: float,
                  high_mask: np.ndarray | None, low_mask: np.ndarray | None) -> list[dict]:
    """三组太古代样品（全部 / 高弧亲和 / 低弧亲和）在现代 95% / 99% 域内的覆盖率。"""
    groups = [("All Archean basalts", np.ones(len(dist_arch), dtype=bool))]
    if high_mask is not None:
        groups.append((f"High arc-affinity\n(P$_{{arc}}$ ≥ {HIGH_CONF_THRESHOLD:.2f})", high_mask))
    if low_mask is not None:
        groups.append((f"Low arc-affinity\n(P$_{{arc}}$ < {LOW_CONF_THRESHOLD:.2f})", low_mask))
    rows = []
    for label, m in groups:
        d = dist_arch[m]
        rows.append({
            "group": label, "n": int(m.sum()),
            "cov95": round(100 * float(np.mean(d <= q95)), 1),
            "cov99": round(100 * float(np.mean(d <= q99)), 1),
        })
    return rows


# =========================
# 绘图
# =========================
def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    """二维点云的置信椭圆。"""
    if len(x) < 5:
        return None
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ellipse = Ellipse((0, 0), width=np.sqrt(1 + pearson) * 2,
                      height=np.sqrt(1 - pearson) * 2, **kwargs)
    transf = (mpl.transforms.Affine2D().rotate_deg(45)
              .scale(np.sqrt(cov[0, 0]) * n_std, np.sqrt(cov[1, 1]) * n_std)
              .translate(np.mean(x), np.mean(y)))
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def _panel_tag(ax, tag: str, x_pts: float = -40, y_pts: float = 12) -> None:
    """子图左上角外侧加粗标注 (a)/(b)/(c)。

    用「相对轴左上角的固定点数偏移」定位，与面板宽高无关——这样不同高度/宽度的子图
    标号离各自顶边、左边的绝对距离一致（顶边对齐的 (a)/(b) 标号高度相同）。
    x_pts 越负越靠左，y_pts 越大越靠上。
    """
    ax.annotate(tag, xy=(0, 1), xycoords="axes fraction",
                xytext=(x_pts, y_pts), textcoords="offset points",
                ha="left", va="bottom", fontsize=13, fontweight="bold", color=C_TEXT)


def _style_axes(ax) -> None:
    """统一去掉上、右边框，左/下边框用浅深灰细线，刻度低调。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C_SPINE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(length=3, width=0.7, color=C_SPINE, labelcolor="#3A4045")


def _panel_a_map(ax, PC_train, PC_arch, high_mask, ad, var1, var2):
    """(a) Modern basalt reference space：极淡现代参考云 + 95/99 置信椭圆 envelope + 太古代叠加。"""
    xt, yt = PC_train[:, 0], PC_train[:, 1]

    # 中文注释：所有现代玄武岩合并成极淡浅灰参考云，点径小、透明度低，不压住太古代点。
    ax.scatter(xt, yt, s=6, c=C_MODERN_FILL, alpha=0.13, edgecolors="none", zorder=1)

    # 中文注释：现代训练域 95%（实线）/ 99%（长虚线）置信椭圆 envelope，低饱和细线、点在其上。
    confidence_ellipse(xt, yt, ax, n_std=ELLIPSE_99_SCALE, edgecolor=C_ENV_99,
                       facecolor="none", lw=1.0, ls=(0, (5, 4)), alpha=0.75, zorder=3)
    confidence_ellipse(xt, yt, ax, n_std=ELLIPSE_95_SCALE, edgecolor=C_ENV_95,
                       facecolor="none", lw=1.2, alpha=0.85, zorder=3)

    # 中文注释：太古代样品；高弧亲和性单独低饱和橙高亮，其余蓝灰点。
    if high_mask is not None:
        ax.scatter(PC_arch[~high_mask, 0], PC_arch[~high_mask, 1], s=18, c=C_ARCHEAN,
                   alpha=0.60, edgecolors="none", zorder=4, label="Archean")
        ax.scatter(PC_arch[high_mask, 0], PC_arch[high_mask, 1], s=26, c=C_HIGH_ARC,
                   alpha=0.80, edgecolors="white", linewidths=0.4, zorder=6,
                   label="High arc-affinity")
    else:
        ax.scatter(PC_arch[:, 0], PC_arch[:, 1], s=18, c=C_ARCHEAN,
                   alpha=0.60, edgecolors="none", zorder=4, label="Archean")

    ax.text(0.025, 0.97, "Modern basalt reference space", transform=ax.transAxes,
            ha="left", va="top", fontsize=11.5, style="italic", color=C_TEXT)
    ax.set_xlabel(f"PC1 (modern-reference PCA; {var1 * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({var2 * 100:.1f}% variance)")

    # 中文注释：坐标范围放宽到现代点 ~4.4σ，让 95/99 envelope 留出余白、看起来更舒展、
    #          不顶到图框；同时与太古代点 99% 范围取并集，保证 99% 椭圆和太古代点都不被裁。
    mx, my = xt.mean(), yt.mean()
    sx, sy = xt.std(), yt.std()
    ax_x = np.percentile(np.abs(PC_arch[:, 0] - mx), 99) + 0.5 * sx
    ax_y = np.percentile(np.abs(PC_arch[:, 1] - my), 99) + 0.5 * sy
    rx = max(4.4 * sx, ELLIPSE_99_SCALE * sx + 0.8 * sx, ax_x)
    ry = max(4.4 * sy, ELLIPSE_99_SCALE * sy + 0.8 * sy, ax_y)
    ax.set_xlim(mx - rx, mx + rx)
    ax.set_ylim(my - ry, my + ry)
    ax.grid(True, color=C_GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_MODERN_FILL,
               markeredgecolor="#C4CACE", markersize=6, label="Modern reference"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_ARCHEAN,
               markeredgecolor="none", markersize=6, label="Archean"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_HIGH_ARC,
               markeredgecolor="white", markersize=6.5, label="High arc-affinity"),
        Line2D([0], [0], color=C_ENV_95, lw=1.1, label="95% envelope"),
        Line2D([0], [0], color=C_ENV_99, lw=1.0, ls=(0, (5, 4)), label="99% envelope"),
    ]
    if high_mask is None:
        handles = [h for h in handles if h.get_label() != "High arc-affinity"]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.95,
                    edgecolor="#D0D4D8", fontsize=8.5, handletextpad=0.5,
                    borderpad=0.6, labelspacing=0.45)
    leg.get_frame().set_linewidth(0.6)
    leg.set_zorder(20)  # 中文注释：图例置于所有散点之上，避免被高 zorder 的橙色点遮挡。
    _panel_tag(ax, "(a)")


def _panel_b_distance(ax, ad):
    """(b) Applicability-domain distance：太古代到现代域的 kNN 距离直方图 + 95/99 阈值。"""
    dist_arch, dist_modern = ad["dist_arch"], ad["dist_modern"]
    q95, q99 = ad["q95"], ad["q99"]

    xmax = max(q99 * 1.8, np.quantile(dist_arch, 0.97))
    bins = np.linspace(0, xmax, 46)

    # 中文注释：太古代直方图，按 ≤95% / 95–99% / >99% 三段着色。
    counts, edges, patches = ax.hist(dist_arch, bins=bins, density=True, zorder=3)
    centers = 0.5 * (edges[:-1] + edges[1:])
    for c, p in zip(centers, patches):
        p.set_facecolor(C_IN_DOMAIN if c <= q95 else C_MARGINAL if c <= q99 else C_OUT_DOMAIN)
        p.set_edgecolor("white")
        p.set_linewidth(0.3)
        p.set_alpha(0.85)

    # 中文注释：现代训练集自身距离分布用细深灰阶梯轮廓叠加作参照。
    ax.hist(dist_modern, bins=bins, density=True, histtype="step",
            color=C_MODERN_OUTLINE, lw=1.0, zorder=4)

    ymax = ax.get_ylim()[1]
    ax.axvline(q95, color=C_THRESH, lw=1.05, ls="-", zorder=5)
    ax.axvline(q99, color=C_THRESH, lw=1.05, ls=(0, (4, 3)), zorder=5)
    ax.text(q95, ymax * 0.97, "95%", rotation=90, va="top", ha="right",
            fontsize=9, color=C_THRESH)
    ax.text(q99, ymax * 0.97, "99%", rotation=90, va="top", ha="right",
            fontsize=9, color=C_THRESH)

    handles = [
        Line2D([0], [0], color=C_MODERN_OUTLINE, lw=1.0, label="Modern reference"),
        mpl.patches.Patch(facecolor=C_IN_DOMAIN, edgecolor="none", label="Archean ≤95%"),
        mpl.patches.Patch(facecolor=C_MARGINAL, edgecolor="none", label="95–99%"),
        mpl.patches.Patch(facecolor=C_OUT_DOMAIN, edgecolor="none", label=">99%"),
    ]
    leg = ax.legend(handles=handles, loc="upper right", frameon=True, framealpha=0.85,
                    edgecolor="#D8DCDF", fontsize=8, handlelength=1.1,
                    handletextpad=0.5, labelspacing=0.3, borderpad=0.5)
    leg.get_frame().set_linewidth(0.5)

    ax.set_xlim(0, xmax)
    ax.set_xlabel("kNN distance to modern reference")
    ax.set_ylabel("Density")
    ax.set_title("Applicability-domain distance", fontsize=11, pad=6, color=C_TEXT)
    ax.grid(True, axis="y", color=C_GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)
    _panel_tag(ax, "(b)")


def _panel_c_coverage(ax, cov_rows):
    """(c) Quantitative coverage：三组太古代在现代 95%/99% 域内的覆盖率 dumbbell plot。"""
    n = len(cov_rows)
    ys = np.arange(n)[::-1]  # 中文注释：第一组放最上面。
    for y, r in zip(ys, cov_rows):
        ax.plot([r["cov95"], r["cov99"]], [y, y], color=C_CONNECT, lw=1.3, zorder=2)
        ax.scatter(r["cov95"], y, s=52, marker="o", color=C_COV95, zorder=4,
                   edgecolors="white", linewidths=0.6)
        ax.scatter(r["cov99"], y, s=64, marker="s", color=C_COV99, zorder=3,
                   edgecolors=C_COV99_EDGE, linewidths=0.6)
        lo, hi = sorted([r["cov95"], r["cov99"]])
        ax.text(lo - 1.3, y, f"{lo:.1f}", va="center", ha="right", fontsize=9, color=C_TEXT)
        ax.text(hi + 1.3, y, f"{hi:.1f}", va="center", ha="left", fontsize=9, color=C_TEXT)

    ax.set_yticks(ys)
    ax.set_yticklabels([r["group"] for r in cov_rows], fontsize=9)
    # 中文注释：样品数放在每组左侧、对应行下方，避免挤在图中间。
    for y, r in zip(ys, cov_rows):
        ax.text(0.015, y - 0.24, f"n={r['n']:,}", transform=ax.get_yaxis_transform(),
                ha="left", va="top", fontsize=8.5, color=C_NLABEL)

    all_vals = [v for r in cov_rows for v in (r["cov95"], r["cov99"])]
    lo = 45 if min(all_vals) >= 50 else 40
    ax.set_xlim(lo, 100)
    ax.set_ylim(-0.6, n - 0.05)
    ax.set_xlabel("Coverage within modern domain (%)")
    ax.set_title("Quantitative coverage", fontsize=11, pad=6, color=C_TEXT)
    ax.grid(True, axis="x", color=C_GRID, lw=0.6, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    _style_axes(ax)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_COV95,
               markeredgecolor="white", markersize=7.5, label="≤95%"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_COV99,
               markeredgecolor=C_COV99_EDGE, markersize=7.5, label="≤99%"),
    ]
    leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                    ncol=2, frameon=True, framealpha=0.9, edgecolor="#D8DCDF",
                    fontsize=9, handletextpad=0.4, columnspacing=1.4, borderpad=0.5)
    leg.get_frame().set_linewidth(0.5)
    _panel_tag(ax, "(c)")


def plot_applicability_domain(PC_train, PC_arch, high_mask, ad, cov_rows,
                              var1, var2, paths):
    """Applicability-domain / PCA coverage 三联图：左 (a) 大，右上 (b)、右下 (c)。"""
    fig = plt.figure(figsize=(12.5, 6.8))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.0, 1.1], height_ratios=[1.0, 1.0],
                          wspace=0.22, hspace=0.38)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    _panel_a_map(ax_a, PC_train, PC_arch, high_mask, ad, var1, var2)
    _panel_b_distance(ax_b, ad)
    _panel_c_coverage(ax_c, cov_rows)

    for p in paths:
        fig.savefig(p, dpi=600, bbox_inches="tight",
                    facecolor="white", transparent=False)
        print(f"    Saved: {p}")
    plt.close(fig)


def plot_per_class(PC_train, PC_arch, labels, classes, colors, high_mask,
                   var1, var2, variant, path):
    n = len(classes)
    ncols, nrows = 3, int(np.ceil(n / 3))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.7 * nrows),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes).flatten()

    pad = 0.8
    xlim = (PC_train[:, 0].min() - pad, PC_train[:, 0].max() + pad)
    ylim = (PC_train[:, 1].min() - pad, PC_train[:, 1].max() + pad)

    for i, cls in enumerate(classes):
        ax = axes[i]
        m = labels == cls

        ax.scatter(PC_train[~m, 0], PC_train[~m, 1], c="lightgrey",
                   s=3, alpha=0.12, edgecolors="none")
        ax.scatter(PC_train[m, 0], PC_train[m, 1], c=[colors[cls]],
                   s=8, alpha=0.45, edgecolors="none")

        if m.sum() >= 20:
            confidence_ellipse(PC_train[m, 0], PC_train[m, 1], ax, n_std=ELLIPSE_95_SCALE,
                               edgecolor="black", facecolor="none", lw=1.3, ls="--", alpha=0.8)

        if high_mask is not None:
            ax.scatter(PC_arch[~high_mask, 0], PC_arch[~high_mask, 1], c="lightgrey",
                       s=9, alpha=0.4, edgecolors="black", linewidths=0.2, zorder=3)
            ax.scatter(PC_arch[high_mask, 0], PC_arch[high_mask, 1], c="crimson",
                       s=14, alpha=0.75, edgecolors="black", linewidths=0.3, zorder=4)
        else:
            ax.scatter(PC_arch[:, 0], PC_arch[:, 1], c="crimson",
                       s=14, alpha=0.7, edgecolors="black", linewidths=0.3, zorder=4)

        ax.set_title(f"{abbr(cls)}\nmodern n = {m.sum():,}", fontsize=10)
        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.axhline(0, color="grey", lw=0.4, ls="--", alpha=0.4)
        ax.axvline(0, color="grey", lw=0.4, ls="--", alpha=0.4)
        ax.grid(True, alpha=0.15)
        if i % ncols == 0:
            ax.set_ylabel(f"PC2 ({var2 * 100:.1f}%)")
        if i // ncols == nrows - 1:
            ax.set_xlabel(f"PC1 ({var1 * 100:.1f}%)")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey", markersize=6, label="Other modern classes"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="C0", markersize=6, label="Current class"),
        Line2D([0], [0], color="black", ls="--", lw=1.3, label="95% confidence ellipse"),
    ]
    if high_mask is not None:
        legend += [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgrey",
                   markeredgecolor="black", markersize=7, label="Archean low confidence"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                   markeredgecolor="black", markersize=7, label=f"Archean arc-prob >= {HIGH_CONF_THRESHOLD}"),
        ]
    else:
        legend.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="crimson",
                             markeredgecolor="black", markersize=7, label="Archean samples"))

    fig.legend(handles=legend, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02), frameon=True, fontsize=9)
    fig.suptitle(f"Per-class PCA panels - {variant}", fontsize=11, y=1.00)
    plt.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"    Saved: {path}")


# =========================
# 椭圆内占比统计
# =========================
def compute_overlap(PC_train, PC_arch, labels, classes, high_mask) -> pd.DataFrame:
    rows = []
    for cls in classes:
        m = labels == cls
        if m.sum() < 20:
            continue
        mu = PC_train[m].mean(axis=0)
        cov = np.cov(PC_train[m].T)
        try:
            inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            continue
        diff = PC_arch - mu
        inside = np.sum(diff @ inv * diff, axis=1) < CHI2_95_DF2

        row = {
            "class_full": cls, "class_abbr": abbr(cls), "modern_n": int(m.sum()),
            "archean_inside_all_n": int(inside.sum()),
            "archean_inside_all_pct": round(100 * inside.sum() / len(PC_arch), 2),
        }
        if high_mask is not None:
            row["archean_inside_high_n"] = int((inside & high_mask).sum())
            row["archean_inside_high_pct"] = round(100 * (inside & high_mask).sum() / high_mask.sum(), 2)
            row["archean_inside_low_n"] = int((inside & ~high_mask).sum())
            row["archean_inside_low_pct"] = round(100 * (inside & ~high_mask).sum() / (~high_mask).sum(), 2)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("modern_n", ascending=False)


# =========================
# 无插值版本主流程
# =========================
def run_no_impute(train_feats: pd.DataFrame, arch_feats: pd.DataFrame,
                  labels: np.ndarray, classes: list[str], colors: dict,
                  high_mask: np.ndarray | None, low_mask: np.ndarray | None,
                  arch_conf: np.ndarray | None) -> dict:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUT_NO_IMPUTE.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 80}\nVariant: no_impute\nOutput : {OUT_ROOT}\nMode   : no_impute")

    # 中文注释：拟合完整 PCA；前 2 个 PC 用于 (a) 散点，前若干个 PC（~85% 方差）用于 kNN 距离。
    X_train, X_arch = to_pca_space(train_feats, arch_feats)
    pca = PCA(random_state=RANDOM_SEED).fit(X_train)
    PC_train_full = pca.transform(X_train)
    PC_arch_full = pca.transform(X_arch)
    PC_train, PC_arch = PC_train_full[:, :2], PC_arch_full[:, :2]
    var1, var2 = pca.explained_variance_ratio_[:2]
    print(f"  PC1: {var1 * 100:.1f}%  PC2: {var2 * 100:.1f}%  PC1+PC2: {(var1 + var2) * 100:.1f}%")

    # 中文注释：在多维 PCA 子空间里计算 applicability-domain kNN 距离与覆盖率。
    print("  Computing applicability-domain kNN distances ...")
    ad = compute_ad_distances(PC_train_full, PC_arch_full, pca.explained_variance_ratio_)
    cov_rows = coverage_rows(ad["dist_arch"], ad["q95"], ad["q99"], high_mask, low_mask)
    print(f"  AD subspace: {ad['n_pc']} PCs ({ad['cum_var'] * 100:.1f}% var), k={ad['k']}, "
          f"q95={ad['q95']:.3f}, q99={ad['q99']:.3f}")
    for r in cov_rows:
        print(f"    {r['group'].splitlines()[0]:24s} n={r['n']:5d}  cov95={r['cov95']:5.1f}  cov99={r['cov99']:5.1f}")

    print("  Plotting applicability-domain triptych ...")
    plot_applicability_domain(PC_train, PC_arch, high_mask, ad, cov_rows,
                              var1, var2,
                              paths=[AD_FIG_PNG, AD_COMPAT_PNG])

    cov_df = pd.DataFrame([{**{k: v for k, v in r.items() if k != "group"},
                            "group": r["group"].replace("\n", " ")} for r in cov_rows])
    cov_df = cov_df[["group", "n", "cov95", "cov99"]]
    cov_df.insert(0, "n_pc", ad["n_pc"])
    cov_df["q95_dist"] = round(ad["q95"], 4)
    cov_df["q99_dist"] = round(ad["q99"], 4)
    cov_df.to_csv(AD_COVERAGE_CSV, index=False)
    print(f"    Saved: {AD_COVERAGE_CSV}")

    # 中文注释：保留旧版逐类 PCA 面板与椭圆占比统计（写到 pca_no_impute 子目录）。
    print("  Plotting per-class panels ...")
    plot_per_class(PC_train, PC_arch, labels, classes, colors, high_mask,
                   var1, var2, "no_impute", PCA_PER_CLASS_PATH)

    print("  Computing ellipse overlap ...")
    summary = compute_overlap(PC_train, PC_arch, labels, classes, high_mask)
    summary.to_csv(PCA_OVERLAP_SUMMARY_PATH, index=False)
    print(summary.to_string(index=False))

    np.savez(PCA_COORDS_PATH,
             PC_train=PC_train, PC_arch=PC_arch, train_labels=labels,
             arch_conf=arch_conf if arch_conf is not None else np.array([]),
             dist_modern=ad["dist_modern"], dist_arch=ad["dist_arch"],
             ad_q95=ad["q95"], ad_q99=ad["q99"], ad_n_pc=ad["n_pc"],
             explained_variance_ratio=pca.explained_variance_ratio_,
             components=pca.components_)

    loadings = pd.DataFrame(pca.components_[:2].T, columns=["PC1_loading", "PC2_loading"], index=ALL_ELEMENTS)
    loadings["PC1_abs"], loadings["PC2_abs"] = loadings["PC1_loading"].abs(), loadings["PC2_loading"].abs()
    loadings.sort_values("PC1_abs", ascending=False).to_csv(PCA_LOADINGS_PATH)
    print(f"    Saved: pca_coords.npz, pca_loadings.csv, pca_overlap_summary.csv")

    return {
        "variant": "no_impute", "output_dir": str(OUT_ROOT),
        "pc1_variance_pct": round(var1 * 100, 2),
        "pc2_variance_pct": round(var2 * 100, 2),
        "pc12_variance_pct": round((var1 + var2) * 100, 2),
        "ad_n_pc": ad["n_pc"], "ad_cum_var_pct": round(ad["cum_var"] * 100, 1),
        "ad_q95": round(ad["q95"], 4), "ad_q99": round(ad["q99"], 4),
        "modern_classes": len(classes), "modern_samples": int(len(PC_train)),
        "archean_samples": int(len(PC_arch)),
        "high_conf_samples": int(high_mask.sum()) if high_mask is not None else None,
        "low_conf_samples": int(low_mask.sum()) if low_mask is not None else None,
    }


# =========================
# 入口
# =========================
def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Loading data ...")

    modern_raw = clean_columns(pd.read_csv(RAW_MODERN_PATH, low_memory=False))
    archean_raw = pd.read_csv(ARCHEAN_RAW_PATH, low_memory=False)
    print(f"  Modern raw : {len(modern_raw):,} rows")
    print(f"  Archean raw: {len(archean_raw):,} rows")

    for name, df in [("Modern raw", modern_raw), ("Archean raw", archean_raw)]:
        missing = [e for e in ALL_ELEMENTS if e not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    # 中文注释：太古代数据预处理（完整度 + 玄武岩主量筛选），现代训练集不做该筛选。
    archean_raw = preprocess_archean(archean_raw)

    # 标签清理
    modern_raw.dropna(subset=[TECTONIC_COL], inplace=True)
    modern_raw[TECTONIC_COL] = modern_raw[TECTONIC_COL].astype(str).str.strip().str.upper()
    modern_raw.reset_index(drop=True, inplace=True)
    print(f"  Modern raw after label cleanup: {len(modern_raw):,}")

    labels_raw = modern_raw[TECTONIC_COL].values
    classes = sorted(np.unique(labels_raw), key=lambda c: -(labels_raw == c).sum())
    print(f"\nClasses ({len(classes)}):")
    for cls in classes:
        print(f"  {abbr(cls):6s} <- {cls:32s} n_raw = {(labels_raw == cls).sum():,}")

    palette = plt.cm.tab10(np.linspace(0, 1, max(10, len(classes))))
    colors = {cls: palette[i] for i, cls in enumerate(classes)}

    # 太古代置信度（用原始 CSV，跟太古代行索引同步）
    arch_conf = (pd.to_numeric(archean_raw[CONFIDENCE_COL], errors="coerce").to_numpy(dtype=float)
                 if CONFIDENCE_COL in archean_raw.columns else None)
    high_mask = (arch_conf >= HIGH_CONF_THRESHOLD) if arch_conf is not None else None
    low_mask = (arch_conf < LOW_CONF_THRESHOLD) if arch_conf is not None else None
    if high_mask is not None:
        print(f"\nArchean high arc-affinity (P_arc >= {HIGH_CONF_THRESHOLD}): "
              f"{high_mask.sum():,} / {len(arch_conf):,}")
        print(f"Archean low  arc-affinity (P_arc <  {LOW_CONF_THRESHOLD}): "
              f"{low_mask.sum():,} / {len(arch_conf):,}")

    # 中文注释：只保留无插值版本，现代和太古代都直接从原始 CSV 抽取元素特征。
    modern_raw_feats = extract_elements(modern_raw)
    arch_raw_feats = extract_elements(archean_raw)

    # 中文注释：只运行无插值 PCA，不再生成 imputed 版本或版本比较表。
    result = run_no_impute(modern_raw_feats, arch_raw_feats, labels_raw, classes, colors,
                           high_mask, low_mask, arch_conf)
    print(f"\nNo-impute PCA summary:\n{pd.DataFrame([result]).to_string(index=False)}\n\nDone.")


if __name__ == "__main__":
    main()