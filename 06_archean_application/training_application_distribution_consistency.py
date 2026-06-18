from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

# === 统一路径配置：所有数据路径来自 config/paths.py ===
import sys as _cfg_sys
_cfg_sys.path.insert(0, str(Path(__file__).resolve().parent))
_cfg_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import (
    TRAIN_MAJOR_NORM_CSV, ARCHEAN_POOL_CSV, ARCHEAN_CONSISTENCY_DIR,
)

from archean_s3_preprocess import preprocess_archean


# =========================
# 路径配置（集中在 config/paths.py）
# =========================
OUTPUT_DIR = Path(str(ARCHEAN_CONSISTENCY_DIR))
TRAIN_CSV_PATH = Path(str(TRAIN_MAJOR_NORM_CSV))
ARCHEAN_CSV_PATH = Path(str(ARCHEAN_POOL_CSV))

FIG_HARKER_PATH = OUTPUT_DIR / "fig_harker_train_vs_archean.png"
FIG_CLASSIC_PATH = OUTPUT_DIR / "fig_classic_discrimination_train_vs_archean.png"
FIG_RATIO_PATH = OUTPUT_DIR / "fig_ratio_density_train_vs_archean.png"
RATIO_SUMMARY_PATH = OUTPUT_DIR / "ratio_density_summary.csv"
REPORT_PATH = OUTPUT_DIR / "distribution_consistency_report.md"
APPENDIX_PATH = OUTPUT_DIR / "appendix_training_application_distribution_consistency.md"


# =========================
# 视觉风格
# =========================
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10.5,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 14,
        "axes.labelcolor": "#000000",
        "axes.edgecolor": "#000000",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.color": "#000000",
        "ytick.color": "#000000",
        "text.color": "#000000",
        "legend.fontsize": 8.5,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.linestyle": (0, (4, 3)),
        "grid.linewidth": 0.55,
        "grid.alpha": 0.85,
        "grid.color": "#D9D9D9",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "ps.fonttype": 42,
    }
)

COLOR_TRAIN = "#6F9FC9"
COLOR_ARCHEAN = "#B3453A"
COLOR_ARCHEAN_LINE = "#B3453A"
COLOR_BOUNDARY = "#8C8C8C"
COLOR_TEXT = "#000000"
COLOR_LABEL = "#000000"
COLOR_GRID = "#D9D9D9"

# 中文注释：现代核密度采用去饱和的板岩蓝渐变 + 逐像素透明度，
# 以连续“光晕”方式渲染，避免分级填充产生的色带台阶。
MODERN_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "modern_density_glow",
    ["#EAF0F6", "#C7D9E8", "#9CBAD7", "#6F97C0", "#4C78A6"],
)

# 中文注释：太古代核密度填充采用从浅砖红直达 #B3453A 的渐变，
# 配合较高的核心不透明度，使高密度区呈现明确的暗红色。
ARCHEAN_DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "archean_density_fill",
    ["#F3D8D3", "#E3ABA2", "#D17E70", "#C05A4C", "#B3453A"],
)

TECTONIC_COLUMN = "TECTONICSETTING"
CFB_LABEL = "CONTINENTAL FLOOD BASALT"
CFB_TARGET_COUNT = 6920
ARCHEAN_TARGET_COUNT = 3012
RANDOM_SEED = 42


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """把不同来源的列名统一成大写、去单位的形式。"""
    rename_map: dict[str, str] = {}
    for col in df.columns:
        clean = re.sub(r"\(.*?\)", "", str(col))
        clean = re.sub(r"[^0-9A-Za-z]+", "", clean).upper()
        rename_map[col] = clean
    return df.rename(columns=rename_map)


def load_dataset(path: Path, dataset_name: str) -> pd.DataFrame:
    """读取并清洗单个数据集。"""
    df = pd.read_csv(path, low_memory=False)
    df = normalize_columns(df)
    df["DATASET"] = dataset_name
    return df


def load_current_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """按PCA脚本的正式口径构造现代训练集和太古代应用集。"""
    train = load_dataset(TRAIN_CSV_PATH, "Modern training")
    if TECTONIC_COLUMN not in train.columns:
        raise ValueError(f"现代训练集缺少列: {TECTONIC_COLUMN}")

    # 中文注释：与PCA脚本一致，统一标签后固定随机种子保留6920条CFB。
    train[TECTONIC_COLUMN] = (
        train[TECTONIC_COLUMN].astype(str).str.strip().str.upper()
    )
    train = train.loc[train[TECTONIC_COLUMN].ne("NAN")].reset_index(drop=True)
    cfb_indices = np.flatnonzero(
        train[TECTONIC_COLUMN].to_numpy() == CFB_LABEL
    )
    if len(cfb_indices) < CFB_TARGET_COUNT:
        raise ValueError(
            f"现代训练集CFB只有 {len(cfb_indices)} 条，"
            f"不能保留 {CFB_TARGET_COUNT} 条"
        )
    selected_cfb = np.sort(
        np.random.default_rng(RANDOM_SEED).choice(
            cfb_indices,
            size=CFB_TARGET_COUNT,
            replace=False,
        )
    )
    non_cfb_indices = np.flatnonzero(
        train[TECTONIC_COLUMN].to_numpy() != CFB_LABEL
    )
    keep_indices = np.sort(
        np.concatenate([non_cfb_indices, selected_cfb])
    )
    train = train.iloc[keep_indices].reset_index(drop=True)

    # 中文注释：太古代数据不再使用2116条插补文件，改用正式3012条无插补数据。
    archean_raw = pd.read_csv(ARCHEAN_CSV_PATH, low_memory=False)
    if "AGE" not in archean_raw.columns or archean_raw["AGE"].isna().any():
        raise ValueError("太古代候选集的AGE列仍有缺失")
    archean = preprocess_archean(
        archean_raw,
        expected_sample_count=ARCHEAN_TARGET_COUNT,
        dataset_name="Harker太古代应用集",
    )
    archean = normalize_columns(archean)
    archean["DATASET"] = "Archean application"
    return train, archean


def to_numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """将目标列强制转成数值，便于后续作图与统计。"""
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def ratio_series(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """计算正值比值，仅保留分子分母都为正的样本。"""
    num = pd.to_numeric(numerator, errors="coerce")
    den = pd.to_numeric(denominator, errors="coerce")
    ratio = pd.Series(np.nan, index=num.index, dtype=float)
    mask = num.gt(0) & den.gt(0) & num.notna() & den.notna()
    ratio.loc[mask] = num.loc[mask] / den.loc[mask]
    return ratio


def finite_values(*arrays: np.ndarray) -> np.ndarray:
    """拼接多个数组中的有限值。"""
    values = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            values.append(arr)
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


def robust_range(*arrays: np.ndarray, lower: float = 0.5, upper: float = 99.5, pad: float = 0.06) -> tuple[float, float]:
    """根据分位数估计稳定坐标范围。"""
    values = finite_values(*arrays)
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    return lo - span * pad, hi + span * pad


def smooth_density(values: np.ndarray, *, grid_size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """用直方图加高斯平滑近似一维核密度曲线。

    这里不依赖 SciPy，避免环境差异导致脚本不可复现。
    """
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    lo, hi = np.percentile(values, [0.5, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(values))
        hi = float(np.nanmax(values))
    if hi <= lo:
        hi = lo + 1.0

    hist, edges = np.histogram(values, bins=grid_size, range=(lo, hi), density=True)
    centers = (edges[:-1] + edges[1:]) / 2.0
    bin_width = edges[1] - edges[0]

    std = float(np.nanstd(values))
    if not np.isfinite(std) or std <= 0:
        std = bin_width
    bandwidth = 1.06 * std * (values.size ** (-1.0 / 5.0))
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        bandwidth = bin_width * 2.0

    sigma_bins = max(bandwidth / bin_width, 1.0)
    radius = int(math.ceil(4.0 * sigma_bins))
    kernel_x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (kernel_x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(hist, kernel, mode="same")
    return centers, density


def valid_xy(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[np.ndarray, np.ndarray]:
    """提取二维投影中的有效样品，仅做基础数值 QC，不使用模型置信度筛选。"""
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y >= 0)
    return x[mask], y[mask]


def display_sample(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """仅对绘图点做固定随机抽样，统计与密度计算仍使用全量数据。"""
    if len(x) <= max_points:
        return x, y
    indices = np.sort(
        np.random.default_rng(seed).choice(
            len(x),
            size=max_points,
            replace=False,
        )
    )
    return x[indices], y[indices]


def stratified_display_sample_by_x(
    x: np.ndarray,
    y: np.ndarray,
    max_points: int,
    *,
    seed: int,
    bins: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """按 MgO 分箱进行固定随机分层抽样，仅用于左侧散点展示。"""
    if len(x) <= max_points:
        return x, y

    rng = np.random.default_rng(seed)
    valid = np.isfinite(x) & np.isfinite(y)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size <= max_points:
        return x[valid_indices], y[valid_indices]

    x_valid = x[valid_indices]
    edges = np.linspace(float(np.nanmin(x_valid)), float(np.nanmax(x_valid)), bins + 1)
    bin_ids = np.clip(np.digitize(x_valid, edges[1:-1], right=False), 0, bins - 1)
    bin_indices = [valid_indices[bin_ids == bin_id] for bin_id in range(bins)]
    non_empty = [idx for idx in bin_indices if idx.size > 0]
    if not non_empty:
        return display_sample(x, y, max_points, seed=seed)

    quotas = np.array([1 for _ in non_empty], dtype=int)
    counts = np.array([idx.size for idx in non_empty], dtype=int)
    remaining = max_points - int(quotas.sum())
    if remaining > 0:
        weights = counts / counts.sum()
        extra = np.floor(weights * remaining).astype(int)
        quotas += extra
        leftover = max_points - int(quotas.sum())
        if leftover > 0:
            order = np.argsort((weights * remaining) - extra)[::-1]
            for pos in order[:leftover]:
                quotas[pos] += 1

    selected: list[np.ndarray] = []
    for idx, quota in zip(non_empty, quotas):
        draw_count = min(int(quota), idx.size)
        selected.append(rng.choice(idx, size=draw_count, replace=False))

    selected_indices = np.sort(np.concatenate(selected))
    if selected_indices.size > max_points:
        selected_indices = np.sort(
            rng.choice(selected_indices, size=max_points, replace=False)
        )
    return x[selected_indices], y[selected_indices]


def gaussian_kernel1d(sigma_bins: float) -> np.ndarray:
    """构造一维高斯核，用于二维直方图的可复现平滑。"""
    sigma_bins = max(float(sigma_bins), 0.8)
    radius = int(math.ceil(4.0 * sigma_bins))
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_hist2d(hist: np.ndarray, sigma_bins: float = 1.4) -> np.ndarray:
    """对二维直方图做可分离高斯平滑，避免依赖 SciPy。"""
    kernel = gaussian_kernel1d(sigma_bins)
    smooth_x = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, hist)
    smooth_xy = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, smooth_x)
    return smooth_xy


def normalized_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    *,
    x_bins: int = 150,
    y_bins: int = 120,
    smooth_sigma: float = 6.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算独立归一化的二维密度网格，用于比较高密度区形态而不是样本量大小。

    采用较高分辨率网格 + 较宽高斯核，得到连续平滑的密度场，
    既能让 imshow 渲染出柔和光晕，也能让等值线呈现干净的同心椭圆。
    """
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins], range=[x_limits, y_limits])
    if hist.sum() > 0:
        hist = hist / hist.sum()
    # 使用较宽的平滑核，避免低密度外围被解释成有意义的小尺度结构。
    density = smooth_hist2d(hist, sigma_bins=smooth_sigma)
    if np.nanmax(density) > 0:
        density = density / np.nanmax(density)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    return grid_x, grid_y, density.T


def density_level_for_mass(density: np.ndarray, mass_fraction: float) -> float:
    """根据完整密度网格计算指定概率质量对应的等密度阈值。"""
    flat = np.asarray(density, dtype=float).ravel()
    flat = flat[np.isfinite(flat) & (flat > 0)]
    if flat.size == 0:
        return np.nan
    sorted_density = np.sort(flat)[::-1]
    cumulative = np.cumsum(sorted_density)
    cumulative = cumulative / cumulative[-1]
    index = int(np.searchsorted(cumulative, mass_fraction, side="left"))
    index = min(index, sorted_density.size - 1)
    return float(sorted_density[index])


def save_figure(fig: plt.Figure, path: Path) -> None:
    """保存图件。"""
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plain_log_tick(value: float, _position: int) -> str:
    """把对数坐标主刻度显示为普通数字，而不是 10 的幂。"""
    if value <= 0 or not np.isfinite(value):
        return ""
    return f"{value:g}"


def use_plain_log_tick_labels(ax: plt.Axes) -> None:
    """对 log 坐标轴使用普通数字刻度标签，并隐藏 minor tick 标签。"""
    formatter = mticker.FuncFormatter(plain_log_tick)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())


def markdown_table(df: pd.DataFrame) -> str:
    """把 DataFrame 转成不依赖额外库的 Markdown 表格。"""
    columns = [str(col) for col in df.columns]
    rows = df.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def cleanup_old_outputs() -> None:
    """清理旧的马氏距离相关输出，避免附录目录里残留无关文件。"""
    if not OUTPUT_DIR.exists():
        return
    for item in OUTPUT_DIR.iterdir():
        name = item.name
        if (
            name.startswith("mahalanobis_")
            or name.startswith("fig_mahalanobis_")
            or name == "fig_ratio_density_ioa_focus.png"
        ):
            if item.is_file():
                item.unlink()


def style_axes(ax: plt.Axes, *, log_grid: bool = False) -> None:
    """统一Nature/GCA风格的坐标轴、刻度和浅色网格。"""
    ax.set_axisbelow(True)
    ax.grid(
        True,
        which="major",
        color=COLOR_GRID,
        linestyle=(0, (4, 3)),
        linewidth=0.55,
        alpha=0.85,
    )
    if log_grid:
        ax.grid(
            True,
            which="minor",
            color="#F1F1F1",
            linewidth=0.30,
            alpha=0.22,
        )
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#000000")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(
        axis="both",
        which="major",
        colors="#000000",
        width=0.7,
        length=3.5,
    )


def box_axes(ax: plt.Axes) -> None:
    """补全子图四条边界线，用于 Harker 六联图的完整图框。"""
    for spine in ("left", "right", "top", "bottom"):
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_color("#000000")
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(top=False, right=False)


def add_log_ellipse(
    ax: plt.Axes,
    center: tuple[float, float],
    width_log: float,
    height_log: float,
    *,
    angle: float,
    facecolor: str,
    edgecolor: str,
    alpha: float,
    zorder: float,
) -> None:
    """在 log-log 坐标中绘制椭圆参考域，输入中心为真实数据坐标。"""
    theta = np.linspace(0.0, 2.0 * np.pi, 240)
    angle_rad = np.deg2rad(angle)
    x0, y0 = np.log10(center[0]), np.log10(center[1])
    x_local = 0.5 * width_log * np.cos(theta)
    y_local = 0.5 * height_log * np.sin(theta)
    x_rot = x_local * np.cos(angle_rad) - y_local * np.sin(angle_rad)
    y_rot = x_local * np.sin(angle_rad) + y_local * np.cos(angle_rad)
    coords = np.column_stack([10 ** (x0 + x_rot), 10 ** (y0 + y_rot)])
    patch = mpatches.Polygon(
        coords,
        closed=True,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(patch)


def style_legend(legend: plt.Legend) -> None:
    """使用低对比度边框和白底，避免图例抢占视觉焦点。"""
    legend.get_frame().set_facecolor("#FFFFFF")
    legend.get_frame().set_edgecolor("#D8D8D8")
    legend.get_frame().set_linewidth(0.55)


def build_harker_figure(train: pd.DataFrame, archean: pd.DataFrame) -> None:
    """绘制主量元素二维分布一致性图。"""
    # 中文注释：Harker 图恢复为两套数据的直接覆盖对比，保持原始图件的分布质感。
    modern_color = COLOR_TRAIN
    archean_color = COLOR_ARCHEAN

    panels = [
        ("MGO", "TIO2", "MgO (wt.%)", "TiO$_2$ (wt.%)"),
        ("MGO", "FEOT", "MgO (wt.%)", "FeO$_T$ (wt.%)"),
        ("MGO", "AL2O3", "MgO (wt.%)", "Al$_2$O$_3$ (wt.%)"),
    ]
    panel_titles = ["TiO$_2$-MgO", "FeO$_T$-MgO", "Al$_2$O$_3$-MgO"]

    # 中文注释：右侧 KDE 子图信息密度较低，设置为稍窄列以减少空白。
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(12.8, 13.2),
        gridspec_kw={"width_ratios": [1.0, 0.90]},
    )
    # fig.suptitle("Major-element distributional comparability check", y=0.985, fontsize=12, fontweight="bold")

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for row, (xcol, ycol, xlabel, ylabel) in enumerate(panels):
        scatter_ax = axes[row, 0]
        density_ax = axes[row, 1]

        x_train, y_train = valid_xy(train, xcol, ycol)
        x_arc, y_arc = valid_xy(archean, xcol, ycol)
        x_train_display, y_train_display = display_sample(
            x_train,
            y_train,
            9000,
            seed=RANDOM_SEED + row,
        )
        x_lo, x_hi = robust_range(x_train, x_arc, lower=0.5, upper=99.5, pad=0.04)
        y_lo, y_hi = robust_range(y_train, y_arc, lower=0.5, upper=99.5, pad=0.08)
        x_lo = max(0.0, x_lo)
        y_lo = max(0.0, y_lo)

        # 中文注释：散点仅表达两套数据来源，不再按构造类型拆分颜色或符号。
        scatter_ax.scatter(
            x_train_display,
            y_train_display,
            s=3.2,
            marker="o",
            facecolors=modern_color,
            alpha=0.36,
            edgecolors="none",
            linewidths=0,
            rasterized=True,
            zorder=1,
            label="Modern training",
        )
        scatter_ax.scatter(
            x_arc,
            y_arc,
            s=15,
            marker="o",
            facecolors=archean_color,
            alpha=0.96,
            edgecolors="white",
            linewidths=0.18,
            rasterized=True,
            zorder=3,
            label="Archean application",
        )

        _, _, density_train = normalized_density_grid(x_train, y_train, (x_lo, x_hi), (y_lo, y_hi))
        grid_x_arc, grid_y_arc, density_arc = normalized_density_grid(x_arc, y_arc, (x_lo, x_hi), (y_lo, y_hi))
        # 中文注释：右侧密度图为“现代蓝色连续光晕（底图）+ 太古代砖红填充（上层）”。
        # 现代用 imshow 逐像素透明度做柔和光晕；太古代同样用 imshow，
        # 但核心接近不透明，使高密度区呈现明确的暗红 #B3453A，再叠细同心线强调形态。
        # ---- 现代分布：imshow 连续光晕（底图）----
        modern_alpha = np.clip(density_train, 0.0, 1.0) ** 0.60 * 0.92
        density_ax.imshow(
            density_train,
            origin="lower",
            extent=(x_lo, x_hi, y_lo, y_hi),
            cmap=MODERN_DENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
            alpha=modern_alpha,
            aspect="auto",
            interpolation="bilinear",
            zorder=1,
        )
        # ---- 太古代分布：清晰可见的砖红填充 + 细同心线 ----
        # gamma 较小让中高密度都较实，核心 alpha 接近 0.9 呈现暗红。
        arc_alpha = np.clip(density_arc, 0.0, 1.0) ** 0.55 * 0.88
        density_ax.imshow(
            density_arc,
            origin="lower",
            extent=(x_lo, x_hi, y_lo, y_hi),
            cmap=ARCHEAN_DENSITY_CMAP,
            vmin=0.0,
            vmax=1.0,
            alpha=arc_alpha,
            aspect="auto",
            interpolation="bilinear",
            zorder=3,
        )
        density_ax.contour(
            grid_x_arc,
            grid_y_arc,
            density_arc,
            levels=[0.25, 0.45, 0.65, 0.85],
            colors="#7E2820",
            linewidths=0.55,
            alpha=0.65,
            antialiased=True,
            zorder=4,
        )

        # 中心标记只放在 KDE 图中，避免左侧散点图被误读为异常点。
        density_ax.scatter(
            np.nanmedian(x_train),
            np.nanmedian(y_train),
            marker="D",
            s=52,
            facecolors="white",
            edgecolors=modern_color,
            linewidths=1.30,
            zorder=7,
            label="Modern median center",
        )
        density_ax.scatter(
            np.nanmedian(x_arc),
            np.nanmedian(y_arc),
            marker="o",
            s=58,
            facecolors="white",
            edgecolors=COLOR_ARCHEAN_LINE,
            linewidths=1.30,
            zorder=8,
            label="Archean median center",
        )

        for col, ax in enumerate([scatter_ax, density_ax]):
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            style_axes(ax)
            box_axes(ax)
            ax.text(
                -0.12,
                1.06,
                panel_labels[row * 2 + col],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=18,
                fontweight="bold",
                color=COLOR_TEXT,
            )

        scatter_ax.set_title(panel_titles[row], pad=7)
        density_ax.set_title(panel_titles[row], pad=7)

    scatter_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=modern_color,
            markeredgecolor="none",
            markersize=6.8,
            alpha=0.65,
            label="Modern training points",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=archean_color,
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=6.8,
            alpha=0.90,
            label="Archean application points",
        ),
    ]
    scatter_legend = axes[0, 0].legend(handles=scatter_handles, loc="upper right", frameon=True, framealpha=0.94, fontsize=14, borderpad=0.42, handletextpad=0.45, labelspacing=0.34)
    style_legend(scatter_legend)

    median_modern_handle = Line2D(
        [0],
        [0],
        marker="D",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=modern_color,
        markeredgewidth=1.20,
        markersize=6.0,
    )
    median_archean_handle = Line2D(
        [0],
        [0],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=COLOR_ARCHEAN_LINE,
        markeredgewidth=1.20,
        markersize=6.0,
    )
    density_handles = [
        mpatches.Patch(facecolor="#9CBAD7", edgecolor="none", alpha=0.92),
        mpatches.Patch(facecolor="#B3453A", edgecolor="#7E2820", linewidth=0.8, alpha=0.90),
        (median_modern_handle, median_archean_handle),
    ]
    density_labels = ["Modern density", "Archean density", "Median centers"]
    density_legend = axes[0, 1].legend(
        density_handles,
        density_labels,
        loc="upper right",
        frameon=True,
        framealpha=0.94,
        fontsize=14,
        borderpad=0.7,
        handlelength=2.8,
        handletextpad=0.7,
        labelspacing=0.34,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    style_legend(density_legend)

    # 中文注释：右列变窄后适当收紧列间距，避免中间空白过大。
    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95], w_pad=2)
    save_figure(fig, FIG_HARKER_PATH)


def build_classic_figure(train: pd.DataFrame, archean: pd.DataFrame) -> None:
    """绘制经典构造判别图。"""
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6))
    # fig.suptitle("Classic tectonic-discrimination coordinates: training vs application", y=0.98, fontsize=12, fontweight="bold")
    modern_color = COLOR_TRAIN
    archean_color = COLOR_ARCHEAN
    reference_line_color = "#8C8C8C"
    reference_text_color = "#000000"
    label_box = dict(boxstyle="round,pad=0.10", facecolor="white", edgecolor="none", alpha=0.60)
    panel_labels = ["(a)", "(b)", "(c)"]

    # -------------------------
    # Ti-V 图
    # -------------------------
    ax = axes[0]
    ti_train = train["TIO2"].to_numpy(dtype=float) * 5994.0 / 1000.0
    ti_arc = archean["TIO2"].to_numpy(dtype=float) * 5994.0 / 1000.0
    v_train = train["V"].to_numpy(dtype=float)
    v_arc = archean["V"].to_numpy(dtype=float)
    ti_train_display, v_train_display = display_sample(
        ti_train,
        v_train,
        10000,
        seed=RANDOM_SEED + 10,
    )

    ax.scatter(ti_train_display, v_train_display, s=3.0, c=modern_color, alpha=0.24, linewidths=0, rasterized=True, zorder=1, label="Modern training")
    ax.scatter(ti_arc, v_arc, s=15, c=archean_color, alpha=0.88, edgecolors="white", linewidths=0.28, rasterized=True, zorder=3, label="Archean application")

    x_max = max(np.nanpercentile(finite_values(ti_train, ti_arc), 99.5), 1.0)
    y_max = max(np.nanpercentile(finite_values(v_train, v_arc), 99.5), 1.0)
    x_line = np.linspace(max(0.0, np.nanpercentile(finite_values(ti_train, ti_arc), 0.5)), x_max, 300)
    # 中文注释：按 Ti/V 比值线给出淡色参考域，帮助定位 IAT、MORB/BABB 与 OIB 趋势。
    y_20 = 1000.0 * x_line / 20.0
    y_50 = 1000.0 * x_line / 50.0
    y_100 = 1000.0 * x_line / 100.0
    ax.fill_between(x_line, np.minimum(y_20, y_max * 1.02), y_max * 1.02, color="#E7E2D6", alpha=0.28, zorder=0)
    ax.fill_between(x_line, y_50, y_20, where=y_20 <= y_max * 1.02, color="#DDE8EE", alpha=0.30, zorder=0)
    ax.fill_between(x_line, y_100, y_50, where=y_50 <= y_max * 1.02, color="#E9EEF1", alpha=0.24, zorder=0)
    for ratio in [20, 50, 100]:
        y_line = 1000.0 * x_line / ratio
        ax.plot(x_line, y_line, linestyle=(0, (4, 3)), color=reference_line_color, linewidth=1.2, alpha=0.90, zorder=2)
    # 中文注释：仅保留关键构造环境名称，避免小字标签堆叠。
    ax.annotate(
        "MORB / BABB",
        xy=(6.0, 260),
        xytext=(1.4, 545),
        fontsize=12,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.annotate(
        "OIB",
        xy=(16.0, 450),
        xytext=(14.6, 515),
        fontsize=12,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.set_title("Ti-V", pad=6)
    ax.set_xlabel("Ti/1000 (ppm)")
    ax.set_ylabel("V (ppm)")
    ax.set_xlim(0.0, x_max * 1.02)
    ax.set_ylim(0.0, y_max * 1.02)
    style_axes(ax)
    box_axes(ax)
    ax.text(
        -0.08,
        1.08,
        panel_labels[0],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLOR_TEXT,
    )
    scatter_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=modern_color, markeredgecolor="none", markersize=4.0, alpha=0.55, label="Modern training"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=archean_color, markeredgecolor="white", markeredgewidth=0.5, markersize=5.2, alpha=0.90, label="Archean application"),
    ]
    classic_legend = ax.legend(handles=scatter_handles, loc="upper right", frameon=True, framealpha=0.92, fontsize=9.5, borderpad=0.45, handletextpad=0.45, labelspacing=0.32)
    style_legend(classic_legend)

    # -------------------------
    # Th/Yb - Nb/Yb 图
    # -------------------------
    ax = axes[1]
    nb_train_raw = train["NB"].to_numpy(dtype=float)
    yb_train_raw = train["YB"].to_numpy(dtype=float)
    th_train_raw = train["TH"].to_numpy(dtype=float)
    nb_arc_raw = archean["NB"].to_numpy(dtype=float)
    yb_arc_raw = archean["YB"].to_numpy(dtype=float)
    th_arc_raw = archean["TH"].to_numpy(dtype=float)

    mask_train = np.isfinite(nb_train_raw) & np.isfinite(yb_train_raw) & np.isfinite(th_train_raw) & (nb_train_raw > 0) & (yb_train_raw > 0) & (th_train_raw > 0)
    mask_arc = np.isfinite(nb_arc_raw) & np.isfinite(yb_arc_raw) & np.isfinite(th_arc_raw) & (nb_arc_raw > 0) & (yb_arc_raw > 0) & (th_arc_raw > 0)

    nb_yb_train = nb_train_raw[mask_train] / yb_train_raw[mask_train]
    th_yb_train = th_train_raw[mask_train] / yb_train_raw[mask_train]
    nb_yb_arc = nb_arc_raw[mask_arc] / yb_arc_raw[mask_arc]
    th_yb_arc = th_arc_raw[mask_arc] / yb_arc_raw[mask_arc]
    nb_yb_train_display, th_yb_train_display = display_sample(
        nb_yb_train,
        th_yb_train,
        10000,
        seed=RANDOM_SEED + 11,
    )

    ax.scatter(nb_yb_train_display, th_yb_train_display, s=3.0, c=modern_color, alpha=0.24, linewidths=0, rasterized=True, zorder=1, label="Modern training")
    ax.scatter(nb_yb_arc, th_yb_arc, s=15, c=archean_color, alpha=0.88, edgecolors="white", linewidths=0.28, rasterized=True, zorder=3, label="Archean application")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 使用经典判别图的固定坐标范围，便于与 Pearce (2008) 图解体系和前文图件对照。
    x_min, x_max = 0.1, 100.0
    y_min, y_max = 0.01, 10.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    use_plain_log_tick_labels(ax)
    # 中文注释：参考 Pearce 风格图解，在 log-log 坐标中添加 MORB-OIB 斜带和弧环境包络域。
    array_x = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    ax.fill_between(
        array_x,
        np.maximum(0.028 * array_x, y_min),
        np.minimum(0.18 * array_x, y_max),
        color="#6E6E6E",
        alpha=0.28,
        zorder=0,
    )
    add_log_ellipse(
        ax,
        center=(0.55, 0.30),
        width_log=0.72,
        height_log=1.20,
        angle=-28,
        facecolor="#7A7A7A",
        edgecolor="#333333",
        alpha=0.34,
        zorder=0.2,
    )
    add_log_ellipse(
        ax,
        center=(1.75, 1.15),
        width_log=0.72,
        height_log=1.22,
        angle=-28,
        facecolor="#E6E6E6",
        edgecolor="#333333",
        alpha=0.58,
        zorder=0.25,
    )
    ax.text(0.32, 0.36, "OA", fontsize=12, ha="center", va="center", color="#222222", zorder=1)
    ax.text(1.05, 1.65, "CA", fontsize=12, ha="center", va="center", color="#222222", zorder=1)

    # 用几条常数 Th/Nb 参考线提示偏移方向，不把图画得太满。
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    for th_nb in [0.05, 0.10, 0.30]:
        y_line = th_nb * x_line
        ax.plot(x_line, y_line, linestyle=(0, (4, 3)), color=reference_line_color, linewidth=1.2, alpha=0.90, zorder=2)

    ax.annotate(
        "MORB-OIB array",
        xy=(0.32, 0.055),
        xytext=(0.14, 0.022),
        fontsize=12,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.set_title("Th/Yb-Nb/Yb", pad=6)
    ax.set_xlabel("Nb/Yb")
    ax.set_ylabel("Th/Yb", labelpad=-8)
    style_axes(ax, log_grid=True)
    box_axes(ax)
    ax.text(
        -0.08,
        1.08,
        panel_labels[1],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLOR_TEXT,
    )

    # -------------------------
    # Zr/Y - Zr 图
    # -------------------------
    ax = axes[2]
    z_train_raw = train["ZR"].to_numpy(dtype=float)
    y_train_raw = train["Y"].to_numpy(dtype=float)
    z_arc_raw = archean["ZR"].to_numpy(dtype=float)
    y_arc_raw = archean["Y"].to_numpy(dtype=float)

    mask_train = np.isfinite(z_train_raw) & np.isfinite(y_train_raw) & (z_train_raw > 0) & (y_train_raw > 0)
    mask_arc = np.isfinite(z_arc_raw) & np.isfinite(y_arc_raw) & (z_arc_raw > 0) & (y_arc_raw > 0)

    z_train = z_train_raw[mask_train]
    zy_train = z_train_raw[mask_train] / y_train_raw[mask_train]
    z_arc = z_arc_raw[mask_arc]
    zy_arc = z_arc_raw[mask_arc] / y_arc_raw[mask_arc]
    z_train_display, zy_train_display = display_sample(
        z_train,
        zy_train,
        10000,
        seed=RANDOM_SEED + 12,
    )

    ax.scatter(z_train_display, zy_train_display, s=3.0, c=modern_color, alpha=0.24, linewidths=0, rasterized=True, zorder=1, label="Modern training")
    ax.scatter(z_arc, zy_arc, s=15, c=archean_color, alpha=0.88, edgecolors="white", linewidths=0.28, rasterized=True, zorder=3, label="Archean application")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 使用经典 Pearce and Norry (1979) 风格的固定坐标范围，避免分位数裁剪改变图解观感。
    x_min, x_max = 5.0, 2000.0
    y_min, y_max = 0.5, 50.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    use_plain_log_tick_labels(ax)

    # 中文注释：添加 Pearce and Norry 风格的淡色参考域，作为 MORB、IAB、WPB 的视觉定位。
    for coords, label, label_xy, color in [
        ([(8, 0.75), (18, 1.05), (55, 2.4), (170, 4.2), (420, 7.0), (210, 11.0), (45, 6.0), (12, 2.0)], "", (36, 3.7), "#DDE8EE"),
        ([(5.5, 0.55), (10, 0.75), (24, 1.4), (58, 2.6), (35, 4.1), (11, 2.2)], "IAB", (8.5, 1.05), "#E6E1D5"),
        ([(95, 4.8), (220, 7.2), (650, 15.0), (1800, 40.0), (2000, 50.0), (420, 31.0), (150, 14.0)], "WPB", (520, 18.0), "#ECECEC"),
    ]:
        ax.add_patch(
            mpatches.Polygon(
                coords,
                closed=True,
                facecolor=color,
                edgecolor="#777777",
                linewidth=0.65,
                alpha=0.34,
                zorder=0,
            )
        )
        if label:
            ax.text(label_xy[0], label_xy[1], label, fontsize=12, color="#333333", ha="center", va="center", zorder=1)

    # 给出一条简单的参考线，帮助读者定位相对 Zr/Y 水平。
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 250)
    for level in [2.0, 3.0, 5.0]:
        ax.plot(x_line, np.full_like(x_line, level), linestyle=(0, (4, 3)), color=reference_line_color, linewidth=1.2, alpha=0.90, zorder=2)

    # 中文注释：构造域名称已写入多边形内部，此处不再重复箭头标注。
    ax.set_title("Zr/Y-Zr", pad=6)
    ax.set_xlabel("Zr (ppm)")
    ax.set_ylabel("Zr/Y", labelpad=-4)
    style_axes(ax, log_grid=True)
    box_axes(ax)
    ax.text(
        -0.08,
        1.08,
        panel_labels[2],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
        color=COLOR_TEXT,
    )

    # 中文注释：收紧三个经典判别子图之间的横向间距。
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94], w_pad=0.2)
    save_figure(fig, FIG_CLASSIC_PATH)


def build_ratio_figure(train: pd.DataFrame, archean: pd.DataFrame) -> pd.DataFrame:
    """绘制判别比值核密度分布，并返回统计汇总表。"""
    modern_color = COLOR_TRAIN
    archean_color = COLOR_ARCHEAN
    ratio_specs = [
        ("Ba/Nb", "BA", "NB"),
        ("Th/Nb", "TH", "NB"),
        ("La/Yb", "LA", "YB"),
        ("Zr/Y", "ZR", "Y"),
    ]
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    summary_rows: list[dict[str, object]] = []

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.2))
    # fig.suptitle("Discriminant-ratio density comparison", y=0.98, fontsize=12, fontweight="bold")
    axes = axes.ravel()

    for idx, (ratio_name, num_col, den_col) in enumerate(ratio_specs):
        ax = axes[idx]
        train_ratio = ratio_series(train[num_col], train[den_col]).to_numpy(dtype=float)
        arc_ratio = ratio_series(archean[num_col], archean[den_col]).to_numpy(dtype=float)

        train_ratio = train_ratio[np.isfinite(train_ratio) & (train_ratio > 0)]
        arc_ratio = arc_ratio[np.isfinite(arc_ratio) & (arc_ratio > 0)]

        summary_rows.append(
            {
                "dataset": "Modern training",
                "ratio": ratio_name,
                "n_positive": int(train_ratio.size),
                "median": float(np.median(train_ratio)) if train_ratio.size else np.nan,
                "q25": float(np.percentile(train_ratio, 25)) if train_ratio.size else np.nan,
                "q75": float(np.percentile(train_ratio, 75)) if train_ratio.size else np.nan,
                "min": float(np.min(train_ratio)) if train_ratio.size else np.nan,
                "max": float(np.max(train_ratio)) if train_ratio.size else np.nan,
            }
        )
        summary_rows.append(
            {
                "dataset": "Archean application",
                "ratio": ratio_name,
                "n_positive": int(arc_ratio.size),
                "median": float(np.median(arc_ratio)) if arc_ratio.size else np.nan,
                "q25": float(np.percentile(arc_ratio, 25)) if arc_ratio.size else np.nan,
                "q75": float(np.percentile(arc_ratio, 75)) if arc_ratio.size else np.nan,
                "min": float(np.min(arc_ratio)) if arc_ratio.size else np.nan,
                "max": float(np.max(arc_ratio)) if arc_ratio.size else np.nan,
            }
        )

        train_log = np.log10(train_ratio)
        arc_log = np.log10(arc_ratio)
        x_train, y_train = smooth_density(train_log)
        x_arc, y_arc = smooth_density(arc_log)

        ax.plot(x_train, y_train, color=modern_color, linewidth=1.5, alpha=0.92, label="Modern training", zorder=2)
        ax.plot(x_arc, y_arc, color=archean_color, linewidth=2.0, alpha=1.0, label="Archean application", zorder=3)
        ax.set_title(ratio_name, pad=6)
        ax.set_xlabel(f"log10({ratio_name})")
        ax.set_ylabel("Density")
        style_axes(ax)
        box_axes(ax)
        ax.text(
            -0.08,
            1.08,
            panel_labels[idx],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=16,
            fontweight="bold",
            color=COLOR_TEXT,
            clip_on=False,
        )

        # 只在第一个面板放置轻量内嵌图例，避免底部全局图例占用版面。
        if idx == 0:
            ratio_legend = ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=14, borderpad=0.45, handlelength=1.65, handletextpad=0.45, labelspacing=0.32)
            style_legend(ratio_legend)

        if x_train.size and x_arc.size:
            x_min = min(np.nanmin(x_train), np.nanmin(x_arc))
            x_max = max(np.nanmax(x_train), np.nanmax(x_arc))
            ax.set_xlim(x_min - 0.08 * (x_max - x_min), x_max + 0.08 * (x_max - x_min))

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.98])
    save_figure(fig, FIG_RATIO_PATH)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(RATIO_SUMMARY_PATH, index=False, encoding="utf-8-sig")
    return summary


def build_report(train: pd.DataFrame, archean: pd.DataFrame, ratio_summary: pd.DataFrame) -> None:
    """写出简短的分析报告。"""
    train_n = int(len(train))
    arc_n = int(len(archean))

    lines = [
        "# 训练集与应用集分布一致性报告",
        "",
        "## 数据来源",
        "",
        f"现代玄武岩训练集（CFB固定保留{CFB_TARGET_COUNT}条）：`{TRAIN_CSV_PATH}`",
        "",
        f"太古代玄武岩应用集（统一筛选、无插补）：`{ARCHEAN_CSV_PATH}`",
        "",
        f"训练集样本数：{train_n}",
        "",
        f"应用集样本数：{arc_n}",
        "",
        "## 图件说明",
        "",
        "图 A1 展示 TiO2-MgO、FeOT-MgO 和 Al2O3-MgO 三组主量元素二维投影。每一行左侧为散点覆盖范围，其中现代训练集散点仅为提高可视化清晰度进行了固定 random_state 抽样，太古代应用集保留全部有效样品；右侧为基于完整数据集分别归一化后的二维密度图，等密度线、二维中位数中心以及所有统计结果均基于完整数据集计算。",
        "",
        "图 A2 展示经典构造判别坐标。Ti-V、Th/Yb-Nb/Yb 以及 Zr/Y-Zr 三类图解给出了与现代训练集可比但并不完全重合的构造判别空间。",
        "",
        "图 A3 展示关键判别比值的核密度分布。Ba/Nb、Th/Nb、La/Yb 与 Zr/Y 一起覆盖了流体活动元素富集、Th 相对 Nb 富集、轻重稀土分异以及 Zr-Y 系统变化等不同过程。",
        "",
        "## 比值统计",
        "",
    ]

    table = ratio_summary.copy()
    for col in ["median", "q25", "q75", "min", "max"]:
        table[col] = table[col].map(lambda x: f"{x:.6g}" if pd.notna(x) else "")
    lines.append(markdown_table(table))
    lines.extend(
        [
            "",
            "## 小结",
            "",
            "主量元素、经典构造判别坐标和判别比值三层证据一致指向：太古代应用集并未脱离现代玄武岩训练集的主要地球化学空间，但两者在峰位与尾部上存在系统性差异。GeoDAN 的输出更适合被理解为现代构造端元地球化学亲和性的判别结果。",
            "",
            "结合正文 5.3.2 节的预测概率阈值（≥0.70），分布检验与模型置信度共同构成了一个双重约束框架：一个反映数据空间的一致性，一个反映预测输出的稳定程度。",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def build_appendix(ratio_summary: pd.DataFrame) -> None:
    """写出附录 markdown 小节。"""
    appendix_text = """### 训练集与应用集的地球化学分布一致性

本节从主量元素变异、经典构造判别坐标和关键判别比值三个层面比较现代玄武岩训练集与太古代玄武岩应用集在地球化学空间中的相对关系。模型是在现代样品上训练得到，因此太古代样品的输出需要结合训练集的覆盖范围一并解释。若应用集主要落在现代训练集的主体分布内，则预测结果可视为较稳定的地球化学亲和性判别；若样品靠近训练空间边缘，则说明其与现代端元之间仍存在一定距离。本节结果表明，太古代应用集整体落在现代训练集的边缘到中间区域，二者在构造判别相关元素空间中存在可比较的重叠。

![](fig_harker_train_vs_archean.png)
图 A1. 现代玄武岩训练集与太古代玄武岩应用集的主量元素二维分布对比。左侧散点图中，现代训练集仅为提高可视化清晰度进行了固定 random_state 抽样，太古代应用集保留全部有效样品；右侧 KDE 密度填充、等密度线、中位数中心及相关统计均基于完整数据集计算。

主量元素二维分布图保留 TiO2-MgO、FeOT-MgO 和 Al2O3-MgO 三组投影，分别对应钛含量、全铁含量和铝含量随 MgO 变化的主要趋势（图 A1）。每一行左侧展示 modern training set 与 Archean application set 的散点覆盖范围，右侧展示两套数据分别归一化后的二维密度等值线；右侧图中的空心菱形和空心圆分别标记现代训练集与太古代应用集的二维中位数中心，因此密度图反映的是各自分布形态而不是样本量差异。太古代应用集使用全部通过基础数值 QC 的样品，不按 softmax ≥ 0.70 置信度阈值筛选。结果显示，太古代样品在三组主量元素投影中均与现代训练集存在可见重叠，同时其高密度区和分布中心相对于现代训练集有一定偏移，说明二者在主量元素空间中具有可比较的覆盖范围和可识别的组成差异。

![](fig_classic_discrimination_train_vs_archean.png)
图 A2. 现代训练集与太古代应用集在经典构造判别坐标中的分布对比。

经典构造判别图提供了比主量元素二维分布图更直接的构造参照。Ti-V 图中，太古代样品主要落在 Ti/V = 20–50 与 50–100 的区间，对应 MORB/BABB 到 OIB 端元之间的过渡带，少数样品向 IAT 边界靠近（图 A2）。Th/Yb-Nb/Yb 图中，太古代样品整体相对现代训练集向更高 Th/Yb 方向偏移，并沿 MORB-OIB array 展开，这种形态通常与俯冲相关组分输入或地壳混染信号有关。Zr/Y-Zr 图中，两套数据主要重叠于 MORB 到 IAB 的范围，部分样品向 WPB 方向延伸，说明应用集在经典构造判别空间中具有可比较的重叠，但其分布重心并不与现代训练集完全重合。

![](fig_ratio_density_train_vs_archean.png)
图 A3. 现代训练集与太古代应用集关键判别比值的核密度分布对比。

判别比值核密度分布进一步约束了这种重叠与偏移的关系。Ba/Nb 主要反映流体活动元素相对高场强元素的富集，Th/Nb 主要约束 Th 相对 Nb 的富集，La/Yb 反映轻重稀土分异及源区富集程度，Zr/Y 则更多指示 Zr-Y 系统变化、部分熔融程度和源区差异（图 A3）。这些比值在两套数据之间具有一定重叠，同时在峰位和尾部上仍保留系统性差异，说明太古代玄武岩与现代训练集在分布上仍存在可识别的差别，但二者仍共享一部分可比的地球化学空间。因此，GeoDAN 的输出更适合解释为太古代样品与现代构造端元之间的地球化学亲和性判别。

结合正文 5.3.2 节的预测概率阈值（≥0.70），本节分布检验与模型置信度共同构成了一个双重约束框架：一个反映数据空间的一致性，一个反映预测输出的稳定程度。
"""
    APPENDIX_PATH.write_text(appendix_text, encoding="utf-8")


def main() -> None:
    """主流程。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_old_outputs()

    train, archean = load_current_datasets()

    needed_columns = [
        "MGO",
        "TIO2",
        "FEOT",
        "AL2O3",
        "V",
        "NB",
        "YB",
        "TH",
        "ZR",
        "Y",
        "LA",
        "BA",
    ]
    train = to_numeric_frame(train, needed_columns)
    archean = to_numeric_frame(archean, needed_columns)

    build_harker_figure(train, archean)
    build_classic_figure(train, archean)
    ratio_summary = build_ratio_figure(train, archean)
    build_report(train, archean, ratio_summary)
    build_appendix(ratio_summary)


if __name__ == "__main__":
    main()
