from __future__ import annotations

# === ?????????? config/paths.py????????????===
import sys as _cfg_sys
from pathlib import Path as _cfg_Path
_cfg_sys.path.insert(0, str(_cfg_Path(__file__).resolve().parents[1]))
from config.paths import (
    ARCHEAN_DIR, TRAIN_NORM_CSV, TEST_NORM_CSV,
    MAIN_MODEL_WEIGHT, QUANTILE_PARAMS_JSON, COMBINED_CSV,
)

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# =========================
# 路径配置
# =========================
# 这里全部使用完整绝对路径，避免路径拼接带来的歧义。
OUTPUT_DIR = (ARCHEAN_DIR / "outputs/distribution_consistency")
TRAIN_CSV_PATH = COMBINED_CSV
ARCHEAN_CSV_PATH = (ARCHEAN_DIR / "data/archean_basalt.csv")

FIG_HARKER_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/fig_harker_train_vs_archean.png")
FIG_CLASSIC_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/fig_classic_discrimination_train_vs_archean.png")
FIG_RATIO_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/fig_ratio_density_train_vs_archean.png")
RATIO_SUMMARY_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/ratio_density_summary.csv")
REPORT_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/distribution_consistency_report.md")
APPENDIX_PATH = (ARCHEAN_DIR / "outputs/distribution_consistency/appendix_training_application_distribution_consistency.md")


# =========================
# 视觉风格
# =========================
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "grid.linestyle": "--",
        "grid.linewidth": 0.45,
        "grid.alpha": 0.45,
        "grid.color": "#B8B8B8",
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "ps.fonttype": 42,
    }
)

COLOR_TRAIN = "#4E79A7"
COLOR_ARCHEAN = "#E4572E"
COLOR_BOUNDARY = "#8A8A8A"


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
    x_bins: int = 68,
    y_bins: int = 52,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """计算独立归一化的二维密度网格，用于比较高密度区形态而不是样本量大小。"""
    hist, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins], range=[x_limits, y_limits])
    if hist.sum() > 0:
        hist = hist / hist.sum()
    # 使用较宽的平滑核，避免低密度外围被解释成有意义的小尺度结构。
    density = smooth_hist2d(hist, sigma_bins=2.8)
    if np.nanmax(density) > 0:
        density = density / np.nanmax(density)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2.0
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2.0
    grid_x, grid_y = np.meshgrid(x_centers, y_centers)
    return grid_x, grid_y, density.T


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


def build_harker_figure(train: pd.DataFrame, archean: pd.DataFrame) -> None:
    """绘制主量元素二维分布一致性图。"""
    modern_color = COLOR_TRAIN
    archean_color = COLOR_ARCHEAN

    panels = [
        ("MGO", "TIO2", "MgO (wt%)", "TiO$_2$ (wt%)"),
        ("MGO", "FEOT", "MgO (wt%)", "FeOT (wt%)"),
        ("MGO", "AL2O3", "MgO (wt%)", "Al$_2$O$_3$ (wt%)"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12.8, 13.2))
    # fig.suptitle("Major-element distributional comparability check", y=0.985, fontsize=12, fontweight="bold")

    panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

    for row, (xcol, ycol, xlabel, ylabel) in enumerate(panels):
        scatter_ax = axes[row, 0]
        density_ax = axes[row, 1]

        x_train, y_train = valid_xy(train, xcol, ycol)
        x_arc, y_arc = valid_xy(archean, xcol, ycol)
        x_lo, x_hi = robust_range(x_train, x_arc, lower=0.5, upper=99.5, pad=0.04)
        y_lo, y_hi = robust_range(y_train, y_arc, lower=0.5, upper=99.5, pad=0.08)
        x_lo = max(0.0, x_lo)
        y_lo = max(0.0, y_lo)

        scatter_ax.scatter(
            x_train,
            y_train,
            s=6,
            c=modern_color,
            alpha=0.075,
            linewidths=0,
            rasterized=True,
            label="Modern training",
        )
        scatter_ax.scatter(
            x_arc,
            y_arc,
            s=16,
            c=archean_color,
            alpha=0.50,
            linewidths=0,
            rasterized=True,
            label="Archean application",
        )

        grid_x_train, grid_y_train, density_train = normalized_density_grid(x_train, y_train, (x_lo, x_hi), (y_lo, y_hi))
        grid_x_arc, grid_y_arc, density_arc = normalized_density_grid(x_arc, y_arc, (x_lo, x_hi), (y_lo, y_hi))
        # 抬高最低密度阈值并减少层数，只保留主体高密度区和主要重叠范围。
        fill_levels = [0.10, 0.22, 0.38, 0.58, 0.80, 1.0]
        line_levels = [0.58, 0.80]

        density_ax.contourf(grid_x_train, grid_y_train, density_train, levels=fill_levels, cmap="Blues", alpha=0.54, antialiased=True)
        density_ax.contour(grid_x_train, grid_y_train, density_train, levels=line_levels, colors=modern_color, linewidths=0.38, alpha=0.24)
        density_ax.contourf(grid_x_arc, grid_y_arc, density_arc, levels=fill_levels, cmap="Reds", alpha=0.56, antialiased=True)
        density_ax.contour(grid_x_arc, grid_y_arc, density_arc, levels=line_levels, colors=archean_color, linewidths=0.38, alpha=0.28)

        # 中心标记只放在 KDE 图中，避免左侧散点图被误读为异常点。
        density_ax.scatter(
            np.nanmedian(x_train),
            np.nanmedian(y_train),
            marker="D",
            s=42,
            facecolors="white",
            edgecolors=modern_color,
            linewidths=1.0,
            zorder=6,
            label="Modern median center",
        )
        density_ax.scatter(
            np.nanmedian(x_arc),
            np.nanmedian(y_arc),
            marker="o",
            s=46,
            facecolors=archean_color,
            edgecolors="white",
            linewidths=0.65,
            zorder=6,
            label="Archean median center",
        )

        for col, ax in enumerate([scatter_ax, density_ax]):
            ax.set_xlim(x_lo, x_hi)
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(True)
            ax.text(
                -0.06,
                1.04,
                panel_labels[row * 2 + col],
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=12,
                fontweight="bold",
                color="#222222",
                path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
            )

        scatter_ax.set_title(f"{ylabel} vs {xlabel}: sample coverage", pad=6)
        density_ax.set_title(f"{ylabel} vs {xlabel}: normalized density", pad=6)

    scatter_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=modern_color, markeredgecolor="none", markersize=4.2, alpha=0.48, label="Modern training"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=archean_color, markeredgecolor="none", markersize=4.8, alpha=0.82, label="Archean application"),
    ]
    scatter_legend = axes[0, 0].legend(handles=scatter_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, framealpha=0.78, fontsize=12, borderpad=0.35, handletextpad=0.45, labelspacing=0.28)
    scatter_legend.get_frame().set_facecolor("#FFFFFF")
    scatter_legend.get_frame().set_edgecolor("#B8B8B8")
    scatter_legend.get_frame().set_linewidth(0.55)

    density_handles = [
        mpatches.Patch(facecolor=modern_color, edgecolor="none", alpha=0.48, label="Density of modern training"),
        mpatches.Patch(facecolor=archean_color, edgecolor="none", alpha=0.48, label="Density of Archean application"),
    ]
    density_legend = axes[0, 1].legend(handles=density_handles, loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True, framealpha=0.78, fontsize=12, borderpad=0.35, handlelength=1.25, handletextpad=0.45, labelspacing=0.28)
    density_legend.get_frame().set_facecolor("#FFFFFF")
    density_legend.get_frame().set_edgecolor("#B8B8B8")
    density_legend.get_frame().set_linewidth(0.55)

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 0.95])
    save_figure(fig, FIG_HARKER_PATH)


def build_classic_figure(train: pd.DataFrame, archean: pd.DataFrame) -> None:
    """绘制经典构造判别图。"""
    fig, axes = plt.subplots(1, 3, figsize=(16.8, 5.6))
    # fig.suptitle("Classic tectonic-discrimination coordinates: training vs application", y=0.98, fontsize=12, fontweight="bold")
    modern_color = COLOR_TRAIN
    archean_color = COLOR_ARCHEAN
    reference_line_color = "#B8B8B8"
    reference_text_color = "#6F6F6F"
    label_box = dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.46)
    panel_labels = ["(a)", "(b)", "(c)"]

    # -------------------------
    # Ti-V 图
    # -------------------------
    ax = axes[0]
    ti_train = train["TIO2"].to_numpy(dtype=float) * 5994.0 / 1000.0
    ti_arc = archean["TIO2"].to_numpy(dtype=float) * 5994.0 / 1000.0
    v_train = train["V"].to_numpy(dtype=float)
    v_arc = archean["V"].to_numpy(dtype=float)

    ax.scatter(ti_train, v_train, s=5.2, c=modern_color, alpha=0.065, linewidths=0, rasterized=True, label="Modern training")
    ax.scatter(ti_arc, v_arc, s=15, c=archean_color, alpha=0.58, linewidths=0, rasterized=True, label="Archean application")

    x_max = max(np.nanpercentile(finite_values(ti_train, ti_arc), 99.5), 1.0)
    y_max = max(np.nanpercentile(finite_values(v_train, v_arc), 99.5), 1.0)
    x_line = np.linspace(max(0.0, np.nanpercentile(finite_values(ti_train, ti_arc), 0.5)), x_max, 300)
    for ratio, label_y in [(20, 0.77), (50, 0.66), (100, 0.57)]:
        y_line = 1000.0 * x_line / ratio
        ax.plot(x_line, y_line, linestyle=":", color=reference_line_color, linewidth=1.5, alpha=1)
        x_lab = x_line[min(len(x_line) - 1, max(0, int(len(x_line) * 0.12)))]
        y_lab = 1000.0 * x_lab / ratio
        if y_lab <= y_max * 0.95:
            ax.text(
                x_lab,
                y_lab,
                f"Ti/V={ratio}",
                fontsize=7.3,
                color=reference_text_color,
                rotation=28,
                va="bottom",
                bbox=label_box,
            )
        else:
            ax.text(
                x_max * 0.82,
                y_max * label_y,
                f"Ti/V={ratio}",
                fontsize=7.3,
                color=reference_text_color,
                rotation=28,
                va="bottom",
                bbox=label_box,
            )

    ax.annotate(
        "IAT / boninite",
        xy=(2.2, 130),
        xytext=(1.6, 80),
        fontsize=7.8,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.95),
    )
    ax.annotate(
        "MORB / BABB",
        xy=(6.0, 260),
        xytext=(4.1, 205),
        fontsize=7.8,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.annotate(
        "OIB",
        xy=(16.0, 450),
        xytext=(14.5, 515),
        fontsize=7.8,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.set_title("Ti-V", pad=6)
    ax.set_xlabel("Ti/1000 (ppm)")
    ax.set_ylabel("V (ppm)")
    ax.set_xlim(0.0, x_max * 1.02)
    ax.set_ylim(0.0, y_max * 1.02)
    ax.grid(True)
    ax.text(
        -0.08,
        1.08,
        panel_labels[0],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#222222",
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
    )
    scatter_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=modern_color, markeredgecolor="none", markersize=4.0, alpha=0.38, label="Modern training"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=archean_color, markeredgecolor="none", markersize=4.8, alpha=0.82, label="Archean application"),
    ]
    classic_legend = ax.legend(handles=scatter_handles, loc="upper right", bbox_to_anchor=(1.06, 1.04), frameon=True, framealpha=0.78, fontsize=12, borderpad=0.35, handletextpad=0.45, labelspacing=0.28)
    classic_legend.get_frame().set_facecolor("#FFFFFF")
    classic_legend.get_frame().set_edgecolor("#B8B8B8")
    classic_legend.get_frame().set_linewidth(0.55)

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

    ax.scatter(nb_yb_train, th_yb_train, s=5.2, c=modern_color, alpha=0.065, linewidths=0, rasterized=True, label="Modern training")
    ax.scatter(nb_yb_arc, th_yb_arc, s=15, c=archean_color, alpha=0.58, linewidths=0, rasterized=True, label="Archean application")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 使用经典判别图的固定坐标范围，便于与 Pearce (2008) 图解体系和前文图件对照。
    x_min, x_max = 0.1, 100.0
    y_min, y_max = 0.01, 10.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    use_plain_log_tick_labels(ax)

    # 用几条常数 Th/Nb 参考线提示偏移方向，不把图画得太满。
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 300)
    for th_nb, label_x in [(0.05, 0.70), (0.10, 0.62), (0.30, 0.54)]:
        y_line = th_nb * x_line
        ax.plot(x_line, y_line, linestyle=":", color=reference_line_color, linewidth=1.5, alpha=1)
        x_lab = x_line[int(len(x_line) * label_x)]
        y_lab = th_nb * x_lab
        if y_lab < y_max * 0.95:
            ax.text(
                x_lab,
                y_lab,
                f"Th/Nb={th_nb:g}",
                fontsize=7.2,
                color=reference_text_color,
                rotation=33,
                va="bottom",
                bbox=label_box,
            )

    ax.annotate(
        "MORB-OIB array",
        xy=(0.025, 0.0035),
        xytext=(0.012, 0.0028),
        fontsize=7.6,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.annotate(
        "higher Th/Yb",
        xy=(20.0, 2.0),
        xytext=(16.5, 2.6),
        fontsize=7.6,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.set_title("Th/Yb-Nb/Yb", pad=6)
    ax.set_xlabel("Nb/Yb")
    ax.set_ylabel("Th/Yb")
    ax.grid(True, which="both")
    ax.text(
        -0.08,
        1.08,
        panel_labels[1],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#222222",
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
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

    ax.scatter(z_train, zy_train, s=5.2, c=modern_color, alpha=0.065, linewidths=0, rasterized=True, label="Modern training")
    ax.scatter(z_arc, zy_arc, s=15, c=archean_color, alpha=0.58, linewidths=0, rasterized=True, label="Archean application")
    ax.set_xscale("log")
    ax.set_yscale("log")

    # 使用经典 Pearce and Norry (1979) 风格的固定坐标范围，避免分位数裁剪改变图解观感。
    x_min, x_max = 5.0, 2000.0
    y_min, y_max = 0.5, 50.0
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    use_plain_log_tick_labels(ax)

    # 给出一条简单的参考线，帮助读者定位相对 Zr/Y 水平。
    x_line = np.logspace(np.log10(x_min), np.log10(x_max), 250)
    for level in [2.0, 3.0, 5.0]:
        ax.plot(x_line, np.full_like(x_line, level), linestyle=":", color=reference_line_color, linewidth=1.5, alpha=1)
        ax.text(
            x_min * 1.18,
            level * 1.02,
            f"Zr/Y={level:g}",
            fontsize=7.2,
            color=reference_text_color,
            bbox=label_box,
        )

    # 固定经典坐标范围后，将构造参照标签放回图内相对空白的位置，避免被点云淹没。
    ax.annotate(
        "MORB",
        xy=(45.0, 3.2),
        xytext=(13.0, 4.4),
        fontsize=8.0,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.annotate(
        "IAB",
        xy=(24.0, 1.6),
        xytext=(8.0, 1.05),
        fontsize=8.0,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.annotate(
        "WPB",
        xy=(210.0, 7.8),
        xytext=(430.0, 14.0),
        fontsize=8.0,
        color=reference_text_color,
        bbox=label_box,
        arrowprops=dict(arrowstyle="-", color=reference_line_color, lw=0.55, alpha=0.65),
    )
    ax.set_title("Zr/Y-Zr", pad=6)
    ax.set_xlabel("Zr (ppm)")
    ax.set_ylabel("Zr/Y")
    ax.grid(True, which="both")
    ax.text(
        -0.08,
        1.08,
        panel_labels[2],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        color="#222222",
        path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
    )

    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])
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

        ax.plot(x_train, y_train, color=modern_color, linewidth=1.8, label="Modern training")
        ax.plot(x_arc, y_arc, color=archean_color, linewidth=1.8, label="Archean application")
        ax.set_title(ratio_name, pad=6)
        ax.set_xlabel(f"log10({ratio_name})")
        ax.set_ylabel("Density")
        ax.grid(True)
        ax.text(
            -0.08,
            1.08,
            panel_labels[idx],
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=14,
            fontweight="bold",
            color="#222222",
            clip_on=False,
            path_effects=[pe.withStroke(linewidth=3.0, foreground="white")],
        )

        # 只在第一个面板放置轻量内嵌图例，避免底部全局图例占用版面。
        if idx == 0:
            ratio_legend = ax.legend(loc="upper right", frameon=True, framealpha=0.78, fontsize=12, borderpad=0.35, handlelength=1.45, handletextpad=0.45, labelspacing=0.28)
            ratio_legend.get_frame().set_facecolor("#FFFFFF")
            ratio_legend.get_frame().set_edgecolor("#B8B8B8")
            ratio_legend.get_frame().set_linewidth(0.55)

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
        f"现代玄武岩训练集：`{TRAIN_CSV_PATH}`",
        "",
        f"太古代玄武岩应用集：`{ARCHEAN_CSV_PATH}`",
        "",
        f"训练集样本数：{train_n}",
        "",
        f"应用集样本数：{arc_n}",
        "",
        "## 图件说明",
        "",
        "图 A1 展示 TiO2-MgO、FeOT-MgO 和 Al2O3-MgO 三组主量元素二维投影。每一行左侧为散点覆盖范围，右侧为分别归一化后的二维密度等值线；右侧图中的符号标记各数据集的二维中位数中心。",
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
图 A1. 现代玄武岩训练集与太古代玄武岩应用集的主量元素二维分布对比。

主量元素二维分布图保留 TiO2-MgO、FeOT-MgO 和 Al2O3-MgO 三组投影，分别对应钛含量、全铁含量和铝含量随 MgO 变化的主要趋势（图 A1）。每一行左侧展示 modern training set 与 Archean application set 的散点覆盖范围，右侧展示两套数据分别归一化后的二维密度等值线；右侧图中的符号标记各数据集的二维中位数中心，因此密度图反映的是各自分布形态而不是样本量差异。太古代应用集使用全部通过基础数值 QC 的样品，不按 softmax ≥ 0.70 置信度阈值筛选。结果显示，太古代样品在三组主量元素投影中均与现代训练集存在可见重叠，同时其高密度区和分布中心相对于现代训练集有一定偏移，说明二者在主量元素空间中具有可比较的覆盖范围和可识别的组成差异。

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

    train = load_dataset(TRAIN_CSV_PATH, "Modern training")
    archean = load_dataset(ARCHEAN_CSV_PATH, "Archean application")

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
