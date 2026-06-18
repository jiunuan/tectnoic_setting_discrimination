# -*- coding: utf-8 -*-
"""绘制六案例构造环境组成图和高弧信号年龄山脊图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib import font_manager
import numpy as np
import pandas as pd

# === 统一路径配置：所有数据路径来自 config/paths.py ===
import sys as _cfg_sys
_cfg_sys.path.insert(0, str(Path(__file__).resolve().parent))
_cfg_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import ARCHEAN_CASE_DIR

from archean_s3_preprocess import CASE_STUDIES_ORDER, CASE_STUDY_TITLES


PREDICTIONS_DIR = Path(str(ARCHEAN_CASE_DIR)) / "predictions"
BARS_OUTPUT_PATH = PREDICTIONS_DIR / "fig_case_studies_bars.png"
RIDGELINE_OUTPUT_PATH = PREDICTIONS_DIR / "fig_case_studies_ridgeline.png"
COMBINED_OUTPUT_PATH = PREDICTIONS_DIR / "fig_case_studies_bars_ridgeline.png"
CASE_PREDICTION_PATHS = {
    "Isua": PREDICTIONS_DIR / "Isua_predictions.csv",
    "Pilbara": PREDICTIONS_DIR / "Pilbara_predictions.csv",
    "Ivisaartoq": PREDICTIONS_DIR / "Ivisaartoq_predictions.csv",
    "Norseman_Kambalda": PREDICTIONS_DIR / "Norseman_Kambalda_predictions.csv",
    "Abitibi": PREDICTIONS_DIR / "Abitibi_predictions.csv",
    "North_China_Craton": PREDICTIONS_DIR / "North_China_Craton_predictions.csv",
}

SCI_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": [
        "Arial", "Helvetica", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "savefig.dpi": 600,
    "ps.fonttype": 42,
}

CLASS_COLORS = {
    # 中文注释：低饱和暖色与KDE的杏橙填色、陶土红曲线保持协调。
    "Continental arc": "#C94A34",
    "Intra-oceanic arc": "#D9755B",
    "Island arc": "#E8A66D",
    "BACK-ARC_BASIN": "#7896AE",
    "SPREADING_CENTER": "#A3BFCA",
    "OCEANIC PLATEAU": "#729675",
    "OCEAN ISLAND": "#A7BD83",
    "CONTINENTAL FLOOD BASALT": "#916B82",
    "CONTINENTAL_RIFT": "#BDA3B3",
}
CLASS_ABBREVS = {
    "Continental arc": "CA",
    "Intra-oceanic arc": "IOA",
    "Island arc": "IA",
    "BACK-ARC_BASIN": "BAB",
    "SPREADING_CENTER": "MOR",
    "OCEANIC PLATEAU": "OP",
    "OCEAN ISLAND": "OI",
    "CONTINENTAL FLOOD BASALT": "CF",
    "CONTINENTAL_RIFT": "CR",
}
ARC_RELATED_LABELS = {"Continental arc", "Intra-oceanic arc", "Island arc"}
LEGEND_ORDER = ["CF", "CR", "OP", "IA", "BAB", "OI", "IOA", "CA", "MOR"]

RIDGE_PERIODS = [
    {"name": "Eoarchean", "start": 4.00, "end": 3.60, "color": "#C8D9EC"},
    {"name": "Paleoarchean", "start": 3.60, "end": 3.20, "color": "#F0CADA"},
    {"name": "Mesoarchean", "start": 3.20, "end": 2.80, "color": "#FAE5C2"},
    {"name": "Neoarchean", "start": 2.80, "end": 2.50, "color": "#CDE8C7"},
]
RIDGE_X_MIN_GA = 2.45
RIDGE_X_MAX_GA = 4.00
RIDGE_KDE_BANDWIDTH_GA = 0.06
RIDGE_LANE_FILL = 0.62
RIDGE_RUG_MAX_N = 45
RIDGE_FILL_COLOR = "#63B7AF"
RIDGE_LINE_COLOR = "#287F7A"


def _configure_chinese_font() -> None:
    """注册Windows常见中文字体。"""
    for font_path in (
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ):
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            family = font_manager.FontProperties(fname=str(font_path)).get_name()
            current = list(plt.rcParams.get("font.sans-serif", []))
            plt.rcParams["font.sans-serif"] = [family] + current
            break


def _soften_color(color: str, white_fraction: float = 0.18) -> tuple[float, float, float]:
    """将类别颜色适度向白色混合。"""
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return tuple(rgb * (1.0 - white_fraction) + white_fraction)


def _case_order_by_age(case_results: dict[str, pd.DataFrame]) -> list[str]:
    """按案例代表年龄从老到新排序。"""
    ordered = sorted(CASE_STUDIES_ORDER, key=lambda item: item[2], reverse=True)
    return [label for label, _, _ in ordered if label in case_results]


def _load_case_results() -> dict[str, pd.DataFrame]:
    """读取六个案例预测结果。"""
    results: dict[str, pd.DataFrame] = {}
    for case_label, _, _ in CASE_STUDIES_ORDER:
        input_path = CASE_PREDICTION_PATHS[case_label]
        if not input_path.exists():
            print(f"[警告] 缺少案例预测文件: {input_path}")
            continue
        results[case_label] = pd.read_csv(
            input_path,
            encoding="utf-8-sig",
            low_memory=False,
        )
    return results


def _draw_compact_legend(fig: plt.Figure) -> None:
    """绘制仅包含颜色和简称的紧凑图例。"""
    abbrev_to_class = {abbr: name for name, abbr in CLASS_ABBREVS.items()}
    handles = [
        mpatches.Patch(
            facecolor=_soften_color(CLASS_COLORS[abbrev_to_class[abbr]]),
            edgecolor="none",
            label=abbr,
        )
        for abbr in LEGEND_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.008),
        ncol=9,
        frameon=False,
        fontsize=11,
        handlelength=1.15,
        handleheight=0.8,
        handletextpad=0.35,
        columnspacing=0.9,
        borderaxespad=0.0,
    )


def plot_six_case_horizontal_bars(
    case_results: dict[str, pd.DataFrame],
    output_path: Path = BARS_OUTPUT_PATH,
) -> None:
    """绘制六联横向柱状图，比较各案例构造环境组成。"""
    cases = _case_order_by_age(case_results)
    fig, axes = plt.subplots(3, 2, figsize=(8.6, 9.2))

    for ax, case_label in zip(axes.flat, cases):
        data = case_results[case_label]
        counts = data["pred_class_name"].value_counts()
        classes = [name for name in counts.index if int(counts[name]) > 0]
        values = np.asarray([int(counts[name]) for name in classes])
        labels = [CLASS_ABBREVS.get(name, name) for name in classes]
        colors = [_soften_color(CLASS_COLORS.get(name, "#888888")) for name in classes]

        y_positions = np.arange(len(classes))
        bars = ax.barh(
            y_positions,
            values,
            height=0.62,
            color=colors,
            edgecolor="white",
            linewidth=0.55,
        )
        max_value = max(int(values.max()), 1)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + max_value * 0.018,
                bar.get_y() + bar.get_height() / 2.0,
                str(int(value)),
                ha="left",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="#303030",
            )

        arc_count = int(data["pred_class_name"].isin(ARC_RELATED_LABELS).sum())
        arc_percent = 100.0 * arc_count / len(data)
        age_ga = dict(
            (label, age) for label, _, age in CASE_STUDIES_ORDER
        )[case_label]
        ax.set_title(
            f"{CASE_STUDY_TITLES.get(case_label, case_label)}"
            f"  (~{age_ga:g} Ga; n={len(data)}; arc={arc_percent:.0f}%)",
            loc="left",
            fontsize=12.5,
            pad=4,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=10.5, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, max_value * 1.02)
        ax.set_xlabel("Sample count")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes.flat[len(cases):]:
        ax.set_axis_off()

    _draw_compact_legend(fig)
    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        top=0.975,
        bottom=0.075,
        hspace=0.34,
        wspace=0.25,
    )
    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[六联柱状图] {output_path}")


def _case_age_ga(data: pd.DataFrame) -> pd.Series:
    """提取样品年龄，优先C_AGE，回退AGE，单位转换为Ga。"""
    age = pd.Series(np.nan, index=data.index, dtype=float)
    if "C_AGE" in data.columns:
        age = pd.to_numeric(data["C_AGE"], errors="coerce")
    if "AGE" in data.columns:
        age = age.fillna(pd.to_numeric(data["AGE"], errors="coerce"))
    return age / 1000.0


def _high_arc_mask(data: pd.DataFrame) -> pd.Series:
    """识别三类弧预测或Arc_probability3不低于0.5的样品。"""
    by_class = data["pred_class_name"].isin(ARC_RELATED_LABELS)
    arc_probability = pd.to_numeric(
        data.get("Arc_probability3", np.nan),
        errors="coerce",
    )
    return by_class | arc_probability.ge(0.5)


def _gaussian_density(ages: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """使用固定带宽计算年龄核密度形状。"""
    density = np.zeros_like(grid)
    for age in ages:
        density += np.exp(
            -((grid - age) ** 2) / (2.0 * RIDGE_KDE_BANDWIDTH_GA ** 2)
        )
    return density


def plot_high_arc_ridgeline(
    case_results: dict[str, pd.DataFrame],
    output_path: Path = RIDGELINE_OUTPUT_PATH,
) -> None:
    """按区域代表年龄从老到新绘制高弧样品KDE山脊图。"""
    cases = _case_order_by_age(case_results)
    age_by_case = {label: age for label, _, age in CASE_STUDIES_ORDER}
    grid = np.linspace(RIDGE_X_MIN_GA, RIDGE_X_MAX_GA, 600)

    fig = plt.figure(figsize=(11.0, 4.2))
    ax = fig.add_axes([0.14, 0.13, 0.82, 0.68])

    for period in RIDGE_PERIODS:
        x_low = min(period["start"], period["end"])
        x_high = max(period["start"], period["end"])
        ax.axvspan(
            x_low,
            x_high,
            color=period["color"],
            alpha=0.32,
            linewidth=0,
            zorder=0,
        )

    for row_index, case_label in enumerate(cases):
        baseline = (len(cases) - 1) - row_index
        data = case_results[case_label]
        arc_mask = _high_arc_mask(data)
        ages = _case_age_ga(data)[arc_mask].dropna().to_numpy(dtype=float)
        ages = ages[np.isfinite(ages)]

        ax.hlines(
            baseline,
            RIDGE_X_MIN_GA,
            RIDGE_X_MAX_GA,
            color="#8A8A8A",
            linewidth=0.55,
            zorder=2,
        )
        density = (
            _gaussian_density(ages, grid)
            if ages.size
            else _gaussian_density(
                np.asarray([age_by_case[case_label]]),
                grid,
            )
        )
        if density.max() > 0:
            curve = baseline + RIDGE_LANE_FILL * density / density.max()
            ax.fill_between(
                grid,
                baseline,
                curve,
                color="#F7CFA5",
                alpha=0.72,
                linewidth=0,
                zorder=3,
            )
            ax.plot(grid, curve, color="#C94A34", linewidth=1.15, zorder=4)

        if 0 < ages.size <= RIDGE_RUG_MAX_N:
            ax.vlines(
                ages,
                baseline - 0.16,
                baseline - 0.045,
                color="#C94A34",
                linewidth=0.6,
                alpha=0.7,
                zorder=3,
            )

    ax.set_xlim(RIDGE_X_MAX_GA, RIDGE_X_MIN_GA)
    ax.set_ylim(-0.55, len(cases) - 1 + RIDGE_LANE_FILL + 0.55)
    ax.set_xticks(np.arange(2.5, 4.01, 0.1))
    ax.set_xlabel("Age (Ga)")
    ax.set_yticks([(len(cases) - 1) - index for index in range(len(cases))])
    ax.set_yticklabels(
        [
            f"{CASE_STUDY_TITLES.get(case, case)}  "
            f"(~{age_by_case[case]:g} Ga)"
            for case in cases
        ],
        fontsize=9.5,
        fontweight="bold",
    )
    ax.tick_params(axis="y", length=0, pad=6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    transform = ax.get_xaxis_transform()
    for period in RIDGE_PERIODS:
        x_low = min(period["start"], period["end"])
        x_high = max(period["start"], period["end"])
        ax.add_patch(
            mpatches.Rectangle(
                (x_low, 1.01),
                x_high - x_low,
                0.05,
                transform=transform,
                facecolor=period["color"],
                edgecolor="white",
                linewidth=0.8,
                clip_on=False,
                zorder=5,
            )
        )
        ax.text(
            (x_low + x_high) / 2.0,
            1.075,
            f"{period['name']}\n{x_high:.1f}-{max(x_low, 2.5):.1f} Ga",
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            linespacing=1.0,
        )

    fig.savefig(output_path, dpi=600, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[高弧KDE山脊图] {output_path}")


def plot_case_studies_bars_ridgeline(
    case_results: dict[str, pd.DataFrame],
    output_path: Path = COMBINED_OUTPUT_PATH,
) -> None:
    """绘制左侧六联柱状图、右侧高弧KDE山脊图的双栏主图。"""
    cases = _case_order_by_age(case_results)
    age_by_case = {label: age for label, _, age in CASE_STUDIES_ORDER}

    fig = plt.figure(figsize=(13.2, 7.6))
    outer_grid = fig.add_gridspec(
        1,
        2,
        width_ratios=[0.8, 0.86],
        left=0.045,
        right=0.985,
        top=0.915,
        bottom=0.085,
        wspace=0.15,
    )

    # 中文注释：左栏使用3行2列，紧凑展示六个案例区的类别组成。
    left_grid = outer_grid[0, 0].subgridspec(3, 2, hspace=0.36, wspace=0.15)
    bar_axes = [fig.add_subplot(left_grid[row, col]) for row in range(3) for col in range(2)]

    for axis_index, (ax, case_label) in enumerate(zip(bar_axes, cases)):
        data = case_results[case_label]
        counts = data["pred_class_name"].value_counts()
        classes = [name for name in counts.index if int(counts[name]) > 0]
        values = np.asarray([int(counts[name]) for name in classes])
        labels = [CLASS_ABBREVS.get(name, name) for name in classes]
        colors = [_soften_color(CLASS_COLORS.get(name, "#888888")) for name in classes]

        y_positions = np.arange(len(classes))
        bars = ax.barh(
            y_positions,
            values,
            height=0.57,
            color=colors,
            edgecolor="white",
            linewidth=0.45,
        )
        max_value = max(int(values.max()), 1)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_width() + max_value * 0.012,
                bar.get_y() + bar.get_height() / 2.0,
                str(int(value)),
                ha="left",
                va="center",
                fontsize=8.2,
                fontweight="bold",
                color="#303030",
            )

        arc_count = int(data["pred_class_name"].isin(ARC_RELATED_LABELS).sum())
        arc_percent = 100.0 * arc_count / len(data)
        age_ga = age_by_case[case_label]
        # 中文注释：标题拆为区域名和灰色统计信息两行，增强层级。
        ax.text(
            0.0,
            1.105,
            CASE_STUDY_TITLES.get(case_label, case_label),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9.6,
            fontweight="bold",
            color="#202020",
            clip_on=False,
        )
        ax.text(
            0.0,
            1.02,
            f"~{age_ga:g} Ga, n = {len(data)}, "
            f"arc-related = {arc_percent:.0f}%",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8.2,
            color="#6A6A6A",
            clip_on=False,
        )
        ax.set_yticks(y_positions)
        ax.set_yticklabels(labels, fontsize=8.5, fontweight="bold")
        ax.invert_yaxis()
        ax.set_xlim(0, max_value * 1.10)
        ax.set_xlabel("Sample count" if axis_index >= 4 else "", fontsize=8.8)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", length=0, pad=5)
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in bar_axes[len(cases):]:
        ax.set_axis_off()

    # 中文注释：右栏集中展示各区域高弧样品的年龄分布。
    ridge_ax = fig.add_subplot(outer_grid[0, 1])
    # 中文注释：左栏标题位于坐标轴外，因此单独抬高右栏顶边与标题首行对齐。
    ridge_position = ridge_ax.get_position()
    first_bar_position = bar_axes[0].get_position()
    aligned_ridge_top = min(
        0.975,
        first_bar_position.y1 + first_bar_position.height * 0.14,
    )
    ridge_ax.set_position(
        [
            ridge_position.x0,
            ridge_position.y0,
            ridge_position.width,
            aligned_ridge_top - ridge_position.y0,
        ]
    )
    grid = np.linspace(RIDGE_X_MIN_GA, RIDGE_X_MAX_GA, 600)
    for period in RIDGE_PERIODS:
        x_low = min(period["start"], period["end"])
        x_high = max(period["start"], period["end"])
        ridge_ax.axvspan(
            x_low,
            x_high,
            color=period["color"],
            alpha=0.32,
            linewidth=0,
            zorder=0,
        )

    for row_index, case_label in enumerate(cases):
        baseline = (len(cases) - 1) - row_index
        data = case_results[case_label]
        ages = _case_age_ga(data)[_high_arc_mask(data)].dropna().to_numpy(dtype=float)
        ages = ages[np.isfinite(ages)]

        ridge_ax.hlines(
            baseline,
            RIDGE_X_MIN_GA,
            RIDGE_X_MAX_GA,
            color="#8A8A8A",
            linewidth=0.55,
            zorder=2,
        )
        density = (
            _gaussian_density(ages, grid)
            if ages.size
            else _gaussian_density(np.asarray([age_by_case[case_label]]), grid)
        )
        if density.max() > 0:
            curve = baseline + RIDGE_LANE_FILL * density / density.max()
            ridge_ax.fill_between(
                grid,
                baseline,
                curve,
                color=RIDGE_FILL_COLOR,
                alpha=0.48,
                linewidth=0,
                zorder=3,
            )
            ridge_ax.plot(
                grid,
                curve,
                color=RIDGE_LINE_COLOR,
                linewidth=1.0,
                zorder=4,
            )

    ridge_ax.set_xlim(RIDGE_X_MAX_GA, RIDGE_X_MIN_GA)
    ridge_y_min = -0.22
    period_bar_bottom = len(cases) - 1 + RIDGE_LANE_FILL + 0.15
    period_bar_height = 0.30
    ridge_y_max = period_bar_bottom + period_bar_height
    ridge_ax.set_ylim(ridge_y_min, ridge_y_max)
    ridge_ax.set_xticks(np.arange(2.6, 4.01, 0.2))
    ridge_ax.set_xlabel("Age (Ga)")
    ridge_label_positions = []
    for index, case in enumerate(cases):
        baseline = (len(cases) - 1) - index
        # 中文注释：标签位于相邻两条横线之间；Isua位于顶部年代条与首条横线之间。
        if case == "Isua":
            label_y = (baseline + period_bar_bottom) / 2.0
        else:
            label_y = baseline + 0.5
        ridge_label_positions.append(label_y)
    ridge_ax.set_yticks(ridge_label_positions)
    ridge_ax.set_yticklabels(
        [
            (
                "Norseman-\nKambalda"
                if case == "Norseman_Kambalda"
                else (
                    "North China\nCraton"
                    if case == "North_China_Craton"
                    else CASE_STUDY_TITLES.get(case, case)
                )
            )
            for case in cases
        ],
        fontsize=9.2,
        fontweight="bold",
    )
    ridge_ax.tick_params(axis="x", labelsize=8.5)
    ridge_ax.tick_params(axis="y", length=0, pad=6)
    ridge_ax.spines["top"].set_visible(False)
    ridge_ax.spines["right"].set_visible(False)
    # 中文注释：平直端帽避免左轴在左下角越过x轴，边界与绘图区严格一致。
    ridge_ax.spines["left"].set_bounds(ridge_y_min, ridge_y_max)
    ridge_ax.spines["left"].set_capstyle("butt")
    ridge_ax.spines["bottom"].set_capstyle("butt")
    ridge_ax.grid(False)

    # 中文注释：年代条放入坐标轴内部，利用第一条KDE上方空间。
    for period_index, period in enumerate(RIDGE_PERIODS):
        x_low = min(period["start"], period["end"])
        x_high = max(period["start"], period["end"])
        # 中文注释：最左侧年代块略微内缩，避免与y轴竖线重合。
        rectangle_x_high = x_high - 0.003 if period_index == 0 else x_high
        ridge_ax.add_patch(
            mpatches.Rectangle(
                (x_low, period_bar_bottom),
                rectangle_x_high - x_low,
                period_bar_height,
                facecolor=period["color"],
                edgecolor="white",
                linewidth=0.8,
                clip_on=True,
                zorder=5,
            )
        )
        ridge_ax.text(
            (x_low + x_high) / 2.0,
            period_bar_bottom + period_bar_height / 2.0,
            period["name"],
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            color="#202020",
            clip_on=True,
            zorder=6,
        )

    # 中文注释：分图编号相对于两栏绘图区定位。
    fig.text(0.012, 0.955, "(a)", fontsize=17, fontweight="bold", va="top")
    updated_ridge_position = ridge_ax.get_position()
    fig.text(
        updated_ridge_position.x0 - 0.012,
        updated_ridge_position.y1,
        "(b)",
        fontsize=17,
        fontweight="bold",
        ha="right",
        va="top",
    )

    fig.savefig(output_path, dpi=1200, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"[左右双栏主图] {output_path}")


def main() -> None:
    """默认输出左右双栏组合主图。"""
    _configure_chinese_font()
    case_results = _load_case_results()
    if not case_results:
        raise FileNotFoundError("没有找到可用的六案例预测CSV")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(SCI_RC):
        plot_case_studies_bars_ridgeline(case_results)


def plot_case_studies_map_ridgeline(
    case_results: dict[str, pd.DataFrame],
    class_names: list[str],
    output_path: Path,
    **_: object,
) -> None:
    """兼容旧调用：不再绘制地图，只输出六联柱图和年龄山脊图。"""
    del class_names, output_path
    plot_six_case_horizontal_bars(case_results)
    plot_high_arc_ridgeline(case_results)


if __name__ == "__main__":
    main()
