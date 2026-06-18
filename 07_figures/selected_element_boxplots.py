import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import TRAIN_RAW_CSV, COMBINED_CSV, FIGURES_DIR


# 默认展示最新版训练流程中“插补前”的训练集实测数据。
# 箱线图用于描述真实观测分布，不使用随机森林插补值、SMOTE 合成值或归一化值。
DATA_SCOPE = "train"

# 数据路径来自 config/paths.py：train=切分后训练集（插补前），full=合并总表。
DATA_PATHS = {
    "train": Path(str(TRAIN_RAW_CSV)),
    "full": Path(str(COMBINED_CSV)),
}

# 输出统一放在 data/figures/selected_elements/ 下。
_BOX_OUTPUT_DIR = FIGURES_DIR / "selected_elements"
OUTPUT_PATHS = {
    "train": _BOX_OUTPUT_DIR / "selected_element_boxplots_train_observed.png",
    "full": _BOX_OUTPUT_DIR / "selected_element_boxplots_full_observed.png",
}

# 需要绘制的元素列名及其图中显示名称（按面板 a–f 顺序排列）。
ELEMENT_LABELS = {
    "BA(PPM)": "Ba (ppm)",
    "TH(PPM)": "Th (ppm)",
    "NB(PPM)": "Nb (ppm)",
    "LA(PPM)": "La (ppm)",
    "DY(PPM)": "Dy (ppm)",
    "TIO2(WT%)": r"TiO$_2$ (wt%)",
}

# 各元素的纵轴尺度与刻度设置。
# Ba、Th、Nb、La 分布明显右偏、高值离群点多，改用 log 尺度并指定规范刻度（不用科学计数法）；
# Dy、TiO2 分布较集中，继续使用线性尺度，刻度交给 MaxNLocator 自动生成。
ELEMENT_SCALES = {
    "BA(PPM)": ("log", [1, 10, 100, 1000, 10000]),
    "TH(PPM)": ("log", [0.01, 0.1, 1, 10, 100]),
    "NB(PPM)": ("log", [0.1, 1, 10, 100, 1000]),
    "LA(PPM)": ("log", [1, 10, 100, 1000]),
    "DY(PPM)": ("linear", None),
    "TIO2(WT%)": ("linear", None),
}

# Ba、Th、Nb 含有极少数超高值，若直接用最大值确定上界会产生大片空白。
# 仅裁剪显示范围，不删除数据；箱体统计量仍基于全部有效实测值。
LOG_Y_LIMIT_QUANTILES = {
    "BA(PPM)": 0.998,
    "TH(PPM)": 0.998,
    "NB(PPM)": 0.998,
}
# 对数轴顶部统一增加 0.18 个数量级，避免最高散点紧贴面板边界。
LOG_Y_LIMIT_PADDING_DECADES = 0.28

# Dy 和 TiO2 的少数高值会把线性轴上限撑得偏高，这里只收紧主分布展示范围。
LINEAR_Y_LIMIT_QUANTILES = {
    "DY(PPM)": 0.998,
    "TIO2(WT%)": 0.998,
}
LINEAR_Y_LIMIT_PADDING = 1.05

# 将原始构造环境名称统一转换为论文图中更简洁的缩写。
TECTONIC_ABBR = {
    "Continental arc": "CA",
    "Island arc": "IA",
    "Intra-oceanic arc": "IOA",
    "BACK-ARC_BASIN": "BAB",
    "SPREADING_CENTER": "MOR",
    "OCEANIC PLATEAU": "OP",
    "OCEAN ISLAND": "OI",
    "CONTINENTAL FLOOD BASALT": "CF",
    "CONTINENTAL_RIFT": "CR",
    "CA": "CA",
    "IA": "IA",
    "IOA": "IOA",
    "BAB": "BAB",
    "MOR": "MOR",
    "OP": "OP",
    "OI": "OI",
    "CF": "CF",
    "CFB": "CF",
    "CR": "CR",
}

# 固定横轴类别顺序：洋中脊 → 弧后 → 弧相关 → 板内/地幔柱相关，体现地质过程逻辑。
# 所有子图共享同一顺序，便于跨子图直接比较。
TECTONIC_ORDER = ["MOR", "BAB", "IOA", "IA", "CA", "CR", "CF", "OP", "OI"]

# 每个构造环境对应一种固定颜色：低饱和、偏灰、分组一致的 Nature 风格配色。
TECTONIC_COLORS = {
    "MOR": "#7FB7B2",  # 灰青色
    "BAB": "#88A9C3",  # 灰蓝色
    "IOA": "#E6B377",  # 柔和橙黄
    "IA": "#EDB67A",   # 浅暖橙
    "CA": "#E98C68",   # 珊瑚橙
    "CR": "#C9B383",   # 卡其棕
    "CF": "#B7AA84",   # 橄榄卡其
    "OP": "#B7A1C9",   # 灰紫色
    "OI": "#C8B9D8",   # 浅紫灰
}

# jitter 散点叠加参数：散点仅用于展示分布形态，箱体统计量始终基于全部数据。
# 为避免大类别（上万样本）散点过于拥挤，每个类别最多随机抽取 JITTER_MAX_POINTS 个点叠加显示。
JITTER_WIDTH = 0.18       # x 方向抖动幅度
JITTER_MAX_POINTS = 300   # 每个类别叠加散点的上限
SCATTER_SIZE = 3.5        # 散点大小（较小）
SCATTER_ALPHA = 0.28      # 散点透明度（较低，比箱体更透明）
BOX_ALPHA = 0.85          # 箱体填充透明度
JITTER_SEED = 0           # 抽样与抖动的随机种子，保证可复现


def load_dataset(scope: str) -> pd.DataFrame:
    """读取最新版流程中插补前的训练集或划分前总表。"""
    return pd.read_csv(DATA_PATHS[scope], low_memory=False)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """检查必要列、统一构造环境标签，并将元素列转换为数值型。"""
    required_columns = ["TECTONIC SETTING", *ELEMENT_LABELS.keys()]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # 只保留本图需要的列，避免无关列影响后续统计。
    plot_df = df[required_columns].copy()
    plot_df["TECTONIC SETTING"] = plot_df["TECTONIC SETTING"].map(TECTONIC_ABBR)
    # 如果出现未映射的构造环境名称，这里会被置为缺失并移除。
    plot_df = plot_df.dropna(subset=["TECTONIC SETTING"])

    for element in ELEMENT_LABELS:
        # 将异常字符串转为 NaN；绘图时会自动忽略这些缺失值。
        plot_df[element] = pd.to_numeric(plot_df[element], errors="coerce")

    return plot_df


def print_dataset_summary(df: pd.DataFrame, scope: str) -> None:
    """在终端输出样本量、类别数量和绘图元素缺失值，方便论文作图时核对。"""
    print(f"Data scope: {scope}")
    print("Value source: observed values before imputation")
    print(f"Rows: {len(df)}")
    print("\nClass counts:")
    print(df["TECTONIC SETTING"].value_counts().reindex(TECTONIC_ORDER).to_string())
    print("\nMissing values in plotted elements:")
    print(df[list(ELEMENT_LABELS.keys())].isna().sum().to_string())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按构造环境绘制 Ba、Th、Nb、La、Dy、TiO2 箱型图。"
    )
    parser.add_argument(
        "--scope",
        choices=["train", "full"],
        default=DATA_SCOPE,
        help=(
            "train：最新版训练流程中插补前的训练集实测值；"
            "full：划分训练集/测试集之前的完整实测总表。"
        ),
    )
    return parser.parse_args()


def _format_log_tick(value: float) -> str:
    """log 轴刻度标签：使用千分位、避免科学计数法、不保留多余小数。"""
    if value >= 1:
        return f"{value:,.0f}"  # 1、10、100、1,000、10,000
    return f"{value:g}"  # 0.1、0.01


def _format_linear_tick(value: float, _pos: int) -> str:
    """线性轴刻度标签：整数直接显示，必要时保留最少小数位。"""
    if float(value).is_integer():
        return f"{int(value)}"
    return f"{value:g}"


def _style_axis(ax: plt.Axes) -> None:
    """统一子图的 Nature 风格外观：去顶/右边框、细坐标轴、淡水平网格、白底。"""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#333333")
        ax.spines[side].set_linewidth(0.8)

    # 网格仅保留很淡的水平点线，置于数据之下，不抢眼。
    ax.set_axisbelow(True)
    ax.grid(axis="y", which="major", linestyle=(0, (1, 3)),
            linewidth=0.5, color="#cccccc", alpha=0.9)
    ax.grid(axis="x", visible=False)

    ax.tick_params(axis="both", which="major", labelsize=8,
                   width=0.8, length=3, colors="#333333")
    ax.tick_params(axis="both", which="minor", length=0)
    # y轴刻度值承载读数信息，使用半粗体让小字号下更清楚。
    for label in ax.get_yticklabels():
        label.set_fontweight("semibold")
    ax.set_xlabel("")
    ax.set_ylabel("")


def create_selected_element_boxplots(
    df: pd.DataFrame, output_path: Path
) -> Path:
    """生成 2×3 的组合箱线图（叠加 jitter 散点），保存为高分辨率 PNG。"""
    # 清爽的无衬线字体；mathtext 也用无衬线，使面板标号与 TiO2 下标风格统一。
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    panel_letters = "abcdef"
    n_categories = len(TECTONIC_ORDER)
    rng = np.random.default_rng(JITTER_SEED)

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.4), constrained_layout=True)

    for idx, (ax, (column, name)) in enumerate(zip(axes.flat, ELEMENT_LABELS.items())):
        scale, ticks = ELEMENT_SCALES[column]

        positions: list[int] = []
        box_data: list[np.ndarray] = []
        box_colors: list[str] = []

        for pos, category in enumerate(TECTONIC_ORDER, start=1):
            values = df.loc[df["TECTONIC SETTING"] == category, column].dropna()
            if scale == "log":
                # log 尺度无法显示非正值，先剔除 <=0 的数据。
                values = values[values > 0]
            array = values.to_numpy()
            if array.size == 0:
                continue

            color = TECTONIC_COLORS[category]
            positions.append(pos)
            box_data.append(array)
            box_colors.append(color)

            # 半透明 jitter 散点叠加：必要时对大类别抽样，避免过于拥挤。
            if array.size > JITTER_MAX_POINTS:
                sample = array[rng.choice(array.size, JITTER_MAX_POINTS, replace=False)]
            else:
                sample = array
            jitter_x = pos + rng.uniform(-JITTER_WIDTH, JITTER_WIDTH, size=sample.size)
            ax.scatter(
                jitter_x, sample,
                s=SCATTER_SIZE, color=color, alpha=SCATTER_ALPHA,
                linewidths=0, zorder=3, rasterized=True,
            )

        # 关闭默认离群点，箱体/须线/帽线均用细线，中位线略深以突出。
        box = ax.boxplot(
            box_data,
            positions=positions,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            # 须线用 2–98 百分位（而非默认 1.5×IQR），避免个别低值把下须拉得过长，
            # 使 log 子图（尤其 La）底部更贴合主分布；完整分布交给叠加的 jitter 散点展示。
            whis=(2, 98),
            boxprops=dict(linewidth=0.8, edgecolor="#555555"),
            whiskerprops=dict(linewidth=0.8, color="#555555"),
            capprops=dict(linewidth=0.8, color="#555555"),
            medianprops=dict(linewidth=1.2, color="#222222"),
            zorder=2,
        )
        for patch, color in zip(box["boxes"], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(BOX_ALPHA)

        # 纵轴尺度与刻度。
        all_values = np.concatenate(box_data) if box_data else np.array([1.0])
        if scale == "log":
            ax.set_yscale("log")
            # 上界贴合数据最高点，下界对齐箱线须线下端，收紧多余空白；
            # 少数低于须线的离群低散点不显示，与 showfliers=False 行为一致。
            whisker_ydata = np.concatenate([w.get_ydata() for w in box["whiskers"]])
            low = whisker_ydata[whisker_ydata > 0].min()
            if column in LOG_Y_LIMIT_QUANTILES:
                high = np.nanquantile(
                    all_values, LOG_Y_LIMIT_QUANTILES[column]
                )
                high *= 10 ** LOG_Y_LIMIT_PADDING_DECADES
            else:
                high = all_values.max() * 1.3
            ax.set_ylim(low / 1.3, high)
            ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))
            # 用 FuncFormatter 按刻度值格式化：范围外刻度被裁剪时标签也不会错位。
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda value, _pos: _format_log_tick(value))
            )
            ax.yaxis.set_minor_locator(mticker.NullLocator())
        else:
            if column in LINEAR_Y_LIMIT_QUANTILES:
                upper_value = np.nanquantile(
                    all_values, LINEAR_Y_LIMIT_QUANTILES[column]
                )
            else:
                upper_value = all_values.max()
            ax.set_ylim(0, upper_value * LINEAR_Y_LIMIT_PADDING)
            # steps 限定为"整"步长，避免出现 1.5/4.5 这类不规整刻度。
            ax.yaxis.set_major_locator(
                mticker.MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
            )
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_linear_tick))

        # 横轴类别（所有子图保持完全一致的顺序）。
        ax.set_xticks(range(1, n_categories + 1))
        # x轴分类缩写数量较多，稍微加粗可以提高论文图中的可读性。
        ax.set_xticklabels(TECTONIC_ORDER, fontweight="semibold")
        ax.set_xlim(0.4, n_categories + 0.6)

        _style_axis(ax)

        # 左对齐的面板标题：标号 (a)–(f) 加粗，元素名称正常字重。
        title = rf"$\mathbf{{({panel_letters[idx]})}}$  {name}"
        # 放到子图内部左上角空白处，使用轴坐标避免受 constrained_layout 重新排版影响。
        ax.text(
            0.02, 0.98, title,
            transform=ax.transAxes,
            fontweight="semibold", 
            ha="left", va="top",
            fontsize=12, color="#222222",
            zorder=5,
        )

    # 共享坐标轴标题。
    fig.supxlabel("Tectonic setting", fontsize=12, color="#222222", fontweight="semibold")
    fig.supylabel("Concentration", x=0.000, fontsize=12, color="#222222", fontweight="semibold")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # 仅导出高分辨率 PNG（600 dpi），用于论文主文图排版。
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


if __name__ == "__main__":
    # 命令示例：
    # python data_analysis/selected_element_boxplots.py --scope train
    # python data_analysis/selected_element_boxplots.py --scope full
    args = parse_args()
    dataset = load_dataset(args.scope)
    dataset = prepare_dataset(dataset)
    print_dataset_summary(dataset, args.scope)
    output_path = create_selected_element_boxplots(dataset, OUTPUT_PATHS[args.scope])
    print(f"\nBoxplot saved to: {output_path}")
