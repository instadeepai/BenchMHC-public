"""Module to define the command line to generate performance plots."""

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.scale as scale
import numpy as np
import polars as pl
import scipy.stats as st
import seaborn as sns
from matplotlib.ticker import FormatStrFormatter

from bench_mhc.utils.io import load_yml
from bench_mhc.utils.logging import system

log = system.get(__name__)

BLUE_DARK = "#1B2838"
GREY_DARK = "#747473"


def generate_performance_plot(
    configuration_file_path: Path,
    output_directory: Path,
    cd8_epitopes: bool,
    ms_ligands: bool,
) -> None:
    """Generate performance plots from metrics files.

    Refer to bench_mhc.cli.generate_performance_plot_command.py::generate_performance_plot
    for the complete documentation.

    Args:
        configuration_file_path: Path to the configuration file.
        output_directory: Directory where to save the plots.
        cd8_epitopes: Whether CD8 epitopes data is the input.
        ms_ligands: Whether MS ligands data is the input.

    Raises:
        ValueError: If both CD8 epitopes and MS ligands data are provided.
        ValueError: If neither CD8 epitopes nor MS ligands data are provided.
    """
    if cd8_epitopes and ms_ligands:
        raise ValueError(
            "You should provide either the --cd8_epitopes or --ms_ligands flag, but not both."
        )
    if not cd8_epitopes and not ms_ligands:
        raise ValueError(
            "You should provide exactly one of the --cd8_epitopes or --ms_ligands flag."
        )

    # Save configuration file.
    output_configuration_path = output_directory / "configuration.yml"
    shutil.copy(configuration_file_path, output_configuration_path)

    log.info(f"Loading configuration from '{configuration_file_path}'...")
    configuration = load_yml(configuration_file_path)

    # Load and combine all metrics files.
    metrics_dfs = []
    for model_name, model_config in configuration.items():
        metrics_path = model_config["metrics_path"]
        log.info(f"Loading metrics for model '{model_name}' from '{metrics_path}'...")
        df = pl.read_csv(metrics_path).with_columns(pl.lit(model_name).alias("model"))
        metrics_dfs.append(df)

    df_metrics = pl.concat(metrics_dfs)

    # Get base colors from seaborn's bright palette.
    # Count only models that are not bright versions of others.
    base_model_count = sum(
        1 for model_config in configuration.values() if "bright_version_of" not in model_config
    )
    base_colors = sns.color_palette("bright", n_colors=base_model_count)

    # Create color mapping.
    model_name2color: dict[str, tuple[float, float, float]] = {}
    color_idx = 0
    for model_name, model_config in configuration.items():
        if "bright_version_of" in model_config:
            # This is a bright version of another model already in the mapping.
            base_model = model_config["bright_version_of"]
            base_color = model_name2color[base_model]
            model_name2color[model_name] = sns.light_palette(base_color, n_colors=10)[5]
        else:
            # This is a base model.
            model_name2color[model_name] = base_colors[color_idx]
            color_idx += 1

    model_order = list(configuration.keys())
    color_palette = [model_name2color[model_name] for model_name in model_order]

    if ms_ligands:
        _plot_ms_ligands(df_metrics, model_order, color_palette, output_directory)

    if cd8_epitopes:
        _plot_cd8_epitopes(df_metrics, model_order, color_palette, output_directory)


def _plot_ms_ligands(
    df_metrics: pl.DataFrame,
    model_order: list[str],
    color_palette: list[tuple[float, float, float]],
    output_directory: Path,
) -> None:
    """Generate MS ligands' performance plots.

    This function creates two plots for MS ligands data:
    1. A bar plot showing Top-K scores per allele for each model.
    2. A box plot showing the distribution of Top-K scores across all alleles for each model.

    The function saves two files:
    - ms_ligands_per_allele_bar_plot.png: Bar plot of Top-K scores per allele.
    - ms_ligands_per_allele_box_plot.png: Box plot of Top-K score distributions.

    Args:
        df_metrics: DataFrame containing the metrics data with columns 'allele', 'Top-K',
            and 'model'.
        model_order: List of model names in the desired order for plotting.
        color_palette: List of RGB tuples defining the color for each model.
        output_directory: Directory where the plots will be saved.
    """
    log.info("Generating bar plot...")
    plt.figure(figsize=(min(40, max(20, len(model_order))), 12), facecolor="w", edgecolor="k")

    sns.barplot(
        x="allele",
        y="Top-K",
        hue="model",
        hue_order=model_order,
        palette=color_palette,
        data=df_metrics,
    )

    plt.legend(loc="lower left", fontsize=20)
    plt.xticks(rotation=90)
    plt.xlabel("Alleles", fontsize=30)
    plt.ylabel("Top-K", fontsize=30)
    plt.tight_layout()

    bar_plot_path = output_directory / "ms_ligands_per_allele_bar_plot.png"
    plt.savefig(bar_plot_path)
    log.info(f"Bar plot saved to '{bar_plot_path}'.")

    # Create box plot.
    log.info("Generating box plot...")
    num_models = df_metrics.select(pl.col("model").n_unique()).item()
    plt.figure(figsize=(20, 1 + num_models / 2), facecolor="w", edgecolor="k")
    sns.set(font_scale=1)
    sns.set_style("whitegrid")

    ax = sns.boxplot(
        y="model",
        x="Top-K",
        data=df_metrics,
        order=model_order,
        hue="model",
        legend=False,
        palette=color_palette,
        showmeans=True,
        meanprops={
            "marker": "s",
            "markerfacecolor": "firebrick",
            "markeredgecolor": "black",
            "markersize": "10",
        },
    )

    # Make box plots semi-transparent.
    for patch in ax.patches:
        r, g, b = patch.get_facecolor()[:3]  # type: ignore
        patch.set_facecolor((r, g, b, 0.5))

    sns.swarmplot(y="model", x="Top-K", data=df_metrics, color=".25", s=4)

    plt.xlim((-0.000001, 1.01))
    plt.xticks(rotation=90)
    plt.xlabel("Top-K", fontsize=20, fontweight="bold")
    plt.ylabel("Models", fontsize=20, fontweight="bold")
    plt.tight_layout()

    box_plot_path = output_directory / "ms_ligands_per_allele_box_plot.png"
    plt.savefig(box_plot_path)
    log.info(f"Box plot saved to '{box_plot_path}'.")


def _plot_cd8_epitopes(
    df_metrics: pl.DataFrame,
    model_order: list[str],
    color_palette: list[tuple[float, float, float]],
    output_directory: Path,
) -> None:
    """Generate CD8 epitopes' performance plots.

    This function creates a box plot showing the distribution of Frank scores for each model,
    including mean, median, and number of perfect scores (Frank score = 0). The plot includes
    jittered scatter points for individual scores and annotations for key statistics.

    The function saves one file:
    - cd8_epitopes_per_epitope_box_plot.png: Box plot of Frank score distributions
        with statistical annotations.

    Args:
        df_metrics: DataFrame containing the metrics data with columns 'model', 'Frank Score',
            and 'epitope'.
        model_order: List of model names in the desired order for plotting.
        color_palette: List of RGB tuples defining the color for each model.
        output_directory: Directory where the plot will be saved.
    """
    log.info("Generating CD8 performance plot...")
    # -2 because we have 2 lines for mean and median.
    nums_epitopes = df_metrics.select(pl.col("epitope").n_unique()).item() - 2

    nums_perfect_frank_scores = [
        df_metrics.select((pl.col("model") == model_name) & (pl.col("Frank Score") == 0))
        .sum()
        .item()
        for model_name in model_order
    ]

    frank_scores_per_model = [
        df_metrics.filter(pl.col("model") == model).select("Frank Score")["Frank Score"].to_numpy()
        for model in model_order
    ]

    jitter = 0.05
    x_data = [
        np.array([i for _ in range(len(frank_scores))])
        for i, frank_scores in enumerate(frank_scores_per_model)
    ]
    x_jittered = [x + st.t(df=10, scale=jitter).rvs(len(x)) for x in x_data]

    num_models = len(model_order)
    plt.figure(figsize=(20, 1 + num_models / 2), facecolor="w", edgecolor="k")

    medianprops = {"linewidth": 2, "color": GREY_DARK, "solid_capstyle": "butt"}
    boxprops = {"linewidth": 2, "color": GREY_DARK}

    ax = plt.gca()
    ax.boxplot(
        frank_scores_per_model,
        positions=list(range(len(model_order))),
        showfliers=False,
        showcaps=True,
        medianprops=medianprops,
        whiskerprops=boxprops,
        boxprops=boxprops,
    )

    ax.set_xticklabels(
        ["\n".join(name.split(" - ")) for name in model_order],
        rotation=0,
        fontsize=10,
    )

    for x, y, color in zip(x_jittered, frank_scores_per_model, color_palette, strict=False):
        ax.scatter(x, y, s=8, color=color, alpha=0.5)

    means = [y.mean() for y in frank_scores_per_model]
    medians = [np.median(y) for y in frank_scores_per_model]

    offset = 0.1

    for i, (mean, median, num_perfect_frank_scores) in enumerate(
        zip(means, medians, nums_perfect_frank_scores, strict=False)
    ):
        ax.scatter(i, mean, s=50, color=BLUE_DARK, zorder=3)
        ax.scatter(i, median, s=50, color=BLUE_DARK, zorder=3)

        ax.plot([i, i + offset], [mean, mean], ls="dashdot", color=BLUE_DARK, zorder=3)
        ax.plot([i, i + offset], [median, median], ls="dashdot", color=BLUE_DARK, zorder=3)

        mean_label = r"$Mean = $" if i == 0 else ""
        ax.text(
            i + offset,
            mean,
            mean_label + str(round(mean, 4)),
            fontsize=12,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round", "pad": 0.15},
            zorder=10,
        )

        median_label = r"$Median = $" if i == 0 else ""
        position = median - 0.001 if i == 0 else median
        ax.text(
            i + offset,
            position,
            median_label + str(round(median, 4)),
            fontsize=12,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round", "pad": 0.15},
            zorder=10,
        )
        ax.plot([i, i + offset], [median, position], ls="dashdot", color=BLUE_DARK, zorder=3)

        perfect_scores_label = r"$ \#(Frank\_scores = 0) = $" if i == 0 else ""
        position = 0.00004 if i == 0 else 0.00001
        ax.text(
            i + offset,
            position,
            perfect_scores_label + str(round(num_perfect_frank_scores, 4)),
            fontsize=12,
            va="center",
            bbox={"facecolor": "white", "edgecolor": "black", "boxstyle": "round", "pad": 0.15},
            zorder=10,
        )
        ax.scatter(i, 0, s=50, color=BLUE_DARK, zorder=3)
        ax.plot([i, i + offset], [0, position], ls="dashdot", color=BLUE_DARK, zorder=3)

    ax.set_yscale(
        # Argument 1 to "SymmetricalLogScale" has incompatible type "Axes";
        # expected "Axis | None"  [arg-type].
        scale.SymmetricalLogScale(ax, linthresh=0.0001),  # type: ignore
        subs=[0.2, 0.3, 4, 5, 6, 7, 8, 9],
        linscale=1,
    )
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.set_ylim(-0.000001, 1)

    plt.title(
        f"Per epitope FRANK scores (log scale for scores > 1e-5) \n "
        f"on CD8 epitopes ({nums_epitopes} epitopes)",
        fontsize=18,
    )
    plt.ylabel("FRANK scores (log scale)", fontsize=16)
    plt.xlabel("Models", fontsize=16)
    plt.tight_layout()

    frank_plot_path = output_directory / "cd8_epitopes_per_epitope_box_plot.png"
    plt.savefig(frank_plot_path)
    log.info(f"FRANK score plot saved to '{frank_plot_path}'.")
