#!/usr/bin/env python3
# coding: utf-8

# Assessment of feature contributions per staining channel, cell compartment, and feature type (Fig 3D)
#
# Calculate mAP to assess phenotypic activity of each perturbation
#  (retrieval its replicates against negative controls) for different subsets of features.

import pathlib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycytominer.cyto_utils import infer_cp_features

from cell_health_utils import (
    subset_6_replicates,
    get_cell_line_colors,
    get_control_barcodes,
)
from map_utils.map import calculate_map
from map_utils.plot import set_plotting_style, save_plot


def process_channel_analysis(profiles, cell_lines, channels, pair_config, map_config):
    results = []
    for cell in cell_lines:
        print(f"Processing cell type: {cell}")
        df = subset_6_replicates(profiles, cell_line=cell)
        meta_features = df.filter(regex="Metadata_").columns.tolist()
        for channel in channels:
            channel_features = df.loc[
                :, df.columns.str.contains(channel)
            ].columns.tolist()
            for drop in [True, False]:
                if drop:
                    dropped_or_exclusive = "dropped"
                    subset_df = df.drop(channel_features, axis="columns")
                else:
                    dropped_or_exclusive = "exclusive"
                    subset_df = df.loc[:, meta_features + channel_features]
                map_results = calculate_map(subset_df, pair_config, map_config)
                map_results["cell_type"] = cell
                map_results["channel"] = channel
                map_results["dropped_or_exclusive"] = dropped_or_exclusive
                results.append(map_results)
    results = pd.concat(results)
    per_channel_df = results.pivot_table(
        index=["Metadata_pert_name", "cell_type", "channel"],
        values="-log10(mAP p-value)",
        columns="dropped_or_exclusive",
    ).reset_index()
    per_channel_df = (
        per_channel_df.assign(
            channel_signal=per_channel_df["exclusive"] - per_channel_df["dropped"]
        )
        .sort_values(by="channel_signal", ascending=False)
        .reset_index(drop=True)
    )
    return per_channel_df


def plot_channel_analysis(
    per_channel_df, output_dir, height=3, point_size=30, style_kwargs=None
):
    set_plotting_style(**style_kwargs if style_kwargs else {})
    cell_line_colors = get_cell_line_colors()
    p_threshold = -np.log10(0.05)
    g = sns.FacetGrid(
        per_channel_df,
        col="channel",
        hue="cell_type",
        height=height,
        aspect=0.5,
        hue_order=list(cell_line_colors.keys()),
        palette=cell_line_colors,
    )
    g.map(
        sns.scatterplot,
        "dropped",
        "exclusive",
        alpha=0.6,
        s=point_size,
        edgecolor="none",
    )

    def add_diag_line(*args, **kwargs):
        ax = plt.gca()
        ax.axline((0, 0), slope=1, color="red", linestyle="dashed", lw=0.5)
        ax.axhline(p_threshold, color="grey", linestyle="--")
        ax.axvline(p_threshold, color="grey", linestyle="--")

    g.map(add_diag_line)

    channels = per_channel_df["channel"].unique()
    overall_means = per_channel_df.groupby("channel").agg(
        dropped_mean=("dropped", lambda s: (s > p_threshold).mean()),
        exclusive_mean=("exclusive", lambda s: (s > p_threshold).mean()),
    )
    for ax, channel in zip(g.axes.flatten(), channels):
        dropped_percent = overall_means.loc[channel, "dropped_mean"]
        exclusive_percent = overall_means.loc[channel, "exclusive_mean"]
        ax.text(
            0.05,
            0.98,
            f"{exclusive_percent:.0%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.text(
            0.95,
            0.0,
            f"{dropped_percent:.0%}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

    g.set_xlabels("")
    g.set_ylabels("-log10(mAP p-value),\n channel exclusive")
    g.fig.text(0.48, 0.98, "Channel (with % retrieved)", ha="center", va="center")
    g.fig.text(
        0.48, 0.025, "-log10(mAP p-value), channel dropped", ha="center", va="center"
    )
    g.set_titles("{col_name}")
    # g.fig.set_size_inches(13, 2.5)
    g.fig.set_size_inches(3.46, 0.85)
    plt.tight_layout()
    plt.legend(
        title="Cell type",
        bbox_to_anchor=(1.05, 1),
        loc=2,
        borderaxespad=0,
        frameon=False,
    )
    save_plot(g.fig, "Fig3D_channel", output_dir)
    return g


def process_compartment_analysis(profiles, cell_lines, pair_config, map_config):
    compartments = ["Cells", "Cytoplasm", "Nuclei"]
    control_barcodes = get_control_barcodes(profiles)
    df_sample = subset_6_replicates(profiles, cell_line=cell_lines[0])
    all_features = infer_cp_features(df_sample, compartments=compartments)
    feature_groups = [
        "AreaShape",
        "Correlation",
        "Granularity",
        "Intensity",
        "RadialDistribution",
        "Texture",
    ]
    process_features = [
        f for f in all_features if any(f"_{fg}_" in f for fg in feature_groups)
    ]
    feature_group_compartments = list(
        set(["_".join(x.split("_")[0:2]) for x in process_features])
    )

    results = []
    for cell in cell_lines:
        print(f"Processing cell type: {cell}")
        df = subset_6_replicates(profiles, cell_line=cell)
        meta_features = df.filter(regex="Metadata_").columns.tolist()
        for fg_comp in feature_group_compartments:
            try:
                compartment, feature_group = fg_comp.split("_")
            except ValueError:
                continue
            compartment_features = df.loc[
                :, df.columns.str.startswith(fg_comp)
            ].columns.tolist()
            for drop in [True, False]:
                if drop:
                    dropped_or_exclusive = "dropped"
                    subset_df = df.drop(compartment_features, axis="columns")
                else:
                    dropped_or_exclusive = "exclusive"
                    subset_df = df.loc[:, meta_features + compartment_features]
                subset_df["Metadata_is_control"] = (
                    subset_df["Metadata_pert_name"]
                    .isin(control_barcodes["cutting_control"])
                    .astype(int)
                )
                map_results = calculate_map(
                    subset_df.reset_index(drop=True), pair_config, map_config
                )
                map_results["cell_line"] = cell
                map_results["compartment"] = compartment
                map_results["feature_group"] = feature_group
                map_results["dropped_or_exclusive"] = dropped_or_exclusive
                results.append(map_results)
    map_subcompartment_results = pd.concat(results).reset_index(drop=True)
    per_featuregroup_df = map_subcompartment_results.pivot_table(
        index=["Metadata_pert_name", "cell_line", "feature_group", "compartment"],
        values="-log10(mAP p-value)",
        columns="dropped_or_exclusive",
    ).reset_index()
    per_featuregroup_df = (
        per_featuregroup_df.assign(
            channel_signal=per_featuregroup_df["exclusive"]
            - per_featuregroup_df["dropped"]
        )
        .sort_values(by="channel_signal", ascending=False)
        .reset_index(drop=True)
    )
    return per_featuregroup_df


def plot_compartment_analysis(per_featuregroup_df, output_dir, style_kwargs=None):
    def add_diag_line(*args, **kwargs):
        ax = plt.gca()
        ax.axline((0, 0), slope=1, color="red", linestyle="dashed", lw=0.5)
        ax.axhline(-np.log10(0.05), color="grey", linestyle="--")
        ax.axvline(-np.log10(0.05), color="grey", linestyle="--")

    def add_text_annotations(ax, data):
        threshold = -np.log10(0.05)
        dropped_percent = (data["dropped"] > threshold).mean()
        exclusive_percent = (data["exclusive"] > threshold).mean()
        ax.text(
            0.05,
            0.95,
            f"{exclusive_percent:.0%}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )
        ax.text(
            0.95,
            0.05,
            f"{dropped_percent:.0%}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

    set_plotting_style(**style_kwargs if style_kwargs else {})
    cell_line_colors = get_cell_line_colors()
    g = sns.FacetGrid(
        per_featuregroup_df,
        col="feature_group",
        row="compartment",
        hue="cell_line",
        height=3,
        aspect=1.05,
        hue_order=list(cell_line_colors.keys()),
        palette=cell_line_colors,
        margin_titles=True,
    )
    g.map(sns.scatterplot, "dropped", "exclusive", alpha=0.5, s=50, edgecolor="none")
    g.map(add_diag_line)
    for ax, (_, sub_data) in zip(g.axes.flatten(), g.facet_data()):
        add_text_annotations(ax, sub_data)
    g.set_xlabels("")
    g.set_ylabels("")
    g.set_titles(col_template="{col_name}", row_template="{row_name}")
    g.fig.text(0.5, 1.0, "Feature groups (with % retrieved)", ha="center", va="center")
    g.fig.text(
        0.5, 0.0, "-log10(mAP p-value), feature group dropped", ha="center", va="center"
    )
    g.fig.text(1.0, 0.5, "Compartments", ha="center", va="center", rotation=270)
    g.fig.text(
        0.0,
        0.5,
        "-log10(mAP p-value), feature group exclusive",
        ha="center",
        va="center",
        rotation="vertical",
    )
    plt.tight_layout()
    plt.legend(title="Cell type", bbox_to_anchor=(1.85, 2), frameon=False)
    save_plot(g.fig, "Fig3D_compartment", output_dir)
    return g


def main():
    profiles_path = pathlib.Path(
        "outputs/cell_health_profiles_merged_wholeplate_normalized_featureselected.tsv.gz"
    )
    profiles = pd.read_csv(profiles_path, sep="\t")
    print(profiles.shape)

    output_dir = pathlib.Path("outputs")
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    channel_map_results = output_dir / "channel_analysis.csv"
    compartment_map_results = output_dir / "compartment_analysis.csv"

    pair_config = {
        "pos_sameby": ["Metadata_pert_name", "Metadata_control_index"],
        "pos_diffby": [],
        "neg_sameby": [],
        "neg_diffby": ["Metadata_pert_name", "Metadata_control_index"],
    }
    map_config = {"null_size": 1000000, "groupby_columns": ["Metadata_pert_name"]}

    # Process and plot channel analysis
    channels = ["DNA", "RNA", "Mito", "AGP", "ER"]
    cell_lines = list(get_cell_line_colors().keys())
    if channel_map_results.exists():
        print(f"Loading existing channel analysis results from {channel_map_results}")
        per_channel_df = pd.read_csv(channel_map_results)
    else:
        per_channel_df = process_channel_analysis(
            profiles, cell_lines, channels, pair_config, map_config
        )
        per_channel_df.to_csv(
            channel_map_results, index=False
        )

    style = {"font_size": 5, "linewidth": 0.35}
    plot_channel_analysis(
        per_channel_df, figure_dir, height=3, point_size=3, style_kwargs=style
    )

    # Process and plot compartment analysis
    if compartment_map_results.exists():
        print(
            f"Loading existing compartment analysis results from {compartment_map_results}"
        )
        per_featuregroup_df = pd.read_csv(compartment_map_results)
    else:
        per_featuregroup_df = process_compartment_analysis(
            profiles, cell_lines, pair_config, map_config
        )
        per_featuregroup_df.to_csv(
            output_dir / "compartment_analysis.csv", index=False
        )
    style = {"font_size": 16, "linewidth": 1}
    plot_compartment_analysis(per_featuregroup_df, figure_dir, style_kwargs=style)


if __name__ == "__main__":
    main()
