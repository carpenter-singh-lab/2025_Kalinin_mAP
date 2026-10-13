from pathlib import Path
from functools import partial

import numpy as np
import plotnine as gg
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MultipleLocator


def save_plot(
    plot_obj,
    base_name,
    output_dir,
    dpi=500,
    width=8,
    height=4,
    bbox_inches="tight",
    formats="svg,png,pdf",
):
    output_dir.mkdir(parents=True, exist_ok=True)
    for format in formats.split(","):
        if format not in ["svg", "png", "pdf"]:
            raise ValueError(
                f"Unsupported format: {format}. Supported formats are: svg, png, pdf."
            )

        save_path = output_dir / f"{base_name}.{format}"
        # If the plot object has a 'save' method (e.g., plotnine), use it.
        if hasattr(plot_obj, "save"):
            plot_obj.save(
                save_path, dpi=dpi, height=height, width=width, bbox_inches=bbox_inches
            )
        # Otherwise, assume it's a matplotlib figure.
        elif hasattr(plot_obj, "savefig"):
            plot_obj.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
        else:
            raise TypeError(
                "The provided plot object does not have a save or savefig method."
            )
        print(f"Saved plot as: {save_path}")


def x_axis_formatter(x, pos, n_labels=5):
    if pos == 1 or pos == n_labels - 1:
        return f"{int(x)}"
    else:
        return f"{x:.2f}"


def remove_inner_ticklabels(fig: plt.Figure):
    for ax in fig.axes:
        try:
            ax.label_outer()
        except AttributeError:
            pass


def add_corner_text_annotations(
    ax,
    data,
    top="exclusive",
    bottom="dropped",
    prefix="",
    h_offset=0.05,
    v_offset=0.05,
    h_offset_bottom=None,
    v_offset_bottom=None,
):
    threshold = -np.log10(0.05)
    dropped_percent = (data[bottom] > threshold).mean()
    exclusive_percent = (data[top] > threshold).mean()
    ax.text(
        h_offset,
        1 - v_offset,
        f"{prefix}{exclusive_percent:.0%}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    h_offset_bottom = h_offset_bottom if h_offset_bottom is not None else 2 * h_offset
    v_offset_bottom = v_offset_bottom if v_offset_bottom is not None else v_offset
    ax.text(
        1 - h_offset_bottom,
        v_offset_bottom,
        f"{prefix}{dropped_percent:.0%}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )


def set_plotting_style(font_size=16, linewidth=1):
    font_path = Path(__file__).parent / "fonts" / "OpenSans-Regular.ttf"
    font_manager.fontManager.addfont(font_path)
    _ = font_manager.FontProperties(fname=font_path)
    plt.rcParams["font.family"] = ["Open Sans"]
    plt.rcParams["font.size"] = font_size
    plt.rcParams["lines.linewidth"] = linewidth
    plt.rcParams.update(
        {
            "axes.linewidth": linewidth,
            "xtick.major.width": linewidth,
            "ytick.major.width": linewidth,
            "xtick.minor.width": linewidth,
            "ytick.minor.width": linewidth,
            "xtick.major.size": 5 * linewidth,
            "ytick.major.size": 5 * linewidth,
            "xtick.minor.size": 2.5 * linewidth,
            "ytick.minor.size": 2.5 * linewidth,
        }
    )


def format_fig3a(ax_scatter, pr_x, pr_y, frs, palette, fr_total):
    """Apply Fig3A specific formatting to the mAP scatter plot."""
    ax_scatter.text(pr_x - 0.33, pr_y, "Retrieved: ", transform=ax_scatter.transAxes)
    ax_scatter.text(
        0.73,
        0.11,
        "p=0.05",
        transform=ax_scatter.transAxes,
        color="grey",
        # fontsize=12,
        fontsize=5,
        fontstyle="italic",
    )
    for i, (hue_value, fr) in enumerate(frs.items()):
        ax_scatter.text(
            (pr_x - 0.33) + i * 0.175,
            pr_y - 0.09,
            f"{fr:.0%}",
            transform=ax_scatter.transAxes,
            color=palette[hue_value],
        )
    ax_scatter.text(
        (pr_x - 0.33) + len(frs) * 0.175,
        pr_y - 0.09,
        f"({fr_total:.0%})",
        transform=ax_scatter.transAxes,
    )


def format_fig3b(ax_scatter, pr_x, pr_y, frs, palette, fr_total):
    """Apply Fig3B specific formatting to the mAP scatter plot."""
    ax_scatter.text(pr_x, pr_y + 0.07, "Retrieved:", transform=ax_scatter.transAxes)
    ax_scatter.text(
        0.68,
        0.3,
        "p=0.05",
        transform=ax_scatter.transAxes,
        color="grey",
        # fontsize=12,
        fontsize=5,
        fontstyle="italic",
    )
    for i, (hue_value, fr) in enumerate(frs.items()):
        ax_scatter.text(
            (pr_x - 0.33) + i * 0.175,
            pr_y - 0.035,
            f"{fr:.0%}",
            transform=ax_scatter.transAxes,
            color=palette[hue_value],
        )
    ax_scatter.text(
        (pr_x - 0.33) + len(frs) * 0.175,
        pr_y - 0.035,
        f"({fr_total:.0%})",
        transform=ax_scatter.transAxes,
    )


def format_fig3e(ax_scatter, pr_x, pr_y, frs, palette, fr_total):
    """Apply Fig3E specific formatting to the mAP scatter plot."""
    ax_scatter.text(pr_x, pr_y + 0.1, "Retrieved:", transform=ax_scatter.transAxes)
    # ax_scatter.text(
    #     0.82,
    #     pr_y - 0.45,
    #     "p=0.05",
    #     transform=ax_scatter.transAxes,
    #     color="grey",
    #     fontsize=12,
    #     fontstyle="italic",
    # )
    for i, (hue_value, fr) in enumerate(frs.items()):
        ax_scatter.text(
            pr_x + i * 0.175,
            pr_y,
            f"{fr:.0%}",
            transform=ax_scatter.transAxes,
            color=palette[hue_value],
        )
    ax_scatter.text(
        pr_x + len(frs) * 0.175,
        pr_y,
        f"({fr_total:.0%})",
        transform=ax_scatter.transAxes,
    )


def apply_figure_formatting(
    figure, ax_scatter, pr_x, pr_y, frs, palette, fr_total, row_value=None, y_label=None
):
    """Apply formatting specific to different figure types."""
    if figure == "Fig3A":
        format_fig3a(ax_scatter, pr_x, pr_y, frs, palette, fr_total)
        if y_label is not None:
            ax_scatter.set_ylabel(f"Preprocessing: {row_value}\n-log10(mAP p-value)")
    elif figure == "Fig3B":
        format_fig3b(ax_scatter, pr_x, pr_y, frs, palette, fr_total)
    elif figure == "Fig3E":
        format_fig3e(ax_scatter, pr_x, pr_y, frs, palette, fr_total)


def plot_map_scatter_kde(
    df,
    col,
    title,
    metric="mAP",
    row=None,
    hue_col=None,
    palette=None,
    y_label=None,
    aspect=None,
    adjust=None,
    pr_x=0.61,
    pr_y=0.35,
    l_x=1.05,
    l_y=0.575,
    m_x=0.52,
    m_y=0.01,
    kde_y=0.4,
    point_size=50,
    size_rescale=None,
    legend=True,
    legend_frameon=False,
    legend_loc="upper center",
    figure=None,
    save_path=None,
):
    """
    Unified function for plotting mAP data with optional hue grouping.

    Parameters:
    -----------
    df : pandas.DataFrame
        The data to plot
    col : str
        Column to use for subplot columns
    title : str
        Plot title
    metric : str, default="mAP"
        Metric to plot on x-axis
    row : str, optional
        Column to use for subplot rows
    hue_col : str, optional
        Column to use for hue grouping. If None, "p < 0.05" is used as hue
    palette : str or dict, optional
        Color palette to use for hue groups
    move_legend : str, default="lower right"
        Position for the legend
    y_label : str, optional
        Custom y-axis label
    aspect : float, optional
        Aspect ratio of the subplots
    adjust : dict, optional
        Parameters for fig.subplots_adjust()
    pr_x, pr_y : float
        Position for retrieved percentage annotation
    l_x, l_y : float
        Position for legend
    m_x, m_y : float
        Position for metric label
    kde_y : float
        Vertical position for KDE annotations
    figure : str, optional
        Figure identifier for specific formatting (e.g., "Fig3A")
    save_path : str, optional
        Path to save the figure

    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure object
    """
    unique_col_values = df[col].unique()
    num_cols = len(unique_col_values)
    num_rows = 1
    unique_row_values = [None]

    # Determine if we're using hue mode
    use_hue_mode = hue_col is not None

    # Set default hue column if in hue mode
    if use_hue_mode:
        hue_col = hue_col if hue_col is not None else "p < 0.05"
        palette = palette if palette is not None else "Set1"

    # Process row information if provided
    if row is not None and row in df.columns:
        unique_row_values = sorted(df[row].unique())[::-1]
        num_rows = len(unique_row_values)

    # Create figure and axes
    panel_size = 4 if size_rescale is None else 4 * size_rescale
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * panel_size, num_rows * panel_size),
        layout="constrained",
        sharex=True,
        sharey=True,
    )

    for row_i, row_value in enumerate(unique_row_values):
        row_df = df[df[row] == row_value] if row is not None else df
        row_axes = axes[row_i] if num_rows > 1 else axes

        for col_i, col_value in enumerate(unique_col_values):
            sub_df = row_df[row_df[col] == col_value]
            mean_map = sub_df[metric].mean()
            fr_total = sub_df["p < 0.05"].mean()

            # Set up the scatter plot axis and the KDE inset axis
            ax_scatter = row_axes[col_i] if num_cols > 1 else axes
            if aspect is not None:
                ax_scatter.set(aspect=aspect)
            ax_kde = ax_scatter.inset_axes([0, 1, 1, 0.15], sharex=ax_scatter)

            # Create scatter plot with appropriate styling based on mode
            if use_hue_mode:
                frs = sub_df.groupby(hue_col)["p < 0.05"].mean()
                sns.scatterplot(
                    ax=ax_scatter,
                    data=sub_df,
                    x=metric,
                    y=f"-log10({metric} p-value)",
                    style="markers",
                    hue=hue_col,
                    palette=palette,
                    markers={"p<0.05": "o", "p>=0.05": "s"},
                    s=point_size,
                )
                # Add horizontal line for p=0.05
                ax_scatter.axhline(-np.log10(0.05), color="grey", linestyle="--")

                # Apply figure-specific formatting if applicable
                if figure in ["Fig3A", "Fig3B", "Fig3E"]:
                    apply_figure_formatting(
                        figure,
                        ax_scatter,
                        pr_x,
                        pr_y,
                        frs,
                        palette,
                        fr_total,
                        row_value,
                        y_label,
                    )
            else:
                sns.scatterplot(
                    ax=ax_scatter,
                    data=sub_df,
                    x=metric,
                    y=f"-log10({metric} p-value)",
                    hue="p < 0.05",
                    s=point_size,
                )
                # ax_scatter.set_title(f"{col_value}", fontsize=16, pad=20)
                ax_scatter.set_title(f"{col_value}")

                # Simple retrieved percentage for non-hue mode
                if col_i == 1:
                    ax_scatter.text(
                        pr_x,
                        0.02,
                        f"Retrieved: {fr_total:.0%}",
                        transform=ax_scatter.transAxes,
                    )
                else:
                    ax_scatter.text(
                        pr_x,
                        pr_y,
                        f"Retrieved: {fr_total:.0%}",
                        transform=ax_scatter.transAxes,
                    )

            # Common scatter plot formatting
            ax_scatter.set_xlabel("")
            ax_scatter.set_xlim(-0.05, 1.05)
            ax_scatter.set_ylim(-0.1, max(sub_df[f"-log10({metric} p-value)"]) + 0.25)
            ax_scatter.xaxis.set_major_locator(MultipleLocator(base=0.25))
            ax_scatter.get_xaxis().set_major_formatter(
                FuncFormatter(partial(x_axis_formatter, n_labels=6))
            )

            # Remove legend from subplot and store handles and labels for main legend
            handles, labels = ax_scatter.get_legend_handles_labels()
            if use_hue_mode:
                labels = [label.replace("markers", "") for label in labels]
            ax_scatter.get_legend().remove()

            # Create KDE plots at the top of each scatter plot
            mmaps = {}
            max_kde_y = 0

            if use_hue_mode:
                for hue_val in sorted(sub_df[hue_col].unique()):
                    sns.kdeplot(
                        ax=ax_kde,
                        data=sub_df[(sub_df[hue_col] == hue_val)],
                        x=metric,
                        label=str(hue_val),
                        cut=1,
                        clip=(-0.05, 1.05),
                        color=palette[hue_val],
                    )
                    mmaps[hue_val] = sub_df[(sub_df[hue_col] == hue_val)][metric].mean()
                    max_kde_y = max(max_kde_y, max(ax_kde.lines[-1].get_ydata()))
            else:
                for p_value in sorted(sub_df["p < 0.05"].unique()):
                    sns.kdeplot(
                        ax=ax_kde,
                        data=sub_df[sub_df["p < 0.05"] == p_value],
                        x=metric,
                        label=str(p_value),
                    )
                    max_kde_y = max(max_kde_y, max(ax_kde.lines[-1].get_ydata()))

                # Add mean line and label for non-hue mode
                ax_kde.axvline(mean_map, color="grey", linestyle="--")
                if mean_map < 0.5:
                    ax_kde.text(
                        mean_map + 0.05,
                        kde_y,
                        f"Mean {metric}: {mean_map:.2f}",
                        transform=ax_kde.transAxes,
                    )
                else:
                    ax_kde.text(
                        mean_map - 0.2,
                        kde_y,
                        f"Mean {metric}: {mean_map:.2f}",
                        transform=ax_kde.transAxes,
                    )

            # Format the KDE subplot
            ax_kde.get_yaxis().set_visible(False)
            ax_kde.get_xaxis().set_visible(False)
            sns.despine(ax=ax_kde, left=True, bottom=True)

    # Add metric label to the figure
    fig.text(m_x, m_y, metric, ha="center", va="center")

    # Handle layout and legend
    plt.tight_layout()
    if legend:
        lgd = fig.legend(
            handles,
            labels,
            title="" if use_hue_mode else "p < 0.05",
            loc=legend_loc,
            bbox_to_anchor=(l_x, l_y),
            frameon=legend_frameon,
        )
        lgd.get_frame().set_linewidth(plt.rcParams["axes.linewidth"])

    if adjust is not None:
        fig.subplots_adjust(**adjust)

    # Save figure if path provided
    if save_path is not None and figure is not None:
        fig.savefig(Path(save_path) / f"{figure}.png", dpi=300)
        fig.savefig(Path(save_path) / f"{figure}.svg")
        fig.savefig(Path(save_path) / f"{figure}.pdf")

    plt.show()
    return fig


def plot_map_activity_scatter(
    df,
    x: str = "relative_activity_day5",
    y: str = "mAP",
    aes_mapping: dict = None,  # e.g. {"fill": "Cell type"} or {"color": "p < 0.05"}
    facet: str = None,
    facet_ncol: int = None,
    density_aes: dict = None,
    density_size: dict = None,
    add_linear_model: bool = False,
    smooth_method: str = "lm",
    smooth_color: str = "black",
    smooth_size: float = 0.3,
    point_shape="o",
    point_size: float = 0.1,
    point_alpha: float = 1.0,
    point_stroke: float = 0.1,
    horizontal_line: float = None,
    x_label: str = None,
    y_label: str = None,
    x_range: tuple = None,
    y_range: tuple = None,
    x_scale: dict = None,  # e.g. {"breaks": [0, 0.5, 1], "labels": ["0", "0.5", "1"], "minor_breaks": []}
    y_scale: dict = None,  # e.g. {"breaks": [0, 0.5, 1], "labels": ["0", "0.5", "1"], "minor_breaks": []}
    scale_manual_values: dict = None,
    guide_title: str = None,
    width: float = 8,
    height: float = 4,
    tick_length: float = 1.0,
    legend_position: str = "none",
    theme_font_size: float = 5,
    theme_font_family: str = "Open Sans",
):
    """
    General global scatter plotting function that supports both bulk and singlecell styles and optional faceting.

    Parameters:
      df: DataFrame with the data.
      x, y: Column names to plot.
      aes_mapping: Dictionary of aesthetic mappings (e.g. {"fill": "Cell type"} or {"color": "p < 0.05"}).
      facet: Column name for faceting; if None, no faceting is applied.
      facet_ncol: Number of facet columns (if faceting).
      bulk_point_size: Point size for bulk scatter layer.
      smooth_method, smooth_color, smooth_size: Smoothing line parameters in bulk mode.
      singlecell_density_point_size, singlecell_density_point_alpha: Point parameters for singlecell mode.
      point_shape, point_alpha, point_stroke: Common point appearance settings.
      horizontal_line: Draws a horizontal dashed line at this y-value if provided.
      xlab_str, ylab_str: Axis labels.
      x_range, y_range: Axis limits applied via coord_cartesian.
      x_scale, y_scale: Additional scale settings (passed to scale_x_continuous/scale_y_continuous).
      scale_manual_values: Manual scale mapping (applied via scale_fill_manual/scale_color_manual).
      guide_title: Title for the guide (if mapping "color").
      width, height: Figure dimensions in inches.
      tick_length: Tick length on the matplotlib axes.
      theme_font_size: Font size for all text.

    Returns:
      A matplotlib Figure.
    """
    if aes_mapping is None:
        aes_mapping = {"fill": "p < 0.05"}

    p = gg.ggplot(df, gg.aes(x=x, y=y, **aes_mapping))

    p = p + gg.geom_point(
        size=point_size,
        shape=point_shape,
        alpha=point_alpha,
        color="white",
        stroke=point_stroke,
    )

    if density_aes is not None:
        p = p + gg.geom_density_2d(gg.aes(**density_aes), size=density_size)

    if add_linear_model:
        p = p + gg.geom_smooth(
            gg.aes(group=1), method=smooth_method, size=smooth_size, color=smooth_color
        )

    if horizontal_line is not None:
        p = p + gg.geom_hline(
            yintercept=horizontal_line, linetype="dashed", color="grey", size=0.2
        )

    if facet is not None:
        p = p + gg.facet_wrap(f"~{facet}", ncol=facet_ncol)

    if x_label is not None:
        p = p + gg.xlab(x_label)
    if y_label is not None:
        p = p + gg.ylab(y_label)

    if x_range is not None or y_range is not None:
        p = p + gg.coord_cartesian(xlim=x_range, ylim=y_range)

    if x_scale is not None:
        p = p + gg.scale_x_continuous(**x_scale)
    if y_scale is not None:
        p = p + gg.scale_y_continuous(**y_scale)

    if scale_manual_values is not None:
        p = p + gg.scale_fill_manual(values=scale_manual_values)
        p = p + gg.scale_color_manual(values=scale_manual_values)

    if guide_title is not None and "color" in aes_mapping:
        p = p + gg.guides(color=gg.guide_colorbar(title=guide_title))

    p = (
        p
        + gg.theme_bw()
        + gg.theme(
            plot_background=gg.element_rect(fill="white", colour=None),
            panel_background=gg.element_rect(fill="white", colour=None),
            panel_grid_major=gg.element_blank(),
            panel_grid_minor=gg.element_blank(),
            panel_border=gg.element_rect(colour="grey", fill=None, linewidth=0.2),
            axis_line=gg.element_line(color="black", linewidth=0.2),
            axis_text=gg.element_text(family=theme_font_family, size=theme_font_size),
            axis_title=gg.element_text(family=theme_font_family, size=theme_font_size),
            legend_text=gg.element_text(family=theme_font_family, size=theme_font_size),
            legend_title=gg.element_text(
                family=theme_font_family, size=theme_font_size
            ),
            strip_text=gg.element_text(
                family=theme_font_family, size=theme_font_size, margin={"t": 0, "b": 0}
            ),
            strip_background=gg.element_rect(
                colour="black", fill="#fdfff4", linewidth=0.2
            ),
            text=gg.element_text(family=theme_font_family, size=theme_font_size),
            legend_position=legend_position,
            axis_ticks=gg.element_line(linewidth=0.2),
            figure_size=(width, height),
        )
    )

    fig = p.draw()
    fig.set_size_inches(width, height)
    for ax in fig.axes:
        ax.tick_params(length=tick_length)

    return fig
