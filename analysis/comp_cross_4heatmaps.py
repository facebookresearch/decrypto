# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.stats import sem


def load_model_mapping(mapping_file):
    # Load model mapping from a JSON file or return default mapping
    if mapping_file:
        try:
            with open(mapping_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load mapping file: {e}")
            print("Using default model mappings")

    # Default mapping
    return {
        "baseline-glove": "GloVe",
        "baseline-word2vec": "Word2Vec",
        "gpt-4o": "GPT-4o",
        "llama3.1_70b": "Llama-70B",
        "llama3.1_8b": "Llama-8B",
        "deepseek_r1_32b": "DS-R1-32B",
    }


def load_model_order(order_file):
    # Load model order from a JSON file or return default order
    if order_file:
        try:
            with open(order_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Warning: Could not load order file: {e}")
            print("Using default model order")

    # Default order
    return ["Word2Vec", "GloVe", "Llama-8B", "Llama-70B", "GPT-4o", "DS-R1-32B"]


def map_model_name(name, model_mapping):
    name = str(name).lower()
    return model_mapping.get(name, name.title())


def calculate_significance_mask(
    pivot_mean, pivot_sem, axis="columns", highlight_lowest=False
):
    if axis == "columns":
        labels = pivot_mean.columns
    else:
        labels = pivot_mean.index
    significance_mask = pd.DataFrame(
        False, index=pivot_mean.index, columns=pivot_mean.columns
    )
    for label in labels:
        if axis == "columns":
            means = pivot_mean[label].dropna()
            sems = pivot_sem[label][means.index]
        else:
            means = pivot_mean.loc[label].dropna()
            sems = pivot_sem.loc[label][means.index]

        # Sort the means in ascending or descending order based on highlight_lowest
        sorted_means = means.sort_values(ascending=highlight_lowest)
        sorted_sems = sems[sorted_means.index]

        # Get the list of indices to bolden
        bold_indices = []

        # Get the top/bottom mean and its interval
        top_mean = sorted_means.iloc[0]
        top_sem = sorted_sems.iloc[0]
        top_interval = (top_mean - top_sem, top_mean + top_sem)
        # Add the top/bottom index to the bold list
        bold_indices.append(sorted_means.index[0])

        # Check overlaps with other entries
        for idx in sorted_means.index[1:]:
            mean = sorted_means[idx]
            sem_local = sorted_sems[idx]
            interval = (mean - sem_local, mean + sem_local)
            # Check if intervals overlap with the top/bottom interval
            if (interval[0] <= top_interval[1]) and (interval[1] >= top_interval[0]):
                # Intervals overlap, add to bold_indices
                bold_indices.append(idx)
            else:
                # No more overlaps, break the loop
                break

        # Set significance_mask for the bold_indices
        for idx in bold_indices:
            if axis == "columns":
                significance_mask.at[idx, label] = True
            else:
                significance_mask.at[label, idx] = True

    return significance_mask


def process_competitive_data(csv_files, model_mapping, preferred_order):
    # Read CSV files into a list of DataFrames
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]

    # Concatenate the DataFrames
    df = pd.concat(dfs, ignore_index=True)

    # Map specific model names and format labels
    df["encoder"] = df["encoder"].apply(lambda x: map_model_name(x, model_mapping))
    df["decoder"] = df["decoder"].apply(lambda x: map_model_name(x, model_mapping))
    df["interceptor"] = df["interceptor"].apply(
        lambda x: map_model_name(x, model_mapping)
    )

    # Combine 'encoder' and 'decoder' into one column since they are the same
    df["encoder_decoder"] = df["encoder"]  # Assuming encoder and decoder are the same

    # Fill NaN values in 'intercepts' and 'both' with 0
    df["intercepts"] = df["intercepts"].fillna(0)
    df["both"] = df["both"].fillna(0)

    # Calculate total intercepts by including 'both' entries
    df["total_intercepts"] = df["intercepts"] + df["both"]

    # Group by 'encoder_decoder' and 'interceptor' and compute statistics
    grouped_turns = (
        df.groupby(["encoder_decoder", "interceptor"])["avg_turns_per_episode"]
        .agg(["mean", sem])
        .reset_index()
    )
    # Divide total_intercepts by 32 before aggregation
    df["total_intercepts_per_game"] = df["total_intercepts"] / 32
    grouped_intercepts = (
        df.groupby(["encoder_decoder", "interceptor"])["total_intercepts_per_game"]
        .agg(["mean", sem])
        .reset_index()
    )

    # Create pivot tables for mean and standard error
    pivot_mean_turns = grouped_turns.pivot(
        index="encoder_decoder", columns="interceptor", values="mean"
    )
    pivot_sem_turns = grouped_turns.pivot(
        index="encoder_decoder", columns="interceptor", values="sem"
    )
    pivot_mean_intercepts = grouped_intercepts.pivot(
        index="encoder_decoder", columns="interceptor", values="mean"
    )
    pivot_sem_intercepts = grouped_intercepts.pivot(
        index="encoder_decoder", columns="interceptor", values="sem"
    )

    # Dynamically create model_order based on actual data
    available_models = sorted(list(set(df["encoder_decoder"].unique().tolist())))

    # Create model order based on preferred order, but only for models that are actually in the data
    model_order = [model for model in preferred_order if model in available_models]

    # Reindex the pivot tables to enforce this order
    pivot_mean_turns = pivot_mean_turns.reindex(index=model_order, columns=model_order)
    pivot_sem_turns = pivot_sem_turns.reindex(index=model_order, columns=model_order)
    pivot_mean_intercepts = pivot_mean_intercepts.reindex(
        index=model_order, columns=model_order
    )
    pivot_sem_intercepts = pivot_sem_intercepts.reindex(
        index=model_order, columns=model_order
    )

    # Calculate significance masks
    significance_mask_turns = calculate_significance_mask(
        pivot_mean_turns, pivot_sem_turns, axis="columns"
    )
    significance_mask_intercepts = calculate_significance_mask(
        pivot_mean_intercepts, pivot_sem_intercepts, axis="index", highlight_lowest=True
    )

    return (
        pivot_mean_turns,
        pivot_sem_turns,
        significance_mask_turns,
        pivot_mean_intercepts,
        pivot_sem_intercepts,
        significance_mask_intercepts,
    )


def process_crossplay_data(csv_files, model_mapping, preferred_order):
    # Read CSV files into a list of DataFrames
    dfs = [pd.read_csv(csv_file) for csv_file in csv_files]

    # Concatenate the DataFrames
    df = pd.concat(dfs, ignore_index=True)

    # Map specific model names and format labels
    df["encoder"] = df["encoder"].apply(lambda x: map_model_name(x, model_mapping))
    df["decoder"] = df["decoder"].apply(lambda x: map_model_name(x, model_mapping))
    df["interceptor"] = df["interceptor"].apply(
        lambda x: map_model_name(x, model_mapping)
    )

    # Ignore entries where the model is 'claude3.5_sonnet' (if required)
    df = df[~df["encoder"].str.contains("claude3.5_sonnet", case=False)]
    df = df[~df["decoder"].str.contains("claude3.5_sonnet", case=False)]
    df = df[~df["interceptor"].str.contains("claude3.5_sonnet", case=False)]

    # Get the interceptor model (assuming it's the same for all entries)
    interceptor_model = df["interceptor"].unique()
    if len(interceptor_model) != 1:
        print(
            "Error: More than one interceptor model found. Please ensure there is only one interceptor model in the data."
        )
        sys.exit(1)
    interceptor_model = interceptor_model[0]

    # Fill NaN values in 'miscommunications' and 'both' with 0
    df["miscomms"] = df["miscomms"].fillna(0)
    df["both"] = df["both"].fillna(0)

    # Calculate total miscommunications by including 'both' entries
    df["total_miscomms"] = df["miscomms"] + df["both"]

    # Group by 'encoder' and 'decoder' and compute statistics
    grouped_turns = (
        df.groupby(["encoder", "decoder"])["avg_turns_per_episode"]
        .agg(["mean", sem])
        .reset_index()
    )
    df["total_miscomms_per_game"] = df["total_miscomms"] / 32
    grouped_miscomms = (
        df.groupby(["encoder", "decoder"])["total_miscomms_per_game"]
        .agg(["mean", sem])
        .reset_index()
    )

    # Create pivot tables for mean and standard error
    pivot_mean_turns = grouped_turns.pivot(
        index="encoder", columns="decoder", values="mean"
    )
    pivot_sem_turns = grouped_turns.pivot(
        index="encoder", columns="decoder", values="sem"
    )
    pivot_mean_miscomms = grouped_miscomms.pivot(
        index="encoder", columns="decoder", values="mean"
    )
    pivot_sem_miscomms = grouped_miscomms.pivot(
        index="encoder", columns="decoder", values="sem"
    )

    # Get available models from the data
    available_models = sorted(
        list(set(list(df["encoder"].unique()) + list(df["decoder"].unique())))
    )

    # Create model order based on preferred order, but only for models that are actually in the data
    model_order = [model for model in preferred_order if model in available_models]

    # Reindex the pivot tables to enforce this order
    pivot_mean_turns = pivot_mean_turns.reindex(index=model_order, columns=model_order)
    pivot_sem_turns = pivot_sem_turns.reindex(index=model_order, columns=model_order)
    pivot_mean_miscomms = pivot_mean_miscomms.reindex(
        index=model_order, columns=model_order
    )
    pivot_sem_miscomms = pivot_sem_miscomms.reindex(
        index=model_order, columns=model_order
    )

    # Calculate significance masks
    significance_mask_turns = calculate_significance_mask(
        pivot_mean_turns, pivot_sem_turns, axis="index"
    )
    significance_mask_miscomms = calculate_significance_mask(
        pivot_mean_miscomms, pivot_sem_miscomms, axis="index", highlight_lowest=True
    )

    return (
        pivot_mean_turns,
        pivot_sem_turns,
        significance_mask_turns,
        pivot_mean_miscomms,
        pivot_sem_miscomms,
        significance_mask_miscomms,
        interceptor_model,
    )


def main(
    competitive_csv_files, crossplay_csv_files, model_mapping_file, model_order_file
):
    try:
        # Load model mapping and order
        model_mapping = load_model_mapping(model_mapping_file)
        preferred_order = load_model_order(model_order_file)

        # Process data for both heatmaps
        comp_results = process_competitive_data(
            competitive_csv_files, model_mapping, preferred_order
        )
        cross_results = process_crossplay_data(
            crossplay_csv_files, model_mapping, preferred_order
        )

        (
            comp_pivot_mean_turns,
            comp_pivot_sem_turns,
            comp_significance_mask_turns,
            comp_pivot_mean_intercepts,
            comp_pivot_sem_intercepts,
            comp_significance_mask_intercepts,
        ) = comp_results

        (
            cross_pivot_mean_turns,
            cross_pivot_sem_turns,
            cross_significance_mask_turns,
            cross_pivot_mean_miscomms,
            cross_pivot_sem_miscomms,
            cross_significance_mask_miscomms,
            interceptor_model,
        ) = cross_results

        # Font sizes
        title_fontsize = 40
        annot_fs = 34
        ticklabel_fs = 34
        label_fs = 38

        # Create a custom colormap
        colors = [
            "#f7fbff",
            "#deebf7",
            "#c6dbef",
            "#9ecae1",
            "#6baed6",
            "#4292c6",
            "#2171b5",
            "#08519c",
            "#08306b",
        ]
        cmap = LinearSegmentedColormap.from_list("custom_blue", colors)

        # Normalize the color scaling for each metric
        norm_miscomms = Normalize(
            vmin=cross_pivot_mean_miscomms.min().min(),
            vmax=cross_pivot_mean_miscomms.max().max(),
        )
        norm_cross_turns = Normalize(
            vmin=cross_pivot_mean_turns.min().min(),
            vmax=cross_pivot_mean_turns.max().max(),
        )
        norm_intercepts = Normalize(
            vmin=comp_pivot_mean_intercepts.min().min(),
            vmax=comp_pivot_mean_intercepts.max().max(),
        )
        norm_comp_turns = Normalize(
            vmin=comp_pivot_mean_turns.min().min(),
            vmax=comp_pivot_mean_turns.max().max(),
        )

        # Increase font sizes for better readability
        sns.set_theme(font_scale=1.2)
        fig, axs = plt.subplots(2, 2, figsize=(36, 32))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        ax1, ax2, ax3, ax4 = axs.flatten()

        ##### Plot 1: Cross-Play Miscommunications #####
        sns.heatmap(
            cross_pivot_mean_miscomms,
            annot=False,
            fmt="",
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            square=True,
            vmin=norm_miscomms.vmin,
            vmax=norm_miscomms.vmax,
            ax=ax1,
            xticklabels=False,
        )
        # ax2.set_xlabel('')

        # Annotate each cell in the first heatmap
        for (i, j), val in np.ndenumerate(cross_pivot_mean_miscomms.values):
            mean_val = cross_pivot_mean_miscomms.iloc[i, j]
            sem_val = cross_pivot_sem_miscomms.iloc[i, j]
            if pd.isna(mean_val):
                continue
            annotation = f"{mean_val:.2f}\n±{sem_val:.2f}"
            # Get the normalized value for color mapping
            value = mean_val
            rgba_color = cmap(norm_miscomms(value))

            # Calculate luminance to decide text color
            r, g, b, _ = rgba_color
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            if cross_significance_mask_miscomms.iloc[i, j]:
                ax1.text(
                    j + 0.5,
                    i + 0.5,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    weight="bold",
                    color=text_color,
                )
            else:
                ax1.text(
                    j + 0.5,
                    i + 0.5,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    color=text_color,
                )

        # Set plot aesthetics for the first heatmap
        ax1.set_title("Cross-Play Miscommunications", fontsize=title_fontsize, pad=30)
        ax1.set_xlabel("Decoder Models", fontsize=label_fs, labelpad=20)
        ax1.set_ylabel("Encoder Models", fontsize=label_fs, labelpad=20)
        ax1.set_yticklabels(
            ax1.get_yticklabels(), fontsize=ticklabel_fs, rotation=45, ha="right"
        )

        ##### Plot 2: Cross-Play Avg Turns Per Episode #####
        sns.heatmap(
            cross_pivot_mean_turns,
            annot=False,
            fmt="",
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            square=True,
            vmin=norm_cross_turns.vmin,
            vmax=norm_cross_turns.vmax,
            ax=ax2,
            yticklabels=False,  # This will ensure y-axis labels are not shown
            xticklabels=False,
            # ylabel=None  # This will remove the y-axis label
        )

        # Remove y-axis ticks and label
        ax2.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax2.set_ylabel("")  # This will remove any existing y-axis label
        ax2.set_xlabel("")

        # Annotate each cell in the second heatmap
        for (i, j), val in np.ndenumerate(cross_pivot_mean_turns.values):
            mean_val = cross_pivot_mean_turns.iloc[i, j]
            sem_val = cross_pivot_sem_turns.iloc[i, j]
            if pd.isna(mean_val):
                continue
            annotation = f"{mean_val:.2f}\n±{sem_val:.2f}"
            # Get the normalized value for color mapping
            value = mean_val
            rgba_color = cmap(norm_cross_turns(value))

            # Calculate luminance to decide text color
            r, g, b, _ = rgba_color
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            if cross_significance_mask_turns.iloc[i, j]:
                ax2.text(
                    j + 0.5,
                    i + 0.5,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    weight="bold",
                    color=text_color,
                )
            else:
                ax2.text(
                    j + 0.5,
                    i + 0.5,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    color=text_color,
                )

        # Set plot aesthetics for the second heatmap
        ax2.set_title(
            "Cross-Play Avg Turns Per Episode", fontsize=title_fontsize, pad=30
        )
        ax2.set_xlabel("Decoder Models", fontsize=label_fs, labelpad=20)

        ##### Plot 3: Competitive Intercepts #####
        sns.heatmap(
            comp_pivot_mean_intercepts,
            annot=False,
            fmt="",
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            square=True,
            vmin=norm_intercepts.vmin,
            vmax=norm_intercepts.vmax,
            ax=ax3,
        )

        # Annotate each cell in the third heatmap
        for (i, j), val in np.ndenumerate(comp_pivot_mean_intercepts.values):
            mean_val = comp_pivot_mean_intercepts.iloc[i, j]
            sem_val = comp_pivot_sem_intercepts.iloc[i, j]
            if pd.isna(mean_val):
                continue
            annotation = f"{mean_val:.2f}\n±{sem_val:.2f}"
            # Get the normalized value for color mapping
            value = mean_val
            rgba_color = cmap(norm_intercepts(value))

            # Calculate luminance to decide text color
            r, g, b, _ = rgba_color
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            if comp_significance_mask_intercepts.iloc[i, j]:
                ax3.text(
                    j + 0.5,
                    i + 0.5,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    weight="bold",
                    color=text_color,
                )
            else:
                ax3.text(
                    j + 0.5,
                    i + 0.5,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    color=text_color,
                )

        # Set plot aesthetics for the third heatmap
        ax3.set_title("Competitive Intercepts", fontsize=title_fontsize, pad=30)
        ax3.set_xlabel("Interceptor Models", fontsize=label_fs, labelpad=20)
        ax3.set_ylabel("Encoder/Decoder Models", fontsize=label_fs, labelpad=20)
        ax3.set_xticklabels(
            ax3.get_xticklabels(), fontsize=ticklabel_fs, rotation=45, ha="right"
        )
        ax3.set_yticklabels(
            ax3.get_yticklabels(), fontsize=ticklabel_fs, rotation=45, ha="right"
        )

        ##### Plot 4: Competitive Avg Turns Per Episode #####
        sns.heatmap(
            comp_pivot_mean_turns,
            annot=False,
            fmt="",
            cmap=cmap,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
            square=True,
            vmin=norm_comp_turns.vmin,
            vmax=norm_comp_turns.vmax,
            ax=ax4,
            yticklabels=False,  # This will ensure y-axis labels are not shown
            # ylabel=None  # This will remove the y-axis label
        )

        # Remove y-axis ticks and label
        ax4.tick_params(axis="y", which="both", left=False, labelleft=False)
        ax4.set_ylabel("")  # This will remove any existing y-axis label

        # Annotate each cell in the fourth heatmap
        for (i, j), val in np.ndenumerate(comp_pivot_mean_turns.values):
            mean_val = comp_pivot_mean_turns.iloc[i, j]
            sem_val = comp_pivot_sem_turns.iloc[i, j]
            if pd.isna(mean_val):
                continue
            annotation = f"{mean_val:.2f}\n±{sem_val:.2f}"
            # Get the normalized value for color mapping
            value = mean_val
            rgba_color = cmap(norm_comp_turns(value))

            # Calculate luminance to decide text color
            r, g, b, _ = rgba_color
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            text_color = "white" if luminance < 0.5 else "black"
            if comp_significance_mask_turns.iloc[i, j]:
                ax4.text(
                    j + 0.5,
                    i + 0.5,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    weight="bold",
                    color=text_color,
                )
            else:
                ax4.text(
                    j + 0.5,
                    i + 0.5,
                    annotation,
                    ha="center",
                    va="center",
                    fontsize=annot_fs,
                    color=text_color,
                )

        # Set plot aesthetics for the fourth heatmap
        ax4.set_title(
            "Competitive Avg Turns Per Episode", fontsize=title_fontsize, pad=30
        )
        ax4.set_xlabel("Interceptor Models", fontsize=label_fs, labelpad=20)
        ax4.set_xticklabels(
            ax4.get_xticklabels(), fontsize=ticklabel_fs, rotation=45, ha="right"
        )

        # Create colorbars for each subplot
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # For ax1
        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="5%", pad=0.1)
        sm1 = cm.ScalarMappable(cmap=cmap, norm=norm_miscomms)
        sm1.set_array([])
        cbar1 = fig.colorbar(sm1, cax=cax1)
        cbar1.ax.tick_params(labelsize=ticklabel_fs)
        cbar1.ax.set_ylabel("Mean Miscommunications", fontsize=label_fs, labelpad=20)

        # For ax2
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.1)
        sm2 = cm.ScalarMappable(cmap=cmap, norm=norm_cross_turns)
        sm2.set_array([])
        cbar2 = fig.colorbar(sm2, cax=cax2)
        cbar2.ax.tick_params(labelsize=ticklabel_fs)
        cbar2.ax.set_ylabel(
            "Mean Avg Turns Per Episode", fontsize=label_fs, labelpad=20
        )

        # For ax3
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.1)
        sm3 = cm.ScalarMappable(cmap=cmap, norm=norm_intercepts)
        sm3.set_array([])
        cbar3 = fig.colorbar(sm3, cax=cax3)
        cbar3.ax.tick_params(labelsize=ticklabel_fs)
        cbar3.ax.set_ylabel("Mean Intercepts", fontsize=label_fs, labelpad=20)

        # For ax4
        divider4 = make_axes_locatable(ax4)
        cax4 = divider4.append_axes("right", size="5%", pad=0.1)
        sm4 = cm.ScalarMappable(cmap=cmap, norm=norm_comp_turns)
        sm4.set_array([])
        cbar4 = fig.colorbar(sm4, cax=cax4)
        cbar4.ax.tick_params(labelsize=ticklabel_fs)
        cbar4.ax.set_ylabel(
            "Mean Avg Turns Per Episode", fontsize=label_fs, labelpad=20
        )

        plt.savefig(
            "src/figures/combined_heatmap_4plots.pdf", format="pdf", bbox_inches="tight"
        )
        plt.show()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except pd.errors.EmptyDataError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except pd.errors.ParserError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a figure with 4 subplots for Cross-Play and Competitive evaluations"
    )
    parser.add_argument(
        "--competitive_csv_files",
        nargs="+",
        required=True,
        help="List of CSV files for competitive data",
    )
    parser.add_argument(
        "--crossplay_csv_files",
        nargs="+",
        required=True,
        help="List of CSV files for cross-play data",
    )
    parser.add_argument(
        "--model_mapping", type=str, help="JSON file containing model name mappings"
    )
    parser.add_argument(
        "--model_order", type=str, help="JSON file containing preferred model order"
    )
    args = parser.parse_args()

    main(
        args.competitive_csv_files,
        args.crossplay_csv_files,
        args.model_mapping,
        args.model_order,
    )
