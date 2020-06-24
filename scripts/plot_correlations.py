import sys
import os
import argparse
import seaborn
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import numpy as np
from functools import reduce
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--correlation_filename', required=True)
    parser.add_argument('--task_names', default="/home/aizenberg/dgellis/fCNN/data/labels/ALL-TAVOR_name-file.txt")
    parser.add_argument('--level', default="overall", choices=["overall", "domain", "task"])
    parser.add_argument('--stats_filename', default=None)
    return vars(parser.parse_args())


def normalize_correlation_matrix_by_axis(matrix, new_max, new_min, axis=0):
    matrix = (matrix - matrix.min(axis=axis))/(matrix.max(axis=axis) - matrix.min(axis=axis))
    matrix = ((new_max - new_min) * matrix) + new_min
    return matrix


def p_value_to_string(p_value, decimals=3):
    lower_limit = 10**(-decimals)
    if p_value >= lower_limit:
        p_value_string = ("{:." + str(decimals) + "f}").format(p_value)[1:]
        return "p=" + p_value_string
    else:
        return "p<" + str(lower_limit)[1:]


def normalize_correlation_matrix(matrix, new_max, new_min, axes=(0, 1)):
    for axis in axes:
        matrix = normalize_correlation_matrix_by_axis(matrix, new_max, new_min, axis=axis)
    return matrix


def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def save_fig(fig, filename, dpi=1200, extensions=('.pdf',), **kwargs):
    for extension in extensions:
        if extension in ('.jpg', '.png'):
            fig.savefig(filename + extension, dpi=dpi, **kwargs)
        else:
            fig.savefig(filename + extension, **kwargs)


def extract_diagonal_and_extra_diagonal_elements(matrix):
    diagonal_mask = np.diag(np.ones(matrix.shape[0], dtype=bool))
    diag_values = matrix[diagonal_mask]
    extra_diag_values = matrix[diagonal_mask == False]
    return diag_values, extra_diag_values


def plot_hist(correlations, ax, set_xlabel=True, set_ylabel=True, title=None, plot_p_value=True,
              p_value_fontsize='medium', legend=False, legend_loc="upper left"):
    diag_values, extra_diag_values = extract_diagonal_and_extra_diagonal_elements(correlations)
    for m, label, color in zip((diag_values, extra_diag_values),
                               ("correlation with self", "correlation with other"),
                               ("C1", "C0")):
        seaborn.distplot(m, ax=ax, kde_kws={"shade": True}, label=label, color=color)
    if title is not None:
        ax.set_title(title)
    if set_xlabel:
        ax.set_xlabel("Correlation")
        ax.tick_params(labelbottom=True)
    if set_ylabel:
        ax.set_ylabel("Density")
    d_value, p_value = ks_2samp(diag_values, extra_diag_values)
    if plot_p_value:
        ax.text(1, 1, "\n".join(("D=" + "{:.3f}".format(d_value)[1:], p_value_to_string(p_value))),
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=p_value_fontsize)
    if legend:
        ax.legend(loc=legend_loc)
    return d_value, p_value


def plot_heatmap(data, ax, vmin, vmax, cmap, cbar=True, cbar_ax=None, set_xlabel=True, set_ylabel=True, title=None,
                 tick_label_spacing=25, xlabel="Subjects (actual)", ylabel="Subjects (predicted)"):
    if set_xlabel:
        xticklabels = tick_label_spacing
    else:
        xticklabels = False
    if set_ylabel:
        yticklabels = tick_label_spacing
    else:
        yticklabels = False
    seaborn.heatmap(data=data, ax=ax, cbar=cbar, cbar_ax=cbar_ax, xticklabels=xticklabels, yticklabels=yticklabels,
                    vmax=vmax, vmin=vmin, cmap=cmap)
    if title is not None:
        ax.set_title(title)
    if set_xlabel:
        ax.set_xlabel(xlabel)
    if set_ylabel:
        ax.set_ylabel(ylabel)


def plot_correlation_panel(corr_matrix, column_width=3, row_height=3, norm_vmax=3, norm_vmin=-3, output_dir="."):
    # define a separate color bar for the overall corr matrix
    overall_cbar_fig, overall_cbar_ax = plt.subplots(figsize=(0.5, 5))
    norm_cbar_fig, norm_cbar_ax = plt.subplots(figsize=(0.5, 5))
    overall_cmap = seaborn.diverging_palette(220, 10, sep=1, center="light", as_cmap=True)

    overall_all_fig, (_overall_ax, _overall_cbar_ax, _overall_norm_ax, _overall_hist_ax) = plt.subplots(1, 4,
                                                                                    figsize=(5 * 3 + 1, 5),
                                                                                    gridspec_kw={'width_ratios': [5,
                                                                                                                  1,
                                                                                                                  5,
                                                                                                                  5]})

    overall_fig, overall_ax = plt.subplots(figsize=(column_width, row_height))
    overall_vmax = np.max(corr_matrix)
    overall_vmin = np.min(corr_matrix)
    plot_heatmap(data=corr_matrix, ax=overall_ax, set_xlabel=True, set_ylabel=True, cbar=True, vmax=overall_vmax,
                 vmin=overall_vmin, cmap=overall_cmap, cbar_ax=overall_cbar_ax)

    plot_heatmap(data=corr_matrix, ax=_overall_ax, set_xlabel=True, set_ylabel=True, cbar=False, vmax=overall_vmax,
                 vmin=overall_vmin, cmap=overall_cmap)
    _overall_cbar_ax.axis("off")
    corr_matrix_norm = normalize_correlation_matrix(corr_matrix, norm_vmax, norm_vmin, axes=(0, 1))
    overall_norm_fig, overall_norm_ax = plt.subplots(figsize=(column_width, row_height))
    plot_heatmap(data=corr_matrix_norm, ax=overall_norm_ax, set_xlabel=True, set_ylabel=True, cbar=False, vmax=norm_vmax,
                 vmin=norm_vmin, cmap=overall_cmap)
    plot_heatmap(data=corr_matrix_norm, ax=_overall_norm_ax, set_xlabel=True, set_ylabel=True, cbar=True, vmax=norm_vmax,
                 vmin=norm_vmin, cmap=overall_cmap, cbar_ax=norm_cbar_ax)

    save_fig(overall_fig, output_dir + "/correlation_matrix_overall", bbox_inches="tight")
    save_fig(overall_norm_fig, output_dir + "/correlation_matrix_overall_normalized", bbox_inches="tight")
    save_fig(overall_cbar_fig, output_dir + "/correlation_matrix_overall_colorbar", bbox_inches="tight")
    save_fig(norm_cbar_fig, output_dir + "/correlation_matrix_overall_normalized_colorbar", bbox_inches="tight")

    overall_hist_fig, overall_hist_ax = plt.subplots()
    d, p = plot_hist(corr_matrix, overall_hist_ax, title=None, plot_p_value=True, legend=True)
    print("D-value: {:.2f}\tp-value = {:.2e}".format(d, p))
    save_fig(overall_hist_fig, output_dir + "/correlation_overall_histogram", bbox_inches="tight")

    _ = plot_hist(corr_matrix, _overall_hist_ax, title=None, plot_p_value=True, legend=True)

    save_fig(overall_all_fig, output_dir + "/correlation_overall_panel", bbox_inches="tight")


def main():
    seaborn.set_palette('muted')
    seaborn.set_style('whitegrid')
    args = parse_args()

    if args["stats_filename"] is None:
        stats_filename = os.path.join(args["output_dir"], args["level"] + "_correlation_stats.csv")
    else:
        stats_filename = args["stats_filename"]

    if args["level"] == "overall":
        corr_matrix = np.load(args["correlation_filename"])[..., 0]
        plot_correlation_panel(corr_matrix=corr_matrix, output_dir=args["output_dir"])
    else:
        raise NotImplementedError("Level={}".format(args["level"]))


if __name__ == "__main__":
    main()
