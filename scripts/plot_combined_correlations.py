import seaborn
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp
import numpy as np
import sys
from functools import reduce
import pandas as pd


def normalize_correlation_matrix_by_axis(matrix, new_max, new_min, axis=0):
    matrix = (matrix - matrix.min(axis=axis))/(matrix.max(axis=axis) - matrix.min(axis=axis))
    matrix = ((new_max - new_min) * matrix) + new_min
    return matrix


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


def main():
    seaborn.set_palette('muted')
    seaborn.set_style('whitegrid')
    correlation_files = sys.argv[1].split(",")
    name_files = sys.argv[2].split(",")
    output_dir = os.path.abspath(sys.argv[3])
    try:
        stats_filename = os.path.abspath(sys.argv[4])
    except IndexError:
        stats_filename = None

    temp_correlations = list()
    metric_names = list()
    tasks = list()
    subjects = list()
    for c_file, n_file in zip(correlation_files, name_files):
        corr = np.load(c_file)
        print(c_file, corr.shape, n_file)
        subs = np.load(c_file.replace(".npy", "_subjects.npy"))
        subjects.append(subs)
        temp_correlations.append(corr)
        names = read_namefile(n_file)
        task = os.path.basename(n_file).split("_")[0].replace("-TAVOR", "")
        if task == "ALL":
            for name in names:
                t, n = name.split(" ")
                tasks.append(t)
                metric_names.append(n)
        else:
            tasks.extend([task] * len(names))
            metric_names.extend(names)

    if len(temp_correlations) > 1:
        all_subjects = reduce(np.intersect1d, subjects)
        correlations = list()
        for sub_list, corr in zip(np.copy(subjects), temp_correlations):
            s, i, i_all = np.intersect1d(sub_list, all_subjects, return_indices=True)
            np.testing.assert_equal(s, all_subjects)
            correlations.append(corr[i][:, i])
        correlations = np.concatenate(correlations, axis=-2)
    else:
        correlations = np.asarray(temp_correlations[0])

    corr_matrices = np.asarray(correlations)[..., 0]
    vmin = corr_matrices.min()
    vmax = corr_matrices.max()

    n_plots = len(metric_names)
    plots_per_row = 6
    n_rows = int(np.ceil(n_plots/plots_per_row))
    row_height = 4
    column_height = 4
    fig, axes = plt.subplots(nrows=n_rows, ncols=plots_per_row, figsize=(plots_per_row*column_height,
                                                                         n_rows*row_height))
    hist_fig, hist_axes = plt.subplots(nrows=n_rows, ncols=plots_per_row, figsize=(plots_per_row*column_height,
                                                                                   n_rows*row_height),
                                       sharex=True, sharey=True)
    cbar_fig, cbar_ax = plt.subplots(figsize=(0.5, 5))
    cmap = plt.get_cmap("jet")
    # cmap = seaborn.diverging_palette(220, 10, sep=1, center="light", as_cmap=True)
    # cmap = seaborn.cubehelix_palette(n_colors=8, as_cmap=True)
    # cmap = seaborn.color_palette("cubehelix", 1000)
    # cmap = seaborn.color_palette("nipy_spectral", 1000)
    names = list()
    result = list()
    stats = list()
    for i, (task, metric_name) in enumerate(zip(tasks, metric_names)):
        title = task + " " + metric_name
        names.append(title)
        ax = np.ravel(axes)[i]
        if i + 1 == len(metric_names):
            cbar = True
        else:
            cbar = False
        corr_matrix = corr_matrices[..., i]
        seaborn.heatmap(data=corr_matrix, ax=ax, cbar=cbar, cbar_ax=cbar_ax, xticklabels=False, yticklabels=False,
                        vmax=vmax, vmin=vmin, cmap=cmap)
        ax.set_title(title)
        hist_ax = np.ravel(hist_axes)[i]
        diagonal_mask = np.diag(np.ones(corr_matrix.shape[0], dtype=bool))
        diag_values = corr_matrix[diagonal_mask]
        extra_diag_values = corr_matrix[diagonal_mask == False]
        for m in (extra_diag_values, diag_values):
            seaborn.distplot(m, ax=hist_ax, kde_kws={"shade": True})
        hist_ax.set_title(title)
        hist_ax.set_xlabel("Correlation")
        hist_ax.set_ylabel("Count")
        d_value, p_value = ks_2samp(diag_values, extra_diag_values)
        stats.append([task, metric_name, d_value, p_value])
        print(title, "D-value: {:.2f}\tp-value = {:.8f}".format(d_value, p_value))
        result.append((diag_values.mean() - extra_diag_values.mean())/extra_diag_values.mean())
    save_fig(fig, output_dir + "/correlation_matrices")
    save_fig(hist_fig, output_dir + "/correlation_matrices_histograms")
    save_fig(cbar_fig, output_dir + "/correlation_matrices_colorbar", bbox_inches="tight")

    # define a seperate collor bar for the average histograms
    avg_cbar_fig, avg_cbar_ax = plt.subplots(figsize=(0.5, 5))
    avg_cmap = cmap = seaborn.diverging_palette(220, 10, sep=1, center="light", as_cmap=True)

    avg_fig, avg_ax = plt.subplots(figsize=(column_height, row_height))
    avg_corr = corr_matrices.mean(axis=-1)
    avg_vmax = np.max(avg_corr)
    avg_vmin = np.min(avg_corr)

    seaborn.heatmap(data=avg_corr, ax=avg_ax, xticklabels=False, yticklabels=False, cbar=True, vmax=avg_vmax,
                    vmin=avg_vmin, cmap=avg_cmap, cbar_ax=avg_cbar_ax)
    avg_ax.set_ylabel("subjects (predicted)")
    avg_ax.set_xlabel("subjects (actual)")
    save_fig(avg_fig, output_dir + "/correlation_matrix_average")

    avg_corr_norm = normalize_correlation_matrix(avg_corr, avg_vmax, avg_vmin, axes=(0, 1))

    avg_fig, avg_ax = plt.subplots(figsize=(column_height, row_height))

    seaborn.heatmap(data=avg_corr_norm, ax=avg_ax, xticklabels=False, yticklabels=False, cbar=False, vmax=avg_vmax,
                    vmin=avg_vmin, cmap=avg_cmap)
    avg_ax.set_ylabel("subjects (predicted)")
    avg_ax.set_xlabel("subjects (actual)")
    save_fig(avg_fig, output_dir + "/correlation_matrix_average_normalized")
    save_fig(avg_cbar_fig, output_dir + "/correlation_matrix_average_colorbar", bbox_inches="tight")

    fig, axes = plt.subplots(nrows=n_rows, ncols=plots_per_row, figsize=(plots_per_row*column_height,
                                                                         n_rows*row_height))

    for i, (task, metric_name) in enumerate(zip(tasks, metric_names)):
        title = "{task} ".format(task=task) + metric_name
        ax = np.ravel(axes)[i]
        seaborn.heatmap(data=normalize_correlation_matrix(corr_matrices[..., i], vmax, vmin, axes=(0, 1)), ax=ax,
                        cbar=False, xticklabels=False, yticklabels=False, vmax=vmax, vmin=vmin, cmap=cmap)
        ax.set_title(title)
    save_fig(fig, output_dir + "/correlation_matrices_normalized")

    fig, ax = plt.subplots()
    diagonal_mask = np.diag(np.ones(avg_corr.shape[0], dtype=bool))
    diag_values = avg_corr[diagonal_mask]
    extra_diag_values = avg_corr[diagonal_mask == False]

    _ = seaborn.distplot(extra_diag_values, kde_kws={"shade": True}, ax=ax, label="predicted vs other subjects")
    _ = seaborn.distplot(diag_values, kde_kws={"shade": True}, ax=ax, label="predicted vs actual")

    ax.set_ylabel("Count")
    ax.set_xlabel("Correlation")
    ax.legend()
    save_fig(fig, output_dir + "/average_correlation_histogram")
    d, p = ks_2samp(diag_values, extra_diag_values)
    stats.append(["Average", "ALL", d, p])
    print("D-value: {:.2f}\tp-value = {:.8f}".format(d, p))

    if stats_filename is not None:
        stats_df = pd.DataFrame(stats, columns=["Task", "Contrast", "D-Value", "P-Value"])
        stats_df.to_csv(stats_filename)

    w = 6
    width = 0.4
    gap = 0.
    h = (width + gap) * len(names)
    fig, ax = plt.subplots(figsize=(w, h))
    for i, task in enumerate(np.unique(tasks)):
        mask = np.asarray(tasks[::-1]) == task
        ax.barh(np.squeeze(np.where(mask)), np.asarray(result)[::-1][mask] * 100, label=task, color="C{}".format(i))
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names[::-1])
    ax.legend()
    ax.set_xlabel("Self vs other increase (in %)")
    seaborn.despine(ax=ax, top=True)
    save_fig(fig, output_dir + "/increase_correlation_over_mean_correlation", bbox_inches="tight")


if __name__ == "__main__":
    main()
