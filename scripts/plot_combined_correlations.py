import seaborn
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp
import numpy as np
import sys
from functools import reduce


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


def main():
    seaborn.set_palette('muted')
    seaborn.set_style('whitegrid')
    correlation_files = sys.argv[1].split(",")
    name_files = sys.argv[2].split(",")
    output_dir = os.path.abspath(sys.argv[3])

    temp_correlations = list()
    metric_names = list()
    tasks = list()
    subjects = list()
    for c_file, n_file in zip(correlation_files, name_files):
        corr = np.load(c_file)
        print(c_file, corr.shape)
        subs = np.load(c_file.replace(".npy", "_subjects.npy"))
        subjects.append(subs)
        temp_correlations.append(corr)
        names = read_namefile(n_file)
        metric_names.extend(names)
        task = os.path.basename(n_file).split("_")[0].replace("-TAVOR", "")
        tasks.extend([task] * len(names))

    all_subjects = reduce(np.intersect1d, subjects)
    correlations = list()
    for sub_list, corr in zip(np.copy(subjects), temp_correlations):
        s, i, i_all = np.intersect1d(sub_list, all_subjects, return_indices=True)
        np.testing.assert_equal(s, all_subjects)
        correlations.append(corr[i][:, i])

    # all_subjects = np.unique(subjects)
    # indices = [np.in1d(all_subjects, subs) for subs in subjects]
    # index = np.all(indices, axis=0)
    # included_subjects = all_subjects[index]
    # indices = [np.in1d(subs, included_subjects) for subs in subjects]
    # correlations = [corr[ind, ind] for corr, ind in zip(correlations, indices)]

    # unique_tasks = np.unique(tasks)
    correlations = np.concatenate(correlations, axis=-2)
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
    cmap =  seaborn.diverging_palette(220, 10, sep=1, center="light", as_cmap=True)
    # cmap = seaborn.cubehelix_palette(n_colors=8, as_cmap=True)
    # cmap = seaborn.color_palette("cubehelix", 1000)
    # cmap = seaborn.color_palette("nipy_spectral", 1000)
    names = list()
    result = list()
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
        print(title, "D-value: {:.2f}\tp-value = {:.8f}".format(*ks_2samp(diag_values, extra_diag_values)))
        result.append((diag_values.mean() - extra_diag_values.mean())/extra_diag_values.mean())
    fig.savefig(output_dir + "/correlation_matrices.png")
    hist_fig.savefig(output_dir + "/correlation_matrices_histograms.png")
    cbar_fig.savefig(output_dir + "/correlation_matrices_colorbar.png", bbox_inches="tight")

    avg_fig, avg_ax = plt.subplots(figsize=(column_height, row_height))
    avg_corr = corr_matrices.mean(axis=-1)
    seaborn.heatmap(data=avg_corr, ax=avg_ax, xticklabels=False, yticklabels=False, cbar=False, vmax=vmax, vmin=vmin,
                    cmap=cmap)
    # avg_ax.set_title()
    avg_ax.set_ylabel("subjects (predicted)")
    avg_ax.set_xlabel("subjects (actual)")
    avg_fig.savefig(output_dir + "/correlation_matrix_average.png")
    # prediction_errors.melt(id_vars=['CUE', 'LF', 'LH', 'RF', 'RH', 'T', 'AVG', 'CUE-AVG', 'LF-AVG', 'LH-AVG', 'RF-AVG', 'RH-AVG', 'T-AVG'])

    # corr_matrices.mean(axis=1)
    avg_corr = corr_matrices.mean(axis=-1)

    avg_corr_norm = normalize_correlation_matrix(avg_corr, vmax, vmin, axes=(0, 1))

    # corr_matrices_col_norm = (corr_matrices_col_norm/corr_matrices.max(axis=1))
    # corr_matrices_col_norm = vmax * (corr_matrices_col_norm/corr_matrices_col_norm.max())
    avg_fig, avg_ax = plt.subplots(figsize=(column_height, row_height))

    seaborn.heatmap(data=avg_corr_norm, ax=avg_ax, xticklabels=False, yticklabels=False, cbar=False, vmax=vmax, vmin=vmin,
                     cmap=cmap)
    # seaborn.heatmap(data=avg_corr, ax=avg_ax, xticklabels=False, yticklabels=False, cmap=cmap, square=True)
    # avg_ax.set_title(task)
    avg_ax.set_ylabel("subjects (predicted)")
    avg_ax.set_xlabel("subjects (actual)")
    avg_fig.savefig(output_dir + "/correlation_matrix_average_normalized.png")

    fig, axes = plt.subplots(nrows=n_rows, ncols=plots_per_row, figsize=(plots_per_row*column_height, n_rows*row_height))
    cbar_fig, cbar_ax = plt.subplots(figsize=(0.5, 5))

    for i, (task, metric_name) in enumerate(zip(tasks, metric_names)):
        title = "{task} ".format(task=task) + metric_name
        ax = np.ravel(axes)[i]
        if i + 1 == len(metric_names):
            cbar = True
        else:
            cbar = False
        # seaborn.heatmap(data=corr_matrices[..., i], ax=ax, cbar=cbar, cbar_ax=cbar_ax, xticklabels=False, yticklabels=False,
        #                 cmap=cmap)
        seaborn.heatmap(data=normalize_correlation_matrix(corr_matrices[..., i], vmax, vmin, axes=(0, 1)), ax=ax, cbar=cbar,
                        cbar_ax=cbar_ax, xticklabels=False, yticklabels=False,
                        vmax=vmax, vmin=vmin, cmap=cmap)
        ax.set_title(title)
    fig.savefig(output_dir + "/correlation_matrices_normalized.png")

    fig, ax = plt.subplots()
    diagonal_mask = np.diag(np.ones(avg_corr.shape[0], dtype=bool))
    diag_values = avg_corr[diagonal_mask]
    extra_diag_values = avg_corr[diagonal_mask == False]
    # seaborn.distplot(extra_diag_values, norm_hist=True)
    # _ = seaborn.distplot(extra_diag_values, hist_kws=dict(density=True, stacked=True), ax=ax)
    _ = seaborn.distplot(extra_diag_values, kde_kws={"shade": True}, ax=ax, label="predicted vs other subjects")
    _ = seaborn.distplot(diag_values, kde_kws={"shade": True}, ax=ax, label="predicted vs actual")
    # ax.set_title(task)
    ax.set_ylabel("Count")
    ax.set_xlabel("Correlation")
    ax.legend()
    fig.savefig(output_dir + "/average_correlation_histogram.png")
    d, p = ks_2samp(diag_values, extra_diag_values)
    print("D-value: {:.2f}\tp-value = {:.8f}".format(d, p))

    w = 6
    width = 0.4
    gap = 0.
    h = (width + gap) * len(names)
    fig, ax = plt.subplots(figsize=(w, h))
    for i, task in enumerate(np.unique(tasks)[::-1]):
        mask = np.asarray(tasks) == task
        ax.barh(np.squeeze(np.where(mask)), np.asarray(result)[mask] * 100, label=task, color="C{}".format(i))
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.legend()
    ax.set_xlabel("Self vs other increase (in %)")
    seaborn.despine(ax=ax, top=True)
    fig.savefig(output_dir + "/increase_correlation_over_mean_correlation.png", bbox_inches="tight")


if __name__ == "__main__":
    main()
