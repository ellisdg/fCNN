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
              p_value_fontsize='medium'):
    diag_values, extra_diag_values = extract_diagonal_and_extra_diagonal_elements(correlations)
    for m in (extra_diag_values, diag_values):
        seaborn.distplot(m, ax=ax, kde_kws={"shade": True})
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
    norm_vmin = -3
    norm_vmax = 3

    n_plots = len(metric_names)
    plots_per_row = 6
    n_rows = int(np.ceil(n_plots/plots_per_row))
    row_height = 3
    column_width = 3
    fig, axes = plt.subplots(nrows=n_rows, ncols=plots_per_row, figsize=(plots_per_row*column_width,
                                                                         n_rows*row_height))
    norm_fig, norm_axes = plt.subplots(nrows=n_rows, ncols=plots_per_row, figsize=(plots_per_row*column_width,
                                                                                   n_rows*row_height))
    hist_fig, hist_axes = plt.subplots(nrows=n_rows, ncols=plots_per_row, figsize=(plots_per_row*column_width,
                                                                                   n_rows*row_height),
                                       sharex=True, sharey=True)
    cbar_fig, cbar_ax = plt.subplots(figsize=(0.5, row_height))
    cmap = plt.get_cmap("jet")
    # cmap = seaborn.diverging_palette(220, 10, sep=1, center="light", as_cmap=True)
    # cmap = seaborn.cubehelix_palette(n_colors=8, as_cmap=True)
    # cmap = seaborn.color_palette("cubehelix", 1000)
    # cmap = seaborn.color_palette("nipy_spectral", 1000)
    names = list()
    result = list()
    stats = list()
    n_empty_plots = plots_per_row * n_rows - len(tasks)
    for i in range(n_rows * plots_per_row):
        ax = np.ravel(axes)[i]
        norm_ax = np.ravel(norm_axes)[i]
        hist_ax = np.ravel(hist_axes)[i]
        try:
            corr_matrix = corr_matrices[..., i]
            task = tasks[i]
            metric_name = metric_names[i]
            title = task + " " + metric_name
            names.append(title)
        except IndexError:
            ax.axis("off")
            norm_ax.axis("off")
            hist_ax.axis("off")
            continue
        set_ylabel = (i % plots_per_row) == 0
        set_xlabel = ((i + n_empty_plots) / plots_per_row) >= (n_rows - 1)
        if i + 1 == len(metric_names):
            cbar = True
        else:
            cbar = False
        plot_heatmap(data=corr_matrix, ax=ax, cbar=cbar, cbar_ax=cbar_ax, set_xlabel=set_xlabel, set_ylabel=set_ylabel,
                     vmax=vmax, vmin=vmin, cmap=cmap, title=title)
        plot_heatmap(data=normalize_correlation_matrix(corr_matrix, norm_vmax, norm_vmin, axes=(0, 1)), ax=norm_ax,
                     cbar=False, set_xlabel=set_xlabel, set_ylabel=set_ylabel, vmax=norm_vmax, vmin=norm_vmin,
                     cmap=cmap, title=title)
        d_value, p_value = plot_hist(corr_matrix, hist_ax, set_ylabel=set_ylabel, set_xlabel=set_xlabel, title=title,
                                     plot_p_value=True, p_value_fontsize="large")
        stats.append([task, metric_name, d_value, p_value])
        print(title, "D-value: {:.2f}\tp-value = {:.2e}".format(d_value, p_value))
        diag_values, extra_diag_values = extract_diagonal_and_extra_diagonal_elements(corr_matrix)
        result.append((diag_values.mean() - extra_diag_values.mean())/extra_diag_values.mean())
    save_fig(fig, output_dir + "/correlation_matrices", bbox_inches="tight")
    save_fig(hist_fig, output_dir + "/correlation_matrices_histograms", bbox_inches="tight")
    save_fig(cbar_fig, output_dir + "/correlation_matrices_colorbar", bbox_inches="tight")
    save_fig(norm_fig, output_dir + "/correlation_matrices_normalized", bbox_inches="tight")

    # define a separate color bar for the average histograms
    avg_cbar_fig, avg_cbar_ax = plt.subplots(figsize=(0.5, 5))
    norm_cbar_fig, norm_cbar_ax = plt.subplots(figsize=(0.5, 5))
    avg_cmap = seaborn.diverging_palette(220, 10, sep=1, center="light", as_cmap=True)

    avg_all_fig, (_avg_ax, _avg_cbar_ax, _avg_norm_ax, _avg_hist_ax) = plt.subplots(1, 4,
                                                                                    figsize=(5 * 3 + 1, 5),
                                                                                    gridspec_kw={'width_ratios': [5,
                                                                                                                  1,
                                                                                                                  5,
                                                                                                                  5]})

    avg_fig, avg_ax = plt.subplots(figsize=(column_width, row_height))
    avg_corr = corr_matrices.mean(axis=-1)
    avg_vmax = np.max(avg_corr)
    avg_vmin = np.min(avg_corr)

    plot_heatmap(data=avg_corr, ax=avg_ax, set_xlabel=True, set_ylabel=True, cbar=True, vmax=avg_vmax,
                 vmin=avg_vmin, cmap=avg_cmap, cbar_ax=avg_cbar_ax)

    plot_heatmap(data=avg_corr, ax=_avg_ax, set_xlabel=True, set_ylabel=True, cbar=False, vmax=avg_vmax,
                 vmin=avg_vmin, cmap=avg_cmap)
    _avg_cbar_ax.axis("off")
    avg_corr_norm = normalize_correlation_matrix(avg_corr, norm_vmax, norm_vmin, axes=(0, 1))
    avg_norm_fig, avg_norm_ax = plt.subplots(figsize=(column_width, row_height))
    plot_heatmap(data=avg_corr_norm, ax=avg_norm_ax, set_xlabel=True, set_ylabel=True, cbar=False, vmax=norm_vmax,
                 vmin=norm_vmin, cmap=avg_cmap)
    plot_heatmap(data=avg_corr_norm, ax=_avg_norm_ax, set_xlabel=True, set_ylabel=True, cbar=True, vmax=norm_vmax,
                 vmin=norm_vmin, cmap=avg_cmap, cbar_ax=norm_cbar_ax)

    save_fig(avg_fig, output_dir + "/correlation_matrix_average", bbox_inches="tight")
    save_fig(avg_norm_fig, output_dir + "/correlation_matrix_average_normalized", bbox_inches="tight")
    save_fig(avg_cbar_fig, output_dir + "/correlation_matrix_average_colorbar", bbox_inches="tight")
    save_fig(norm_cbar_fig, output_dir + "/correlation_matrix_normalized_colorbar", bbox_inches="tight")

    avg_hist_fig, avg_hist_ax = plt.subplots()
    d, p = plot_hist(avg_corr, avg_hist_ax, title=None, plot_p_value=True)
    stats.append(["Average", "ALL", d, p])
    print("D-value: {:.2f}\tp-value = {:.2e}".format(d, p))
    save_fig(avg_hist_fig, output_dir + "/correlation_average_histogram", bbox_inches="tight")

    _ = plot_hist(avg_corr, _avg_hist_ax, title=None, plot_p_value=True)

    save_fig(avg_all_fig, output_dir + "/correlation_average_panel", bbox_inches="tight")

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
