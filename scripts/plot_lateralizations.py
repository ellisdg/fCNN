import seaborn
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp, pearsonr
import numpy as np
import sys
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


def plot_scatter(predicted, actual, ax, color="C3", title=None):
    seaborn.regplot(x=predicted, y=actual, color=color, ax=ax)
    ax.set_xlabel(r"RIGHT $\leftarrow$ Predicted Lateralization Index $\rightarrow$ LEFT")
    ax.set_ylabel(r"RIGHT $\leftarrow$ Actual Lateralization Index $\rightarrow$ LEFT")
    if title is not None:
        ax.set_title(title)


def plot_bar(predicted, actual, x, ax, title=None):
    ax.bar(x=x, height=actual, width=0.5, label="actual")
    ax.bar(x=x + 0.5, height=predicted, width=0.5, label="predicted")
    seaborn.despine(ax=ax, top=True, left=False, bottom=False, right=True)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel("Subjects")
    ax.set_ylabel(r"RIGHT $\leftarrow$ Lateralization Index $\rightarrow$ LEFT")
    ax.legend()
    if title is not None:
        ax.set_title(title)


def main():
    seaborn.set_palette('muted')
    seaborn.set_style('whitegrid')
    lateralization_file = sys.argv[1]
    name_file = sys.argv[2]
    output_dir = os.path.abspath(sys.argv[3])
    try:
        stats_filename = os.path.abspath(sys.argv[4])
    except IndexError:
        stats_filename = None

    contrast_names = list()
    tasks = list()
    lateralizations = np.load(lateralization_file)
    stats = list()

    names = read_namefile(name_file)
    task = os.path.basename(name_file).split("_")[0].replace("-TAVOR", "")
    if task == "ALL":
        for name in names:
            t, n = name.split(" ")
            tasks.append(t)
            contrast_names.append(n)
    else:
        tasks.extend([task] * len(names))
        contrast_names.extend(names)

    for task_ind in range(len(tasks)):
        task = tasks[task_ind]
        contrast = contrast_names[task_ind]
        fig_width = 18
        fig_height = 6
        title = " ".join((task, contrast))
        x = np.arange(lateralizations.shape[0])
        ind = np.argsort(lateralizations[..., task_ind, 1], axis=0)
        actual = lateralizations[..., task_ind, 1][ind]
        predicted = lateralizations[..., task_ind, 0][ind]

        fig_bar, ax_bar = plt.subplots()
        plot_bar(predicted, actual, x, ax_bar, title=title)
        save_fig(fig_bar, output_dir + '/lateralization_{task}_{name}_bar'.format(task=task, name=contrast))
        plt.close(fig_bar)

        fig_scatter, ax_scatter = plt.subplots()
        plot_scatter(predicted=predicted, actual=actual, color="C3", ax=ax_scatter, title=title)
        save_fig(fig_scatter, output_dir + '/lateralization_{task}_{name}_scatter'.format(task=task, name=contrast))
        plt.close(fig_scatter)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height), gridspec_kw={'width_ratios': [2, 1]})
        plot_bar(predicted, actual, x=x, ax=ax1, title=title)
        plot_scatter(predicted=predicted, actual=actual, color="C3", ax=ax2)
        save_fig(fig, output_dir + '/lateralization_{task}_{name}_bar_scatter'.format(task=task, name=contrast))
        plt.close(fig)

        correlation, p_value = pearsonr(predicted, actual)
        stats.append((task, contrast, correlation, p_value))

    if stats_filename is not None:
        stats_df = pd.DataFrame(stats, columns=["Task", "Contrast", "Correlation", "P-Value"])
        stats_df.to_csv(stats_filename)


if __name__ == "__main__":
    main()
