import seaborn
import matplotlib.pyplot as plt
import os
from scipy.stats import ks_2samp, pearsonr
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

    print(lateralizations.shape)
    for task_ind in range(lateralizations.shape[-1]):
        task = tasks[task_ind]
        contrast = contrast_names[task_ind]
        fig_width = 18
        fig_height = 6
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        x = np.arange(lateralizations.shape[0])
        ind = np.argsort(lateralizations[..., 1, task_ind], axis=0)
        predicted = lateralizations[..., 1, task_ind][ind]
        actual = lateralizations[..., 0, task_ind][ind]
        ax.bar(x=x, height=actual, width=0.5, label="actual")
        ax.bar(x=x + 0.5, height=predicted, width=0.5, label="predicted")
        seaborn.despine(ax=ax, top=True, left=False, bottom=False, right=True)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Subjects")
        ax.set_ylabel(r"RIGHT $\leftarrow$ Lateralization Index $\rightarrow$ LEFT")
        ax.legend()
        ax.set_title(" ".join((task, contrast)))
        fig.savefig(output_dir + '/lateralization_{task}_{name}_bar.png'.format(task=task, name=contrast))

        fig, ax = plt.subplots()
        seaborn.regplot(x=predicted, y=actual, color="C3", ax=ax)
        ax.set_xlabel("Predicted lateralization index")
        ax.set_ylabel("Actual lateralization index")
        fig.savefig(output_dir + '/lateralization_{task}_{name}_scatter.png'.format(task=task, name=contrast))

        correlation, p_value = pearsonr(predicted, actual)
        stats.append((task, contrast, correlation, p_value))

    if stats_filename is not None:
        stats_df = pd.DataFrame(stats, columns=["Task", "Contrast", "Correlation", "P-Value"])
        stats_df.to_csv(stats_filename)


if __name__ == "__main__":
    main()
