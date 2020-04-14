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


def main():
    seaborn.set_palette('muted')
    seaborn.set_style('whitegrid')
    correlation_files = sys.argv[1].split(",")
    name_files = sys.argv[2].split(",")
    output_dir = os.path.abspath(sys.argv[3])
    labels = sys.argv[4].split(",")

    temp_correlations = list()
    metric_names = list()
    tasks = list()
    subjects = list()
    method_labels = list()

    for c_file, n_file, label in zip(correlation_files, name_files, labels):
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
                method_labels.append(label)
        else:
            tasks.extend([task] * len(names))
            method_labels.extend([label] * len(names))
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
    names = list()
    result = list()
    titles = list()
    for i, (task, metric_name) in enumerate(zip(tasks, metric_names)):
        title = task + " " + metric_name
        names.append(title)
        corr_matrix = corr_matrices[..., i]
        diagonal_mask = np.diag(np.ones(corr_matrix.shape[0], dtype=bool))
        diag_values = corr_matrix[diagonal_mask]
        extra_diag_values = corr_matrix[diagonal_mask == False]
        result.append((diag_values.mean() - extra_diag_values.mean())/extra_diag_values.mean())
        titles.append(title)
    data = np.squeeze(np.dstack([method_labels, tasks, names, titles, np.asarray(result) * 100]))
    df = pd.DataFrame(data, columns=["Method", "Domain", "Contrast", "Task", "Value"])
    w = 6
    width = 0.4
    gap = 0.
    h = (width + gap) * len(names)
    fig, ax = plt.subplots(figsize=(w, h))
    seaborn.barplot(data=df, x="Value", y="Task", hue="Method")
    ax.set_xlabel("Self vs other increase (in %)")
    seaborn.despine(ax=ax, top=True)
    fig.savefig(output_dir + "/{}_increase_correlation_over_mean_correlation.png".format("_".join(method_labels)),
                bbox_inches="tight")


if __name__ == "__main__":
    main()
