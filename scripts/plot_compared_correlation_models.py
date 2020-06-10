import seaborn
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
from functools import reduce
import pandas as pd


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
    output_dir = os.path.abspath(sys.argv[2])
    labels = sys.argv[3].split(",")

    temp_correlations = list()
    subjects = list()
    model_labels = list()
    method_labels = list()

    for c_file, label in zip(correlation_files, labels):
        corr = np.load(c_file)
        print(c_file, corr.shape, label)
        subs = np.load(c_file.replace(".npy", "_subjects.npy"))
        subjects.append(subs)
        temp_correlations.append(corr)
        model_label, method_label = label.split("-")
        method_labels.append(method_label)
        model_labels.append(model_label)

    all_subjects = reduce(np.intersect1d, subjects)
    correlations = list()
    for sub_list, corr in zip(np.copy(subjects), temp_correlations):
        s, i, i_all = np.intersect1d(sub_list, all_subjects, return_indices=True)
        np.testing.assert_equal(s, all_subjects)
        correlations.append(corr[i][:, i])
    correlations = np.mean(np.asarray(correlations)[..., 0], axis=3)
    print(correlations.shape)

    plot(correlations, model_labels, method_labels, output_dir,
         metric_func=self_vs_other_correlation, output_filename="{}_increase_correlation_over_mean_correlation_average",
         xlabel="Self vs other increase (in %)")


def self_vs_other_correlation(corr_matrix):
    diagonal_mask = np.diag(np.ones(corr_matrix.shape[0], dtype=bool))
    diag_values = corr_matrix[diagonal_mask]
    extra_diag_values = corr_matrix[diagonal_mask == False]
    return ((diag_values.mean() - extra_diag_values.mean())/extra_diag_values.mean()) * 100


def plot(corr_matrices, model_labels, method_labels, output_dir, metric_func=self_vs_other_correlation,
         output_filename="{}_increase_correlation_over_mean_correlation", xlabel="Self vs other increase (in %)"):
    result = list()
    for i, (model_name, method_name) in enumerate(zip(model_labels, method_labels)):
        corr_matrix = corr_matrices[i]
        value = metric_func(corr_matrix)
        result.append([model_name, method_name, value])
    df = pd.DataFrame(result, columns=["Model", "Method", "Value"])

    w = 6
    width = 0.4
    gap = 0.
    h = (width + gap) * len(df)
    fig, ax = plt.subplots(figsize=(w, h))
    seaborn.barplot(data=df, x="Value", y="Method", hue="Model", ax=ax)
    ax.set_xlabel(xlabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    seaborn.despine(ax=ax, top=True)
    save_fig(fig, os.path.join(output_dir, output_filename.format("_".join(method_labels + model_labels))),
             bbox_inches="tight")


if __name__ == "__main__":
    main()
