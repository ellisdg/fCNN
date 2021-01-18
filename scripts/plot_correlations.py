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
    parser.add_argument('--correlation_filename', nargs="+", required=True)
    parser.add_argument('--labels', nargs="*")
    parser.add_argument('--task_names', default="/home/aizenberg/dgellis/fCNN/data/labels/ALL-TAVOR_name-file.txt")
    parser.add_argument('--level', default="overall", choices=["overall", "domain", "task"])
    parser.add_argument('--stats_filename', default=None)
    parser.add_argument('--extensions', nargs="*", default=[".pdf"])
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
              p_value_fontsize='medium', legend=False, legend_loc="upper left", legend_fontsize="small"):
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
        ax.legend(loc=legend_loc, fontsize=legend_fontsize)
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


def plot_correlation_panel(corr_matrix, column_width=3, row_height=3, norm_vmax=3, norm_vmin=-3, output_dir=".",
                           extensions=(".pdf",)):
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

    save_fig(overall_fig, output_dir + "/correlation_matrix_overall", bbox_inches="tight", extensions=extensions)
    save_fig(overall_norm_fig, output_dir + "/correlation_matrix_overall_normalized", bbox_inches="tight",
             extensions=extensions)
    save_fig(overall_cbar_fig, output_dir + "/correlation_matrix_overall_colorbar", bbox_inches="tight",
             extensions=extensions)
    save_fig(norm_cbar_fig, output_dir + "/correlation_matrix_overall_normalized_colorbar", bbox_inches="tight",
             extensions=extensions)

    overall_hist_fig, overall_hist_ax = plt.subplots()
    d, p = plot_hist(corr_matrix, overall_hist_ax, title=None, plot_p_value=True, legend=True)
    print("D-value: {:.2f}\tp-value = {:.2e}".format(d, p))
    save_fig(overall_hist_fig, output_dir + "/correlation_overall_histogram", bbox_inches="tight",
             extensions=extensions)

    _ = plot_hist(corr_matrix, _overall_hist_ax, title=None, plot_p_value=True, legend=True)

    save_fig(overall_all_fig, output_dir + "/correlation_overall_panel", bbox_inches="tight", extensions=extensions)


def mean_correlations(array, axis=None):
    # transform the correlations (r) into z values
    z = np.arctanh(array)
    # take the mean of the z values
    z_mean = np.mean(z, axis=axis)
    # convert the mean back into r values
    return np.tanh(z_mean)


def mean_diagonal(matrix):
    return mean_correlations(matrix.diagonal())


def normalized_mean_diagonal(matrix, new_max=3, new_min=-3):
    return mean_diagonal(normalize_correlation_matrix(matrix=matrix, new_max=new_max, new_min=new_min))


def self_vs_other_correlation(corr_matrix):
    diagonal_mask = np.diag(np.ones(corr_matrix.shape[0], dtype=bool))
    diag_values = corr_matrix[diagonal_mask]
    extra_diag_values = corr_matrix[diagonal_mask == False]
    return ((mean_correlations(diag_values) - mean_correlations(extra_diag_values))/
            mean_correlations(extra_diag_values)) * 100


def plot_self_vs_other_correlations(corr_matrices, model_labels, method_labels, output_dir,
                                    metric_func=self_vs_other_correlation,
                                    output_filename="{}_increase_correlation_over_mean_correlation",
                                    xlabel="Self vs other increase (in %)",
                                    extensions=(".pdf",)):
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
    if "{}" in output_filename:
        output_filename = "_".join(method_labels + model_labels)
    save_fig(fig, os.path.join(output_dir, output_filename), bbox_inches="tight", extensions=extensions)


def compare_overall_correlation_models_and_methods(correlation_files, labels, output_directory, extensions=(".pdf",)):
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
    correlations = np.asarray(correlations)[..., 0]
    print(correlations.shape)

    plot_self_vs_other_correlations(correlations, model_labels, method_labels, output_directory,
                                    metric_func=self_vs_other_correlation,
                                    output_filename="compared_increase_correlation_over_mean_correlation_average",
                                    xlabel="Self vs other increase (in %)",
                                    extensions=extensions)

    plot_self_vs_other_correlations(correlations, model_labels, method_labels, output_directory,
                                    metric_func=mean_diagonal,
                                    output_filename="compared_mean_correlation",
                                    xlabel="Mean Correlation", extensions=extensions)

    plot_self_vs_other_correlations(correlations, model_labels, method_labels, output_directory,
                                    metric_func=normalized_mean_diagonal,
                                    output_filename="compared_normalized_mean_correlation",
                                    xlabel="Normalized Mean Correlation", extensions=extensions)


def plot_per_domain(corr_matrices, domains, metric_names, method_labels, labels, output_dir, average_per_domain=True,
         metric_func=self_vs_other_correlation, output_filename="{}_increase_correlation_over_mean_correlation",
         xlabel="Self vs other increase (in %)", extensions=(".pdf",)):
    names = list()
    result = list()
    titles = list()
    for i, (domain, metric_name) in enumerate(zip(domains, metric_names)):
        title = domain + " " + metric_name
        names.append(title)
        corr_matrix = corr_matrices[..., i]
        result.append(metric_func(corr_matrix))
        titles.append(title)
    if average_per_domain:
        rows = list()
        domains = np.asarray(domains)
        method_labels = np.asarray(method_labels)
        result = np.asarray(result)
        for domain in np.unique(domains):
            domain_mask = domains == domain
            for method in labels:
                method_mask = method_labels == method
                value = result[np.logical_and(domain_mask, method_mask)].mean()
                rows.append([method, domain, value])
        df = pd.DataFrame(rows, columns=["Method", "Task", "Value"])
    else:
        data = np.squeeze(np.dstack([method_labels, domains, names, titles, np.asarray(result)]))
        df = pd.DataFrame(data, columns=["Method", "Domain", "Contrast", "Task", "Value"])

    w = 6
    width = 0.4
    gap = 0.
    h = (width + gap) * len(df)
    fig, ax = plt.subplots(figsize=(w, h))
    seaborn.barplot(data=df, x="Value", y="Task", hue="Method", ax=ax)
    ax.set_xlabel(xlabel)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    seaborn.despine(ax=ax, top=True)
    save_fig(fig, os.path.join(output_dir, output_filename.format("_".join(labels))), bbox_inches="tight",
             extensions=extensions)


def compare_domain_correlation_models(correlation_files, name_file, labels, output_directory, average_per_domain=True):
    temp_correlations = list()
    metric_names = list()
    domains = list()
    subjects = list()
    method_labels = list()
    task_names = read_namefile(name_file)

    for c_file, label in zip(correlation_files, labels):
        corr = np.load(c_file)
        print(c_file, corr.shape)
        subs = np.load(c_file.replace(".npy", "_subjects.npy"))
        subjects.append(subs)
        temp_correlations.append(corr)
        for name in task_names:
            d, n = name.split(" ")
            domains.append(d)
            metric_names.append(n)
            method_labels.append(label)

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
    plot_per_domain(corr_matrices, domains, metric_names, method_labels, labels, average_per_domain=average_per_domain,
                    output_dir=output_directory, metric_func=self_vs_other_correlation,
                    output_filename="{}_increase_correlation_over_mean_correlation",
                    xlabel="Self vs other increase (in %)")


def main():
    seaborn.set_palette('muted')
    seaborn.set_style('whitegrid')
    args = parse_args()

    if args["stats_filename"] is None:
        stats_filename = os.path.join(args["output_dir"], args["level"] + "_correlation_stats.csv")
    else:
        stats_filename = args["stats_filename"]

    if args["level"] == "overall":
        if len(args["correlation_filename"]) > 1:
            # compare correlations
            compare_overall_correlation_models_and_methods(args["correlation_filename"], args["labels"],
                                                           args["output_dir"], extensions=args["extensions"])
        else:
            # plot correlation results for a single correlation file
            corr_matrix = np.load(args["correlation_filename"][0])[..., 0]
            plot_correlation_panel(corr_matrix=corr_matrix, output_dir=args["output_dir"], 
                                   extensions=args["extensions"])
    elif args["level"] == "domain":
        compare_domain_correlation_models(args["correlation_filename"], args["task_names"], args["labels"],
                                          args["output_dir"])
    elif args["level"] == "task":
        raise NotImplementedError("Level={}. Use 'plot_combined_correlations.py' instead.".format(args["level"]))
    else:
        raise NotImplementedError("Level={}".format(args["level"]))


if __name__ == "__main__":
    main()
