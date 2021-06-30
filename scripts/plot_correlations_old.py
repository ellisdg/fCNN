import seaborn
import matplotlib.pyplot as plt
import pandas as pd
import glob
import os
from fcnn.utils.utils import load_json
from fcnn.utils.hcp import get_metric_data, extract_cifti_scalar_data
from scipy.stats import pearsonr, ks_2samp
import nibabel as nib
import numpy as np
import sys


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
    task = sys.argv[1]
    config_filename = "/home/aizenberg/dgellis/fCNN/data/v4_struct6_unet_{task}-TAVOR_2mm_v1_pt_config.json".format(task=task)
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    config = load_json(config_filename)
    target_basename = config["target_basenames"]
    prediction_dir = sys.argv[2]
    output_dir = sys.argv[3]
    correlations_file = os.path.join(prediction_dir, "correlations.npy")
    all_prediction_images = glob.glob(os.path.join(prediction_dir, "*.dscalar.nii"))
    target_images = list()
    structure_names = ["CortexLeft", "CortexRight"]
    hemispheres = ["L", "R"]
    surface_template = "T1w/fsaverage_LR32k/{subject}.{hemi}.{surf}.32k_fs_LR.surf.gii"
    all_surfaces = list()
    prediction_images = list()
    surf_name = "midthickness"
    metric_filename = "/home/aizenberg/dgellis/fCNN/data/labels/{task}-TAVOR_name-file.txt".format(task=task)
    metric_names = read_namefile(metric_filename)

    for p_image_fn in all_prediction_images:
        if "target" not in p_image_fn:
            sid = os.path.basename(p_image_fn).split("_")[0]
            target_fn = os.path.join(hcp_dir, sid, target_basename.format(sid)).replace(".nii.gz",
                                                                                        ".{}.dscalar.nii").format(
                surf_name)
            target_images.append(target_fn)
            prediction_images.append(p_image_fn)
            all_surfaces.append([os.path.join(hcp_dir, sid, surface_template.format(subject=sid, hemi=hemi,
                                                                                    surf=surf_name))
                                 for hemi in hemispheres])

    correlations = np.load(correlations_file)
    seaborn.set_style('white')
    corr_matrices = np.asarray(correlations)[..., 0]
    vmin = corr_matrices.min()
    vmax = corr_matrices.max()

    lateralization_file = os.path.join(prediction_dir, "lateralization.npy")
    if not os.path.exists(lateralization_file):
        lateralization = list()
        for p, t in zip(prediction_images, target_images):
            _p0 = np.max(get_metric_data([nib.load(p)], [metric_names], structure_names[:1], None), axis=0)
            _p1 = np.max(get_metric_data([nib.load(p)], [metric_names], structure_names[1:], None), axis=0)
            _t0 = np.max(get_metric_data([nib.load(t)], [metric_names], structure_names[:1], None), axis=0)
            _t1 = np.max(get_metric_data([nib.load(t)], [metric_names], structure_names[1:], None), axis=0)
            lateralization.append([_p0 - _p1, _t0 - _t1])
        lateralization = np.asarray(lateralization)

        np.save(lateralization_file, lateralization)
    else:
        lateralization = np.load(lateralization_file)

    for task_ind in range(lateralization.shape[-1]):
        fig_width = 18
        fig_height = 6
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        x = np.arange(lateralization.shape[0])
        ind = np.argsort(lateralization[..., 1, task_ind], axis=0)
        ax.bar(x=x, height=lateralization[..., 1, task_ind][ind], width=0.5, label="actual")
        ax.bar(x=x + 0.5, height=lateralization[..., 0, task_ind][ind], width=0.5, label="predicted")
        seaborn.despine(ax=ax, top=True, left=False, bottom=False, right=True)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel("Subjects")
        ax.set_ylabel(r"RIGHT $\leftarrow$ Lateralization Index $\rightarrow$ LEFT")
        ax.legend()
        ax.set_title(metric_names[task_ind])
        fig.savefig(output_dir + '/lateralization_{task}_{name}_bar.png'.format(task=task, name=metric_names[task_ind]))


    n_plots = len(config['metric_names'])
    plots_per_row = 4
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
    for i, metric_name in enumerate(config['metric_names']):
        title = "{task} ".format(task=task) + metric_name.split("level2_")[-1].split("_hp200")[0]
        names.append(title)
        ax = np.ravel(axes)[i]
        if i + 1 == len(config['metric_names']):
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
    fig.savefig(output_dir + "/correlation_matrices_{task}.png".format(task=task))
    hist_fig.savefig(output_dir + "/correlation_matrices_histograms_{task}.png".format(task=task))
    cbar_fig.savefig(output_dir + "/correlation_matrices_{task}_colorbar.png".format(task=task), bbox_inches="tight")

    avg_fig, avg_ax = plt.subplots(figsize=(column_height, row_height))
    avg_corr = corr_matrices.mean(axis=-1)
    seaborn.heatmap(data=avg_corr, ax=avg_ax, xticklabels=False, yticklabels=False, cbar=False, vmax=vmax, vmin=vmin,
                    cmap=cmap)
    avg_ax.set_title(task)
    avg_ax.set_ylabel("subjects (predicted)")
    avg_ax.set_xlabel("subjects (actual)")
    avg_fig.savefig(output_dir + "/correlation_matrix_average_{task}.png".format(task=task))
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
    avg_ax.set_title(task)
    avg_ax.set_ylabel("subjects (predicted)")
    avg_ax.set_xlabel("subjects (actual)")
    avg_fig.savefig(output_dir + "/correlation_matrix_average_normalized_{task}.png".format(task=task))

    fig, axes = plt.subplots(nrows=n_rows, ncols=plots_per_row, figsize=(plots_per_row*column_height, n_rows*row_height))
    cbar_fig, cbar_ax = plt.subplots(figsize=(0.5, 5))

    for i, metric_name in enumerate(config['metric_names']):
        title = "{task} ".format(task=task) + metric_name.split("level2_")[-1].split("_hp200")[0]
        ax = np.ravel(axes)[i]
        if i + 1 == len(config['metric_names']):
            cbar = True
        else:
            cbar = False
        # seaborn.heatmap(data=corr_matrices[..., i], ax=ax, cbar=cbar, cbar_ax=cbar_ax, xticklabels=False, yticklabels=False,
        #                 cmap=cmap)
        seaborn.heatmap(data=normalize_correlation_matrix(corr_matrices[..., i], vmax, vmin, axes=(0, 1)), ax=ax, cbar=cbar,
                        cbar_ax=cbar_ax, xticklabels=False, yticklabels=False,
                        vmax=vmax, vmin=vmin, cmap=cmap)
        ax.set_title(title)
    fig.savefig(output_dir + "/correlation_matrices_normalized_{task}.png".format(task=task))

    fig, ax = plt.subplots()
    diagonal_mask = np.diag(np.ones(avg_corr.shape[0], dtype=bool))
    diag_values = avg_corr[diagonal_mask]
    extra_diag_values = avg_corr[diagonal_mask == False]
    # seaborn.distplot(extra_diag_values, norm_hist=True)
    # _ = seaborn.distplot(extra_diag_values, hist_kws=dict(density=True, stacked=True), ax=ax)
    _ = seaborn.distplot(extra_diag_values, kde_kws={"shade": True}, ax=ax, label="predicted vs other subjects")
    _ = seaborn.distplot(diag_values, kde_kws={"shade": True}, ax=ax, label="predicted vs actual")
    ax.set_title(task)
    ax.set_ylabel("Count")
    ax.set_xlabel("Correlation")
    ax.legend()
    fig.savefig(output_dir + "/average_correlation_histogram_{task}.png".format(task=task))
    d, p = ks_2samp(diag_values, extra_diag_values)
    print("D-value: {:.2f}\tp-value = {:.8f}".format(d, p))

    fig, ax = plt.subplots()
    seaborn.barplot(x=np.asanyarray(result)*100, y=names, ax=ax, color='C0')
    ax.set_xlabel("Self vs other increase (in %)")
    seaborn.despine(ax=ax, top=True)
    fig.savefig(output_dir + "/increase_correlation_over_mean_correlation_{task}.png".format(task=task), bbox_inches="tight")


if __name__ == "__main__":
    main()
