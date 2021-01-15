#!/usr/bin/env python
import os
import argparse
from multiprocessing import Pool
from functools import partial
import numpy as np
import nibabel as nib
from fcnn.utils.nipy.ggmixture import GGGM
from fcnn.utils.wquantiles.wquantiles import quantile_1D
from fcnn.utils.hcp import get_metric_data, extract_cifti_scalar_map_names
from nilearn.plotting import plot_surf_stat_map
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", nargs="+", default=[""])
    parser.add_argument("--metric_name", required=True)
    parser.add_argument("--input_name", default="struct14")
    parser.add_argument("--hcp_dir", default="/work/aizenberg/dgellis/HCP/HCP_1200")
    parser.add_argument("--prediction_dir",
                        default="/work/aizenberg/dgellis/fCNN/predictions/v4_{input}_unet_ALL-TAVOR_2mm_v2_pt_test")
    parser.add_argument("--output_dir",
                        default=os.path.join("/work/aizenberg/dgellis", "fCNN", "predictions", "figures", "test",
                                             "weighted", "struct14", "statmaps"))
    parser.add_argument("--group_avg",
                        default="/work/aizenberg/dgellis/fCNN/"
                                "v4_average_test_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii")
    parser.add_argument("--surface_template",
                        default="/work/aizenberg/dgellis/fCNN/v4_training.{hemi}.inflated.32k_fs_LR.surf.gii")
    parser.add_argument("--sulc",
                        default="/work/aizenberg/dgellis/fCNN/v4_average_training.sulc.32k_fs_LR.dscalar.nii")
    parser.add_argument("--domain", default="ALL")
    return parser.parse_args()


def g2gm_threshold(data, iterations=1000):
    model = GGGM()
    membership = model.estimate(data, niter=iterations)
    lower_threshold = quantile_1D(data, membership[..., 0], 0.5)
    upper_threshold = quantile_1D(data, membership[..., 2], 0.5)
    thresholded_data = np.zeros((2,) + data.shape, np.int)
    thresholded_data[0][data >= upper_threshold] = 1
    thresholded_data[1][data <= lower_threshold] = 1
    return thresholded_data


def plot_data(data, surface_fn, sulc_data, title, hemi="left", output_file=None):
    data_threshold_mask = np.any(g2gm_threshold(data), axis=0)
    data_thresholded = data * data_threshold_mask
    figs = list()
    for view in ("lateral", "medial"):
        for positive_only in (True, False):
            if output_file is not None:
                _output_file = output_file.format(view=view)
                if positive_only:
                    _output_file = _output_file.replace(".png", "_positives.png")
                print(_output_file)
            else:
                _output_file = output_file
            if positive_only:
                _data_to_plot = np.copy(data_thresholded)
                _data_to_plot[data_thresholded < 0] = 0
            else:
                _data_to_plot = data_thresholded
            fig = plot_surf_stat_map(surface_fn, _data_to_plot, threshold=0.01, bg_map=-sulc_data, title=title,
                                     hemi=hemi, output_file=None, view=view)
            fig.savefig(_output_file, bbox_inches="tight")
            plt.close(fig)
            figs.append(fig)
    return figs


def compare_data(actual, predicted, group_avg, sulc, surface_fn, metric_name, hemi="left",
                 subject_id=None, sulc_name=None, output_template=None):
    surface_names = ["Cortex" + hemi.capitalize()]
    data = list()
    for image in (actual, predicted, group_avg):
        if image:
            try:
                data.append(np.ravel(get_metric_data([image], [[metric_name]], surface_names, None)))
            except ValueError:
                data.append(np.ravel(get_metric_data([image], [[metric_name.split(" ")[-1]]], surface_names, None)))
        else:
            data.append(None)
    a, p, g = data
    if sulc_name is None:
        sulc_name = extract_cifti_scalar_map_names(sulc)[0]
    sulc_data = np.ravel(get_metric_data([sulc], [[sulc_name]], surface_names, subject_id))
    for d, n in ((a, "actual"), (p, "predicted"), (g, "group average")):
        if d is not None:
            if output_template is not None:
                output_filename = output_template.format(subject=subject_id, task=metric_name, method=n,
                                                         hemi=hemi, view="{view}").replace(" ", "_")
            else:
                output_filename = None
            figs = plot_data(d, surface_fn, sulc_data, title=n, hemi=hemi, output_file=output_filename)


def main():
    namespace = parse_args()
    if not os.path.exists(namespace.output_dir):
        os.makedirs(namespace.output_dir)
    if len(namespace.subject) > 1:
        pool = Pool(len(namespace.subject))
        func = partial(visualize_subject_contrast, contrast=namespace.metric_name,
                       prediction_dir=namespace.prediction_dir,
                       input_name=namespace.input_name, output_dir=namespace.output_dir, hcp_dir=namespace.hcp_dir,
                       domain=namespace.domain, sulc_fn=namespace.sulc,
                       group_avg_fn=namespace.group_avg, surface_template=namespace.surface_template)
        pool.map(func=func, iterable=namespace.subject)
    else:
        visualize_subject_contrast(namespace.subject[0], namespace.metric_name, namespace.prediction_dir,
                                   namespace.input_name, namespace.hcp_dir, namespace.domain, namespace.output_dir,
                                   namespace.group_avg, surface_template=namespace.surface_template,
                                   sulc_fn=namespace.sulc)


def visualize_subject_contrast(subject, contrast, prediction_dir, input_name, hcp_dir, domain, output_dir,
                               group_avg_fn, surface_template=None, sulc_fn=None):
    prediction_dir = prediction_dir.format(input=input_name)

    prediction_basename = os.path.basename(prediction_dir).replace("_test", "")
    input_basename = input_name + "_normalized"

    actual_fn = os.path.join(hcp_dir, subject, "T1w", "Results", "tfMRI_" + domain,
                             "tfMRI_{domain}_hp200_s2_level2.feat".format(domain=domain),
                             "{subject}_tfMRI_ALL_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii".format(
                                 subject=subject))
    predicted_fn = os.path.join(prediction_dir,
                                "{subject}_model_{prediction}_{input}.midthickness.dscalar.nii".format(
                                    subject=subject, prediction=prediction_basename, input=input_basename))
    if sulc_fn is None:
        sulc_fn = os.path.join(hcp_dir, subject, "MNINonLinear", "fsaverage_LR32k",
                               "{subject}.sulc.32k_fs_LR.dscalar.nii".format(subject=subject))

    if surface_template is None:
        surface_template = os.path.join(hcp_dir, subject, "MNINonLinear", "fsaverage_LR32k",
                                        "{subject}.{hemi}.inflated.32k_fs_LR.surf.gii")

    if subject:
        actual = nib.load(actual_fn)
        predicted = nib.load(predicted_fn)
    else:
        actual = None
        predicted = None
    group_avg = nib.load(group_avg_fn)
    sulc = nib.load(sulc_fn)

    for hemi_full in ("left", "right"):
        hemi_letter = hemi_full[0].upper()
        surface_fn = surface_template.format(subject=subject, hemi=hemi_letter)
        compare_data(actual=actual, predicted=predicted, group_avg=group_avg, sulc=sulc, surface_fn=surface_fn,
                     metric_name=contrast, subject_id=subject, hemi=hemi_full,
                     output_template=os.path.join(output_dir, input_name +
                                                  "_{subject}_{task}_{method}_{hemi}_{view}_zstat_thresholded.png"))


if __name__ == "__main__":
    main()
