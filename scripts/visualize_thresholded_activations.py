#!/usr/bin/env python
import sys
import os
import numpy as np
import nibabel as nib
from fcnn.utils.nipy.ggmixture import GGGM
from fcnn.utils.wquantiles.wquantiles import quantile_1D
from fcnn.utils.hcp import get_metric_data
from nilearn.plotting import plot_surf_stat_map


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
    return plot_surf_stat_map(surface_fn, data_thresholded, threshold=0.01, bg_map=sulc_data,
                              title=title, hemi=hemi, output_file=output_file)


def compare_data(actual, predicted, group_avg, sulc, surface_fn, metric_name, hemi="left",
                 subject_id=None, sulc_name="{}_Sulc", output_template=None):
    surface_names = ["Cortex" + hemi.capitalize()]
    data = list()
    for image in (actual, predicted, group_avg):
        try:
            data.append(np.ravel(get_metric_data([image], [[metric_name]], surface_names, None)))
        except ValueError:
            data.append(np.ravel(get_metric_data([image], [[metric_name.split(" ")[-1]]], surface_names, None)))
    a, p, g = data
    sulc_data = np.ravel(get_metric_data([sulc], [[sulc_name]], surface_names, subject_id))
    for d, n in ((a, "actual"), (p, "predicted"), (g, "group average")):
        if output_template is not None:
            output_filename = output_template.format(subject=subject_id, task=metric_name, method=n,
                                                     hemi=hemi).replace(" ", "_")
            print(output_filename)
        else:
            output_filename = None
        fig = plot_data(d, surface_fn, sulc_data, title=n, hemi=hemi, output_file=output_filename)


def main():
    subject = str(sys.argv[1])
    contrast = str(sys.argv[2])
    input_name = str(sys.argv[3])

    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    prediction_dir = "/work/aizenberg/dgellis/fCNN/predictions/v4_{input}_unet_ALL-TAVOR_2mm_v2_pt_test"
    output_dir = os.path.join("/work/aizenberg/dgellis", "fCNN", "predictions", "figures", "test", "weighted",
                              "struct14", "statmaps")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    group_avg_fn = "/work/aizenberg/dgellis/fCNN/" \
                   "v4_tfMRI_group_average_errors_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii".format(
        subject=subject)
    domain = "ALL"
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
    sulc_fn = os.path.join(hcp_dir, subject, "MNINonLinear", "fsaverage_LR32k",
                           "{subject}.sulc.32k_fs_LR.dscalar.nii".format(subject=subject))

    actual = nib.load(actual_fn)
    predicted = nib.load(predicted_fn)
    group_avg = nib.load(group_avg_fn)
    sulc = nib.load(sulc_fn)

    for hemi_full in ("left", "right"):
        hemi_letter = hemi_full[0].upper()
        surface_fn = os.path.join(hcp_dir, subject, "T1w", "fsaverage_LR32k",
                                  "{subject}.{hemi}.inflated.32k_fs_LR.surf.gii".format(subject=subject,
                                                                                        hemi=hemi_letter))
        compare_data(actual=actual, predicted=predicted, group_avg=group_avg, sulc=sulc, surface_fn=surface_fn,
                     metric_name=contrast, subject_id=subject, hemi=hemi_full,
                     output_template=os.path.join(output_dir, input_name +
                                                  "_{subject}_{task}_{method}_{hemi}_zstat_thresholded.png"))


if __name__ == "__main__":
    main()
