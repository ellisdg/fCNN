import subprocess
import os
import nibabel as nib
import numpy as np
from fcnn.utils.utils import load_json, update_progress
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import glob


def run_command(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)


def compute_error(cifti1, cifti2):
    return np.mean(np.abs(np.array(cifti1.dataobj) - np.array(cifti2.dataobj)))


def main():
    config = load_json("/home/aizenberg/dgellis/fCNN/data/subjects_v4.json")
    tasks = ["MOTOR", "LANGUAGE", "WM", "SOCIAL", "RELATIONAL", "EMOTION", "GAMBLING"]
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    group_average_keys = ["training"]
    test_group = "validation"
    cifti_template = os.path.join(hcp_dir, "{subject}", "T1w", "Results", "tfMRI_{task}",
                                  "tfMRI_{task}_hp200_s2_level2.feat",
                                  "{subject}_tfMRI_{task}_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii")
    group_average_template = os.path.join("/work/aizenberg/dgellis/fCNN",
                                          "{subject}_tfMRI_{task}_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii")
    prediction_template = os.path.join("/work/aizenberg/dgellis/fCNN",
                                       "predictions/v4_struct6_unet_{task}_2mm_v1_pt",
                                       "{subject}_model_v4_struct6_unet_{task}*_2mm_v1_pt_struct6_normalized.midthickness.dscalar.nii")

    group_average_errors = list()
    prediction_errors = list()
    group_average_errors_filename = os.path.join("/work/aizenberg/dgellis/fCNN",
                                                 "v4_tfMRI_group_average_errors_level2_zstat_hp200_s2_TAVOR.midthickness.npy")
    prediction_errors_filename = os.path.join("/work/aizenberg/dgellis/fCNN/predictions",
                                              "v4_model_v4_struct6_unet_errors_2mm_v1_pt_struct6_normalized.npy")

    if not os.path.exists(group_average_errors_filename) and not os.path.exists(prediction_errors_filename):
        for task in tasks:
            # make average
            group_average_filename = group_average_template.format(subject="v4training", task=task)
            if not os.path.exists(group_average_filename):
                cmd = ["wb_command", "-cifti-average", group_average_filename]
                for key in group_average_keys:
                    for subject in config[key]:
                        cifti_filename = cifti_template.format(task=task, subject=subject)
                        if os.path.exists(cifti_filename):
                            cmd.extend(["-cifti", cifti_filename])
                run_command(cmd)
            group_average = nib.load(group_average_filename)
            # Compute error from average
            group_average_task_errors = list()
            predicted_task_errors = list()
            for subject in config[test_group]:
                cifti_filename = cifti_template.format(task=task, subject=subject)
                if not os.path.exists(cifti_filename):
                    continue
                cifti = nib.load(cifti_filename)
                prediction_wildcard = prediction_template.format(subject=subject, task=task)
                try:
                    prediction_filename = glob.glob(prediction_wildcard)[0]
                except IndexError:
                    print(prediction_wildcard)
                    continue
                predicted_cifti = nib.load(prediction_filename)
                group_average_task_errors.append(compute_error(group_average, cifti))
                predicted_task_errors.append(compute_error(predicted_cifti, cifti))
            group_average_errors.append(group_average_task_errors)
            prediction_errors.append(predicted_task_errors)
        np.save(group_average_errors_filename, group_average_errors)
        np.save(prediction_errors_filename, prediction_errors)
    else:
        group_average_errors = np.load(group_average_errors_filename, allow_pickle=True)
        prediction_errors = np.load(prediction_errors_filename, allow_pickle=True)

    rows = list()
    for task, ga_errors, p_errors in zip(tasks, group_average_errors, prediction_errors):
        for errors, label in zip((ga_errors, p_errors), ("Group Average", "Prediction")):
            for i, error in enumerate(errors):
                rows.append([error, label, task])
                update_progress((i + 1)/len(errors), message=task + label)
    errors_df = pd.DataFrame(rows, columns=["MAE", "Label", "Task"])
    fig, ax = plt.subplots()
    seaborn.barplot(data=errors_df, x="MAE", y="Task", hue="Label")
    fig.savefig("/work/aizenberg/dgellis/fCNN/predictions/figures/mean_average_error.png")


if __name__ == "__main__":
    main()
