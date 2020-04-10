import subprocess
import os
import nibabel as nib
import numpy as np
from fcnn.utils.utils import load_json, update_progress
from fcnn.utils.hcp import extract_cifti_scalar_data
import seaborn
import pandas as pd
import matplotlib.pyplot as plt
import glob


def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def run_command(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)


def compute_mean_absolute_error(data1, data2, axis=-1):
    return np.mean(np.abs(data1 - data2), axis=axis)


def extract_cifti_scalar_data_for_metric_names(cifti, metric_names):
    return np.array([extract_cifti_scalar_data(cifti, map_name) for map_name in metric_names])


def compute_errors(actual, predictions, metric_names, error_func=compute_mean_absolute_error):
    # For each domain I am going to have the actual maps, group average for that domain, and 3 different predictions
    # from the maps trained on all the domains.
    # I am going to need to know which tasks from the "ALL" predictions to compare for each domain.
    # To do this, I will load the label file for that domain and then extract only the data I need from the ALL
    # domains.
    actual_data = extract_cifti_scalar_data_for_metric_names(actual, metric_names)
    errors = list()
    for prediction in predictions:
        predicted_data = extract_cifti_scalar_data_for_metric_names(prediction, metric_names)
        errors.append(error_func(actual_data, predicted_data))
    return np.array(errors)


def compute_subject_errors(actual_template, pred_template, other_templates, subject, tasks, group_average,
                           metric_names_template):
    errors = list()
    actual = nib.load(actual_template.format(subject=subject))

    other_predictions = list()
    for other_template in other_templates:
        other_predictions.append(nib.load(other_template.format(subject=subject)))

    for task in tasks:
        predictions = [group_average, nib.load(pred_template.format(subject=subject, task=task))]
        predictions.extend(other_predictions)
        metric_names = read_namefile(metric_names_template.format(task=task))
        errors.append(compute_errors(actual, predictions, metric_names))
    return np.concatenate(errors, axis=-1)


def compute_errors_for_all_subjects(actual_template, pred_template, other_templates, tasks, group_average_filename,
                                    metric_names_template, subjects):
    group_average = nib.load(group_average_filename)
    errors = list()
    subjects_key = list()
    for subject in subjects:
        try:
            errors.append(compute_subject_errors(actual_template, pred_template, other_templates, subject, tasks,
                                                 group_average, metric_names_template))
            subjects_key.append(subject)
        except FileNotFoundError as fnf_error:
            print(fnf_error)
    # output array shape is (n_subjects, n_predictions, n_contrasts)
    return np.array(errors), subjects_key


def compute_group_average(output_filename, subjects, cifti_template, overwrite=False):
    if overwrite or not os.path.exists(output_filename):
        cmd = ["wb_command", "-cifti-average", output_filename]
        for subject in subjects:
            cifti_filename = cifti_template.format(subject=subject)
            if os.path.exists(cifti_filename):
                cmd.extend(["-cifti", cifti_filename])
        run_command(cmd)


def main():
    # I want to compare the models that were trained on the individual domains to the models that were trained on
    # all of the domains at once. There are three models trained on all the domains: t1t2, struct6, and struct14.
    # There are seven task domains with one model for each domain.
    # It will also be nice to compare the results to the group average.
    # I would also like the errors to be plotted per task.

    # First, define which subjects we will be looking at
    subjects_config = load_json("/home/aizenberg/dgellis/fCNN/data/subjects_v4.json")
    group_average_key = "training"  # This will be used to create a group average mapping
    test_group = "validation"   # This group will be used to get the error of the predictions

    # Second define the information for the predictions from the models that were trained on individual task domains.
    tasks = ["MOTOR", "LANGUAGE", "WM", "SOCIAL", "RELATIONAL", "EMOTION", "GAMBLING"]
    struct_basename_template = "struct{}_normalized"
    struct6_basename = struct_basename_template.format(6)
    struct14_basename = struct_basename_template.format(14)
    t1t1_basename = "T1T2w_acpc_dc_restore_brain"
    prediction_template = os.path.join("/work/aizenberg/dgellis/fCNN",
                                       "predictions/v4_{input_name}_unet_{task}_2mm_v1_pt",
                                       "{subject}_model_v4_{input_name}_unet_{task}_2mm_v1_pt_{input_basename}.midthickness.dscalar.nii")

    # The other predictions are all under the "ALL-TAVOR" task name
    struct6_all_template = prediction_template.format(input_name="struct6", input_basename=struct6_basename,
                                                      task="ALL-TAVOR", subject="{subject}")
    struct14_all_template = prediction_template.format(input_name="struct14", input_basename=struct14_basename,
                                                       task="ALL-TAVOR", subject="{subject}")
    t1t2_all_template = prediction_template.format(input_name="t1t2", input_basename=t1t1_basename,
                                                      task="ALL-TAVOR", subject="{subject}")
    all_templates = [t1t2_all_template, struct6_all_template, struct14_all_template]

    ds_prediction_template = prediction_template.format(input_name="struct6", task="{task}", subject="{subject}",
                                                        input_basename=struct6_basename)

    predictions_key = ["group_average", "struct6-domain-specific", "t1t2-all", "struct6-all", "struct14-all"]

    # It is going to be less IO to compute the errors per subject for all the task domains for that subject

    # Define the information necessary to get the actual task maps
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    cifti_template = os.path.join(hcp_dir, "{subject}", "T1w", "Results", "tfMRI_{task}",
                                  "tfMRI_{task}_hp200_s2_level2.feat",
                                  "{subject}_tfMRI_{task}_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii")
    cifti_template = cifti_template.format(subject="{subject}", task="ALL")

    group_average_filename = os.path.join("/work/aizenberg/dgellis/fCNN",
                                          "v4_tfMRI_group_average_errors_level2_zstat_hp200_s2_TAVOR.midthickness.dscalar.nii")

    metric_names_template = "/home/aizenberg/dgellis/fCNN/data/labels/{task}-TAVOR_name-file.txt"

    compute_group_average(group_average_filename, subjects_config[group_average_key], cifti_template)

    prediction_errors_filename = os.path.join("/work/aizenberg/dgellis/fCNN/predictions",
                                              "v4_model_error_comparison.npy")

    if not os.path.exists(prediction_errors_filename):

        errors, subjects = compute_errors_for_all_subjects(cifti_template, ds_prediction_template, all_templates, tasks,
                                                           group_average_filename, metric_names_template,
                                                           subjects_config[test_group])

        np.save(prediction_errors_filename, errors)
        np.save(prediction_errors_filename.replace(".npy", "_subjects.npy"), subjects)
    else:
        errors = np.load(prediction_errors_filename)
        subjects = np.load(prediction_errors_filename.replace(".npy", "_subjects.npy"))

    print(errors.shape)

    all_metric_names = list()
    for task in tasks:
        all_metric_names.extend(read_namefile(metric_names_template.format(task=task)))

    # convert the array to a dataframe that works with Seaborn
    df_rows = list()
    for subject_index, subject in enumerate(subjects):
        for prediction_index, prediction_name in enumerate(predictions_key):
            for metric_index, metric_name in enumerate(all_metric_names):
                df_rows.append([subject, prediction_name, metric_name, errors[subject_index, prediction_index,
                                                                              metric_index]])

    errors_df = pd.DataFrame(df_rows, columns=["Subject", "Label", "Task", "Error"])
    fig, ax = plt.subplots(figsize=(6, errors.shape[-1] * 0.8))
    seaborn.barplot(data=errors_df, x="Error", y="Task", hue="Label")
    fig.savefig("/work/aizenberg/dgellis/fCNN/predictions/figures/model_comparison_mean_average_error.png")


if __name__ == "__main__":
    main()
