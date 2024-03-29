import nibabel as nib
import os
import sys
from fcnn.utils.utils import load_json
from fcnn.utils.hcp import extract_cifti_scalar_data, new_cifti_scalar_like
import numpy as np
import subprocess


def main():
    model_name = sys.argv[1]
    cifti_type = "pscalar"
    prediction_dir = "/work/aizenberg/dgellis/fCNN/predictions"
    output_fn = os.path.join(prediction_dir, model_name + "_prediction_error.csv")
    smoothing_level = 4
    smoothing_name = "_s{}_".format(smoothing_level)
    task_name = "LANGUAGE"
    basename = "{subject_id}_tfMRI_{task_name}_level2_hp200_s{smoothing}_MSMAll.model_{model_name}_prediction.{cifti_type}.nii"
    basename = basename.format(task_name=task_name, smoothing=smoothing_level, model_name=model_name,
                               subject_id="{subject_id}", cifti_type=cifti_type)
    prediction_filename = os.path.join(prediction_dir, basename)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    lang_config_fn = os.path.join(data_dir, "trial_lowq_32_LS_LM_config.json")
    lang_config = load_json(lang_config_fn)
    brain_structures = ("CortexLeft", "CortexRight")
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    task_filename = os.path.join(hcp_dir, "{subject_id}", "MNINonLinear/Results",
                                 "tfMRI_{task_name}/tfMRI_{task_name}_hp200_s{smoothing_level}_level2_MSMAll.feat",
                                 "{subject_id}_tfMRI_{task_name}_level2_hp200_s{smoothing_level}_MSMAll.{cifti_type}.nii")

    output_average_fn = prediction_filename.format(subject_id="average").replace("." + cifti_type,
                                                                                 "_error." + cifti_type)
    cmd_args = ["wb_command", "-cifti-average", output_average_fn]

    for subject_id in lang_config['validation']:
        pred_scalar_filename = prediction_filename.format(subject_id=subject_id)
        output_fn = pred_scalar_filename.replace("." + cifti_type,
                                                  "_error." + cifti_type)
        if not os.path.exists(pred_scalar_filename):
            print("Does not exists:", pred_scalar_filename)
            continue
        cmd_args.extend(["-cifti", output_fn])
        if os.path.exists(output_fn):
            print("Already exists:", output_fn)
            continue
        print(subject_id)
        pred_scalar = nib.load(pred_scalar_filename)
        fmri_scalar_filename = task_filename.format(subject_id=subject_id,
                                                     task_name=task_name,
                                                     smoothing_level=smoothing_level,
                                                     cifti_type=cifti_type)
        fmri_scalar = nib.load(fmri_scalar_filename)
        subject_mae = list()
        subject_metric_names = list()
        for metric_name in lang_config['metric_names'][0]:
            print(metric_name)
            subject_metric_name = metric_name.format(subject_id)
            subject_metric_names.append(subject_metric_name)
            pred_metric_data = extract_cifti_scalar_data(pred_scalar, subject_metric_name)
            fmri_metric_data = extract_cifti_scalar_data(fmri_scalar, subject_metric_name)
            structure_mae = np.abs(fmri_metric_data - pred_metric_data)
            subject_mae.append(structure_mae)
        print(np.asarray(subject_mae).shape)
        output_scalar = pred_scalar.__class__(dataobj=np.asarray(subject_mae), header=pred_scalar.header)
        output_scalar.to_filename(output_fn)

    print(" ".join(cmd_args))
    subprocess.call(cmd_args)


if __name__ == "__main__":
    main()

