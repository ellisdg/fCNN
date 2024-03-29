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
        pred_dscalar_filename = prediction_filename.format(subject_id=subject_id)
        output_fn = pred_dscalar_filename.replace("." + cifti_type,
                                                  "_error." + cifti_type)
        if not os.path.exists(pred_dscalar_filename):
            print("Does not exists:", pred_dscalar_filename)
            continue
        cmd_args.extend(["-cifti", output_fn])
        if os.path.exists(output_fn):
            print("Already exists:", output_fn)
            continue
        print(subject_id)
        pred_dscalar = nib.load(pred_dscalar_filename)
        fmri_dscalar_filename = task_filename.format(subject_id=subject_id,
                                                     task_name=task_name,
                                                     smoothing_level=smoothing_level,
                                                     cifti_type=cifti_type)
        fmri_dscalar = nib.load(fmri_dscalar_filename)
        fmri_bmaxis = fmri_dscalar.header.get_axis(1)
        pred_bmaxis = pred_dscalar.header.get_axis(1)
        subject_mae = list()
        subject_metric_names = list()
        for metric_name in lang_config['metric_names'][0]:
            print(metric_name)
            subject_metric_name = metric_name.format(subject_id)
            subject_metric_names.append(subject_metric_name)
            metric_mae = list()
            for brain_structure in brain_structures:
                print(brain_structure)
                pred_metric_data = extract_cifti_scalar_data(pred_dscalar, subject_metric_name,
                                                             brain_structure_name=brain_structure)
                fmri_metric_data = extract_cifti_scalar_data(fmri_dscalar, subject_metric_name,
                                                             brain_structure_name=brain_structure)
                pred_mask = np.in1d(pred_bmaxis.vertex[pred_bmaxis.name == pred_bmaxis.to_cifti_brain_structure_name(brain_structure)], 
                                    fmri_bmaxis.vertex[fmri_bmaxis.name == fmri_bmaxis.to_cifti_brain_structure_name(brain_structure)])

                structure_mae = np.abs(fmri_metric_data - pred_metric_data[pred_mask])
                metric_mae.extend(structure_mae)
                print(np.asarray(metric_mae).shape)
            print(np.asarray(metric_mae).shape)
            subject_mae.append(metric_mae)
        print(np.asarray(subject_mae).shape)
        output_dscalar = new_cifti_scalar_like(array=np.asarray(subject_mae),
                                               structure_names=lang_config["surface_names"],
                                               scalar_names=subject_metric_names,
                                               reference_cifti=fmri_dscalar,
                                               almost_equals_decimals=0)
        output_dscalar.to_filename(output_fn)

    print(" ".join(cmd_args))
    subprocess.call(cmd_args)


if __name__ == "__main__":
    main()

