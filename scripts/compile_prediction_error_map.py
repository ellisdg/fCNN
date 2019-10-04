import os
import sys
from fcnn.utils.utils import load_json
import subprocess


def main():
    model_name = sys.argv[1]

    prediction_dir = "/work/aizenberg/dgellis/fCNN/predictions"
    smoothing_level = 4
    smoothing_name = "_s{}_".format(smoothing_level)
    task_name = "LANGUAGE"
    basename = "{subject_id}_tfMRI_{task_name}_level2_hp200_s{smoothing}_MSMAll.model_{model_name}_prediction.dscalar.nii"
    basename = basename.format(task_name=task_name, smoothing=smoothing_level, model_name=model_name,
                               subject_id="{subject_id}")
    prediction_filename = os.path.join(prediction_dir, basename)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    lang_config_fn = os.path.join(data_dir, "trial_lowq_32_LS_LM_config.json")
    lang_config = load_json(lang_config_fn)
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    task_filename = os.path.join(hcp_dir, "{subject_id}", "MNINonLinear/Results",
                                 "tfMRI_{task_name}/tfMRI_{task_name}_hp200_s{smoothing_level}_level2_MSMAll.feat",
                                 "{subject_id}_tfMRI_{task_name}_level2_hp200_s{smoothing_level}_MSMAll.dscalar.nii")

    for subject_id in lang_config['validation']:
        pred_dscalar_filename = prediction_filename.format(subject_id=subject_id)
        if not os.path.exists(pred_dscalar_filename):
            print("Does not exists:", pred_dscalar_filename)
            continue
        print(subject_id)
        fmri_dscalar_filename = task_filename.format(subject_id=subject_id,
                                                     task_name=task_name,
                                                     smoothing_level=smoothing_level)
        output_fn = pred_dscalar_filename.replace(".dscalar", "_error.dscalar")
        if not os.path.exists(output_fn):
            cmd_args = ["wb_command",
                        "-cifti-math", '"a - b"', output_fn,
                        "-var", '"a"', fmri_dscalar_filename,
                        "-var", '"b"', pred_dscalar_filename]
            print(" ".join(cmd_args))
            subprocess.call(cmd_args)


if __name__ == "__main__":
    main()

