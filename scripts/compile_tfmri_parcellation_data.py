import nibabel as nib
import os
import sys
import pandas as pd
from fcnn.utils.utils import load_json
from fcnn.utils.hcp import extract_parcellated_scalar_parcel_names
from fcnn.utils.hcp import extract_scalar_map


def main():
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    smoothing_levels = (2, 4)
    task_names = ("MOTOR", "LANGUAGE")
    training_fn = sys.argv[1]
    validation_fn = sys.argv[2]

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

    motor_config_fn = os.path.join(data_dir, "trial_lowq_34_MOTOR_config.json")
    lang_config_fn = os.path.join(data_dir, "trial_lowq_32_LS_LM_config.json")

    motor_config = load_json(motor_config_fn)
    lang_config = load_json(lang_config_fn)

    for subset, out_fn in (("training", training_fn),  ("validation", validation_fn)):
        data = dict(subject_id=list())
        for motor_id, lang_id in zip(motor_config[subset], lang_config[subset]):
            assert motor_id == lang_id
            subject_id = motor_id
            data["subject_id"].append(subject_id)
            results_dir = "{hcp_dir}/{subject_id}/MNINonLinear/Results".format(hcp_dir=hcp_dir, subject_id=subject_id)
            for smoothing_level in smoothing_levels:
                smoothing_name = "_s{}_".format(smoothing_level)
                for task_type, config in zip(task_names, (motor_config, lang_config)):
                    task_results_dir = "{results_dir}/tfMRI_{task_name}/tfMRI_{task_name}_hp200_s{smoothing_level}_level2_MSMAll.feat/".format(
                        results_dir=results_dir,
                        task_name=task_type,
                        smoothing_level=smoothing_level)
                    cifti_basename = "{task_results_dir}/{subject_id}_tfMRI_{task_name}_level2_hp200_s{smoothing_level}_MSMAll".format(
                        task_results_dir=task_results_dir,
                        subject_id=subject_id,
                        task_name=task_type,
                        smoothing_level=smoothing_level)
                    parcellated_filename = "{cifti_basename}.pscalar.nii".format(cifti_basename=cifti_basename)
                    print(parcellated_filename)
                    pscalar = nib.load(parcellated_filename)

                    parcel_names = extract_parcellated_scalar_parcel_names(pscalar)
                    for raw_target_name in config['metric_names']:
                        target_name = raw_target_name.format(subject_id)
                        if smoothing_name not in target_name:
                            target_name = target_name.replace(("_s2_", "_s4_")[[4, 2].index(smoothing_level)],
                                                              smoothing_name)
                        parcel_data = extract_scalar_map(pscalar, target_name)
                        task_name = parse_metric_name(target_name)
                        for index, parcel_name in enumerate(parcel_names):
                            column = "_".join((task_name, "s" + str(smoothing_level), parcel_name))
                            if column not in data:
                                data[column] = [parcel_data[index]]
                            else:
                                data[column].append(parcel_data[index])
        pd.DataFrame.from_dict(data).to_csv(out_fn)


def parse_metric_name(metric_name):
    split_name = metric_name.split("_")
    task_type = split_name[2]
    task_name = split_name[4]
    return "_".join((task_type, task_name))


if __name__ == "__main__":
    main()

