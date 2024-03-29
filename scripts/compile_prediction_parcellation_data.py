import nibabel as nib
import os
import sys
import pandas as pd
from fcnn.utils.utils import load_json
from fcnn.utils.hcp import extract_parcellated_scalar_parcel_names
from fcnn.utils.hcp import extract_cifti_scalar_data


def main():
    model_name = sys.argv[1]

    prediction_dir = "/work/aizenberg/dgellis/fCNN/predictions"
    smoothing_level = 4
    smoothing_name = "_s{}_".format(smoothing_level)
    task_name = "LANGUAGE"
    basename = "{subject_id}_tfMRI_{task_name}_level2_hp200_s{smoothing}_MSMAll.model_{model_name}_prediction.pscalar.nii"
    basename = basename.format(task_name=task_name, smoothing=smoothing_level, model_name=model_name,
                               subject_id="{subject_id}")
    prediction_filename = os.path.join(prediction_dir, basename)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    lang_config_fn = os.path.join(data_dir, "{}_config.json".format(model_name))
    lang_config = load_json(lang_config_fn)
    output_fn = os.path.join(prediction_dir, "{}_parcellation_data.csv".format(model_name))

    data = dict(subject_id=list())
    for subject_id in lang_config['validation']:
        parcellated_filename = prediction_filename.format(subject_id=subject_id)
        if not os.path.exists(parcellated_filename):
            print("Does not exist:", parcellated_filename)
            continue
        data["subject_id"].append(subject_id)
        pscalar = nib.load(parcellated_filename)

        parcel_names = extract_parcellated_scalar_parcel_names(pscalar)
        for raw_target_name in lang_config['metric_names'][0]:
            target_name = raw_target_name.format(subject_id)
            if smoothing_name not in target_name:
                target_name = target_name.replace(("_s2_", "_s4_")[[4, 2].index(smoothing_level)],
                                                  smoothing_name)
            parcel_data = extract_cifti_scalar_data(pscalar, target_name)
            task_name = parse_metric_name(target_name)
            for index, parcel_name in enumerate(parcel_names):
                column = "_".join((task_name, "s" + str(smoothing_level), parcel_name))
                if column not in data:
                    data[column] = [parcel_data[index]]
                else:
                    data[column].append(parcel_data[index])
    pd.DataFrame.from_dict(data).to_csv(output_fn)


def parse_metric_name(metric_name):
    split_name = metric_name.split("_")
    task_type = split_name[2]
    task_name = split_name[4]
    return "_".join((task_type, task_name))


if __name__ == "__main__":
    main()

