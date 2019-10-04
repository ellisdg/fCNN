import nibabel as nib
import os
import sys
import pandas as pd
from fcnn.utils.utils import load_json
from fcnn.utils.hcp import extract_cifti_scalar_data
import numpy as np


def main():
    smoothing_level = sys.argv[1]
    smoothing_name = "_s{}_".format(smoothing_level)
    prediction_dir = "/work/aizenberg/dgellis/fCNN/predictions"
    output_fn = os.path.join(prediction_dir, "validation_s{}_average_error.csv".format(smoothing_level))
    task_name = sys.argv[2]
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    lang_config_fn = os.path.join(data_dir, "trial_lowq_32_LS_LM_config.json")
    lang_config = load_json(lang_config_fn)
    brain_structures = ("CortexLeft", "CortexRight")
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    task_filename = os.path.join(hcp_dir, "{subject_id}", "MNINonLinear/Results",
                                 "tfMRI_{task_name}/tfMRI_{task_name}_hp200_s{smoothing_level}_level2_MSMAll.feat",
                                 "{subject_id}_tfMRI_{task_name}_level2_hp200_s{smoothing_level}_MSMAll.dscalar.nii")
    average_map_basenames = ("tfMRI_LANGUAGE_hp200_s4_level2_validation_MSMAll.dscalar.nii",
                             "tfMRI_LANGUAGE_hp200_s4_level2_validation_female_MSMAll.dscalar.nii",
                             "tfMRI_LANGUAGE_hp200_s4_level2_validation_male_MSMAll.dscalar.nii")
    average_map_filenames = (os.path.join(hcp_dir, basename) for basename in average_map_basenames)
    average_maps = [nib.load(filename) for filename in average_map_filenames]
    header = [bn.split('.')[0] for bn in average_map_basenames]
    header.append(model_name)
    header.insert(0, 'subject_id')
    average_subject_ids = ('198855', '173940', '198855')

    data = list()
    for subject_id in lang_config['validation']:
        pred_dscalar_filename = prediction_filename.format(subject_id=subject_id)
        if not os.path.exists(pred_dscalar_filename):
            print("Does not exists:", pred_dscalar_filename)
            continue
        print(subject_id)
        pred_dscalar = nib.load(pred_dscalar_filename)
        fmri_dscalar_filename = task_filename.format(subject_id=subject_id,
                                                     task_name=task_name,
                                                     smoothing_level=smoothing_level)
        fmri_dscalar = nib.load(fmri_dscalar_filename)
        fmri_bmaxis = fmri_dscalar.header.get_axis(1)
        pred_bmaxis = pred_dscalar.header.get_axis(1)
        subject_mae = list()
        for metric_name in lang_config['metric_names']:
            print(metric_name)
            subject_metric_name = metric_name.format(subject_id)
            metric_mae = list()
            for i in range(len(header) - 1):
                metric_mae.append(list())
            for brain_structure in brain_structures:
                print(brain_structure)
                pred_metric_data = extract_cifti_scalar_data(pred_dscalar, subject_metric_name,
                                                             brain_structure_name=brain_structure)
                fmri_metric_data = extract_cifti_scalar_data(fmri_dscalar, subject_metric_name,
                                                             brain_structure_name=brain_structure)
                pred_mask = np.in1d(pred_bmaxis.vertex[pred_bmaxis.name == pred_bmaxis.to_cifti_brain_structure_name(brain_structure)], 
                                    fmri_bmaxis.vertex[fmri_bmaxis.name == fmri_bmaxis.to_cifti_brain_structure_name(brain_structure)])

                all_metric_data = [extract_cifti_scalar_data(average_map,
                                                             metric_name.format(key),
                                                             brain_structure_name=brain_structure)
                                   for average_map, key in zip(average_maps, average_subject_ids)] + [pred_metric_data[pred_mask]]
                structure_mae = [np.abs(fmri_metric_data - metric_data) for metric_data in all_metric_data]
                for i, mae in enumerate(structure_mae):
                    print(np.asarray(mae).shape)
                    metric_mae[i].extend(list(mae))
                print(np.asarray(metric_mae).shape)
            print(np.asarray(metric_mae).shape)
            subject_mae.append(metric_mae)
        print(np.asarray(subject_mae).shape)
        data.append([subject_id] + list(np.mean(subject_mae, axis=(0, 2))))
    pd.DataFrame(data, columns=header).set_index("subject_id").to_csv(output_fn)


if __name__ == "__main__":
    main()

