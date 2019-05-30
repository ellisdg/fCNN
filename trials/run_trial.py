import sys
import os
import json
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fcnn.train import run_training
from fcnn.utils.sequences import WholeBrainRegressionSequence, HCPRegressionSequence
from fcnn.utils.utils import load_json


def generate_hcp_filenames(directory, surface_basename_template, target_basenames, feature_basenames, subject_ids,
                           hemispheres):
    rows = list()
    for subject_id in subject_ids:
        subject_dir = os.path.join(directory, subject_id)
        feature_filenames = [os.path.join(subject_dir, fbn) for fbn in feature_basenames]
        surface_filenames = [os.path.join(subject_dir,
                                          surface_basename_template.format(hemi=hemi, subject_id=subject_id))
                             for hemi in hemispheres]
        metric_filenames = [os.path.join(subject_dir, mbn.format(subject_id)) for mbn in target_basenames]
        rows.append([feature_filenames, surface_filenames, metric_filenames, subject_id])
    return rows


if __name__ == '__main__':
    config_filename = sys.argv[1]
    print("Config: ", config_filename)
    config = load_json(config_filename)

    model_filename = sys.argv[2]
    print("Model: ", model_filename)

    training_log_filename = sys.argv[3]
    print("Log: ", training_log_filename)

    system_config_filename = sys.argv[4]
    print("MP Config: ", system_config_filename)
    system_config = load_json(system_config_filename)

    for name in ("training", "validation"):
        key = name + "_filenames"
        if key not in config:
            config[key] = generate_hcp_filenames(system_config['directory'],
                                                 system_config['surface_basename_template'],
                                                 system_config['target_basenames'],
                                                 system_config['feature_basenames'],
                                                 system_config[name],
                                                 system_config['hemispheres'])
    if "directory" in system_config:
        system_config.pop("directory")

    if os.path.basename(config_filename).split("_")[1] == "wb":
        sequence_class = WholeBrainRegressionSequence
    else:
        sequence_class = HCPRegressionSequence

    run_training(config, model_filename, training_log_filename, sequence_class=sequence_class,
                 **system_config)
