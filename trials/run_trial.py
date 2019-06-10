import sys
import os
from functools import partial, update_wrapper
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from fcnn.train import run_training
from fcnn.utils.sequences import WholeBrainRegressionSequence, HCPRegressionSequence
from fcnn.utils.utils import load_json
from fcnn.utils.custom import get_metric_data_from_config
from fcnn.resnet import compare_scores


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def generate_hcp_filenames(directory, surface_basename_template, target_basenames, feature_basenames, subject_ids,
                           hemispheres):
    rows = list()
    for subject_id in subject_ids:
        subject_dir = os.path.join(directory, subject_id)
        if type(feature_basenames) == str:
            feature_filenames = os.path.join(subject_dir, feature_basenames)
        else:
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

    try:
        group_average_filenames = str(sys.argv[5])
        group_average = get_metric_data_from_config(group_average_filenames, config_filename)
        model_metrics = [wrapped_partial(compare_scores, comparison=group_average)]
        metric_to_monitor = "compare_scores"
    except IndexError:
        model_metrics = []
        metric_to_monitor = "loss"

    for name in ("training", "validation"):
        key = name + "_filenames"
        if key not in config:
            config[key] = generate_hcp_filenames(system_config['directory'],
                                                 config['surface_basename_template'],
                                                 config['target_basenames'],
                                                 config['feature_basenames'],
                                                 config[name],
                                                 config['hemispheres'])
    if "directory" in system_config:
        system_config.pop("directory")

    if os.path.basename(config_filename).split("_")[1] == "wb":
        sequence_class = WholeBrainRegressionSequence
    else:
        sequence_class = HCPRegressionSequence

    run_training(config, model_filename, training_log_filename, sequence_class=sequence_class,
                 model_metrics=model_metrics, metric_to_monitor=metric_to_monitor, **system_config)
