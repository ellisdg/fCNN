import sys
import os
from functools import partial, update_wrapper
import pandas as pd
fcnn_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname((__file__)))))
sys.path.append(fcnn_path)
from fcnn.train import run_training
from fcnn.utils import sequences as keras_sequences
from fcnn.utils.sequences import (WholeBrainRegressionSequence, HCPRegressionSequence, ParcelBasedSequence,
                                  WindowedAutoEncoder)
from fcnn.utils.pytorch import dataset as pytorch_datasets
from fcnn.utils.pytorch.dataset import (WholeBrainCIFTI2DenseScalarDataset, HCPRegressionDataset, AEDataset,
                                        LabeledAEDataset, WindowedAEDataset)
from fcnn.utils.utils import load_json, load_image
from fcnn.utils.custom import get_metric_data_from_config
from fcnn.models.keras.resnet.resnet import compare_scores


def wrapped_partial(func, *args, **kwargs):
    partial_func = partial(func, *args, **kwargs)
    update_wrapper(partial_func, func)
    return partial_func


def generate_hcp_filenames(directory, surface_basename_template, target_basenames, feature_basenames, subject_ids,
                           hemispheres):
    rows = list()
    for subject_id in subject_ids:
        subject_id = str(subject_id)
        subject_dir = os.path.join(directory, subject_id)
        if type(feature_basenames) == str:
            feature_filenames = os.path.join(subject_dir, feature_basenames)
            if not os.path.exists(feature_filenames):
                continue
        else:
            feature_filenames = [os.path.join(subject_dir, fbn) for fbn in feature_basenames]
        if surface_basename_template is not None:
            surface_filenames = [os.path.join(subject_dir,
                                              surface_basename_template.format(hemi=hemi, subject_id=subject_id))
                                 for hemi in hemispheres]
        else:
            surface_filenames = None
        if type(target_basenames) == str:
            metric_filenames = os.path.join(subject_dir, target_basenames)
            if "{}" in metric_filenames:
                metric_filenames = metric_filenames.format(subject_id)
            if not os.path.exists(metric_filenames):
                continue
        elif target_basenames is not None:
            metric_filenames = [os.path.join(subject_dir, mbn.format(subject_id)) for mbn in target_basenames]
        else:
            metric_filenames = None
        rows.append([feature_filenames, surface_filenames, metric_filenames, subject_id])
    return rows


def generate_paired_filenames(directory, subject_ids, group, keys, basename, raise_if_not_exist=False):
    rows = list()
    pair = keys["all"]
    pair_key = list(keys["all"].keys())[0]
    volume_numbers = dict()
    for subject_id in subject_ids:
        subject_id = str(subject_id)
        template = os.path.join(directory, subject_id, basename)
        for key in keys[group]:
            for value in keys[group][key]:
                format_kwargs1 = {key: value, pair_key: pair[pair_key][0]}
                format_kwargs2 = {key: value, pair_key: pair[pair_key][1]}
                filename1 = template.format(**format_kwargs1)
                filename2 = template.format(**format_kwargs2)
                if os.path.exists(filename1) and os.path.exists(filename2):
                    if value not in volume_numbers:
                        volume_numbers[value] = range(load_image(filename1, force_4d=True).shape[-1])
                    for volume_number in volume_numbers[value]:
                        rows.append([filename1, [volume_number], filename2, [volume_number]])
                        rows.append([filename2, [volume_number], filename1, [volume_number]])
                elif raise_if_not_exist:
                    for filename in (filename1, filename2):
                        raise FileNotFoundError(filename)
    return rows


def load_subject_ids(config):
    if "subjects_filename" in config:
        subjects = load_json(os.path.join(fcnn_path, config["subjects_filename"]))
        for key, value in subjects.items():
            config[key] = value


def load_bias(bias_filename):
    import numpy as np
    return np.fromfile(os.path.join(fcnn_path, bias_filename))


def load_sequence(sequence_name):
    try:
        sequence_class = getattr(keras_sequences, sequence_name)
    except AttributeError as error:
        sequence_class = getattr(pytorch_datasets, sequence_name)
    return sequence_class


def main():
    import nibabel as nib
    nib.imageglobals.logger.level = 40

    config_filename = sys.argv[1]
    print("Config: ", config_filename)
    config = load_json(config_filename)
    if "package" in config:
        package = config["package"]
    else:
        package = "keras"

    if "metric_names" in config and not config["n_outputs"] == len(config["metric_names"]):
        raise ValueError("n_outputs set to {}, but number of metrics is {}.".format(config["n_outputs"],
                                                                                    len(config["metric_names"])))

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
        if config['skip_validation']:
            metric_to_monitor = "loss"
        else:
            metric_to_monitor = "val_loss"

    load_subject_ids(config)

    for name in ("training", "validation"):
        key = name + "_filenames"
        if key not in config:
            if "generate_filenames" not in config or config["generate_filenames"] == "classic":
                config[key] = generate_hcp_filenames(system_config['directory'],
                                                     config['surface_basename_template'] if "surface_basename_template" in config else None,
                                                     config['target_basenames'],
                                                     config['feature_basenames'],
                                                     config[name],
                                                     config['hemispheres'] if 'hemispheres' in config else None)
            elif config["generate_filenames"] == "paired":
                config[key] = generate_paired_filenames(system_config['directory'],
                                                        config[name],
                                                        name,
                                                        **config["generate_filenames_kwargs"])
    if "directory" in system_config:
        directory = system_config.pop("directory")
    else:
        directory = "."

    if "sequence" in config:
        sequence_class = load_sequence(config["sequence"])
    elif "_wb_" in os.path.basename(config_filename):
        if "package" in config and config["package"] == "pytorch":
            if config["sequence"] == "AEDataset":
                sequence_class = AEDataset
            elif config["sequence"] == "LabeledAEDataset":
                sequence_class = LabeledAEDataset
            else:
                sequence_class = WholeBrainCIFTI2DenseScalarDataset
        else:
            sequence_class = WholeBrainRegressionSequence
    elif config["sequence"] == "WindowedAutoEncoder":
        sequence_class = WindowedAutoEncoder
    elif config["sequence"] == "WindowedAEDataset":
        sequence_class = WindowedAEDataset
    elif "_pb_" in os.path.basename(config_filename):
        sequence_class = ParcelBasedSequence
        config["sequence_kwargs"]["parcellation_template"] = os.path.join(
            directory, config["sequence_kwargs"]["parcellation_template"])
    else:
        if config["package"] == "pytorch":
            sequence_class = HCPRegressionDataset
        else:
            sequence_class = HCPRegressionSequence

    if "bias_filename" in config and config["bias_filename"] is not None:
        bias = load_bias(config["bias_filename"])
    else:
        bias = None

    if sequence_class == ParcelBasedSequence:
        target_parcels = config["sequence_kwargs"].pop("target_parcels")
        for target_parcel in target_parcels:
            config["sequence_kwargs"]["target_parcel"] = target_parcel
            print("Training on parcel: {}".format(target_parcel))
            if type(target_parcel) == list:
                parcel_id = "-".join([str(i) for i in target_parcel])
            else:
                parcel_id = str(target_parcel)
            _training_log_filename = training_log_filename.replace(".csv", "_{}.csv".format(parcel_id))
            if os.path.exists(_training_log_filename):
                _training_log = pd.read_csv(_training_log_filename)
                if (_training_log[metric_to_monitor].values.argmin()
                        <= len(_training_log) - int(config["early_stopping_patience"])):
                    print("Already trained")
                    continue
            run_training(package,
                         config,
                         model_filename.replace(".h5", "_{}.h5".format(parcel_id)),
                         _training_log_filename,
                         sequence_class=sequence_class,
                         model_metrics=model_metrics,
                         metric_to_monitor=metric_to_monitor,
                         **system_config)

    else:
        run_training(package, config, model_filename, training_log_filename, sequence_class=sequence_class,
                     model_metrics=model_metrics, metric_to_monitor=metric_to_monitor, bias=bias, **system_config)


if __name__ == '__main__':
    main()
