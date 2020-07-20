import os
import argparse
from fcnn.utils.utils import load_json
from fcnn.predict import volumetric_predictions
from fcnn.scripts.run_trial import load_subject_ids, load_sequence, generate_filenames


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename", required=True)
    parser.add_argument("--model_filename", required=True)
    parser.add_argument("--output_directory", required=True)
    parser.add_argument("--machine_config_filename",
                        default="/home/aizenberg/dgellis/fCNN/data/hcc_v100_1gpu_config.json")
    parser.add_argument("--group", default="test")
    parser.add_argument("--eval", default=False, action="store_true")
    return parser.parse_args()


def main():
    namespace = parse_args()
    print("Config: ", namespace.config_filename)
    config = load_json(namespace.config_filename)
    key = namespace.group + "_filenames"
    if key not in config:
        filenames = generate_filenames(config, namespace.group, namespace.machine_config_filename)
    else:
        filenames = config[key]

    print("Model: ", namespace.model_filename)

    print("Machine config: ", namespace.machine_config_filename)
    machine_config = load_json(namespace.machine_config_filename)

    print("Output Directory:", namespace.output_directory)

    if not os.path.exists(namespace.output_directory):
        os.makedirs(namespace.output_directory)

    load_subject_ids(config)

    if "evaluation_metric" in config and config["evaluation_metric"] is not None:
        criterion_name = config['evaluation_metric']
    else:
        criterion_name = config['loss']

    if "model_kwargs" in config:
        model_kwargs = config["model_kwargs"]
    else:
        model_kwargs = dict()

    if "sequence_kwargs" in config:
        sequence_kwargs = config["sequence_kwargs"]
        # make sure any augmentations are set to None
        for key in ["augment_scale_std", "additive_noise_std"]:
            if key in sequence_kwargs:
                sequence_kwargs[key] = None
    else:
        sequence_kwargs = dict()

    if "sequence" in config:
        sequence = load_sequence(config["sequence"])
    else:
        sequence = None

    return volumetric_predictions(model_filename=namespace.model_filename,
                                  filenames=filenames,
                                  output_dir=namespace.output_directory,
                                  model_name=config["model_name"],
                                  n_features=config["n_features"],
                                  window=config["window"],
                                  criterion_name=criterion_name,
                                  package=config['package'],
                                  n_gpus=machine_config['n_gpus'],
                                  batch_size=config['validation_batch_size'],
                                  n_workers=machine_config["n_workers"],
                                  model_kwargs=model_kwargs,
                                  sequence_kwargs=sequence_kwargs,
                                  sequence=sequence,
                                  n_outputs=config["n_outputs"],
                                  metric_names=config["metric_names"],
                                  evaluate_predictions=namespace.eval)


if __name__ == '__main__':
    main()