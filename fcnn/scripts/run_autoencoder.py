import sys
import os
from fcnn.utils.utils import load_json
from fcnn.predict import whole_brain_autoencoder_predictions
from fcnn.scripts.run_trial import load_subject_ids, load_sequence


def main(config_filename, model_filename, machine_config_filename, output_directory, subject_key="validation"):
    print("Config: ", config_filename)
    config = load_json(config_filename)
    print("Model: ", model_filename)

    print("Machine config: ", machine_config_filename)
    machine_config = load_json(machine_config_filename)

    print("Output Directory:", output_directory)

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

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
        for key in ["augment_scale_std", "additive_noise_std"]:
            if key in sequence_kwargs:
                sequence_kwargs[key] = None
    else:
        sequence_kwargs = dict()

    if "sequence" in config:
        sequence = load_sequence(config["sequence"])
    else:
        sequence = None

    return whole_brain_autoencoder_predictions(model_filename=model_filename,
                                               subject_ids=config[subject_key],
                                               hcp_dir=machine_config["directory"],
                                               output_dir=output_directory,
                                               feature_basenames=config["feature_basenames"],
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
                                               target_basenames=config["target_basenames"],
                                               metric_names=config["metric_names"])


if __name__ == '__main__':
    main(config_filename=sys.argv[1], model_filename=sys.argv[2], machine_config_filename=sys.argv[3],
         output_directory=sys.argv[4], subject_key=sys.argv[5])
