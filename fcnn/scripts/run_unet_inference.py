import os
import argparse
from fcnn.utils.utils import load_json, in_config
from fcnn.predict import volumetric_predictions
from fcnn.utils.filenames import generate_filenames, load_subject_ids, load_sequence


def format_parser(parser=argparse.ArgumentParser(), sub_command=False):
    parser.add_argument("--output_directory", required=True)
    if not sub_command:
        parser.add_argument("--config_filename", required=True)
        parser.add_argument("--model_filename", required=True)
        parser.add_argument("--machine_config_filename",
                            default="/home/aizenberg/dgellis/fCNN/data/hcc_v100_2gpu_32gb_config.json")
    parser.add_argument("--directory_template", help="Set this if directory template for running the predictions is "
                                                     "different from the directory used for training.")
    parser.add_argument("--group", default="test")
    parser.add_argument("--eval", default=False, action="store_true",
                        help="Scores the predictions according to the validation criteria and saves the results to a"
                             "csv file in the prediction directory.")
    parser.add_argument("--no_resample", default=False, action="store_true",
                        help="Skips resampling the predicted images into the non-cropped image space. This can help"
                             "save on the storage space as the images can always be resampled back into the original"
                             "space when needed.")
    parser.add_argument("--interpolation", default="linear")
    parser.add_argument("--output_template")
    parser.add_argument("--segmentation", action="store_true", default=False)
    parser.add_argument("--replace", nargs=2)
    parser.add_argument("--threshold", default=0.7, type=float,
                        help="If segmentation is set, this is the threshold for segmentation cutoff.")
    parser.add_argument("--no_sum", default=False, action="store_true",
                        help="Does not sum the predictions before using threshold.")
    parser.add_argument("--use_contours", action="store_true", default=False,
                        help="If the model was trained to predict contours you can use the contours to assist in the"
                             "segmentation. (This has not been shown to improve results.)")
    parser.add_argument("--subjects_config_filename",
                        help="Allows for specification of the config that contains the subject ids. If not set and the"
                             "subject ids are not listed in the main config, then the filename for the subjects config"
                             "will be read from the main config.")
    return parser


def parse_args():
    return format_parser().parse_args()


def main():
    namespace = parse_args()
    run_inference(namespace)


def run_inference(namespace):
    print("Config: ", namespace.config_filename)
    config = load_json(namespace.config_filename)
    key = namespace.group + "_filenames"
    
    if namespace.replace is not None:
        for _key in ("directory", "feature_templates", "target_templates"):
            if _key in config["generate_filenames_kwargs"]:
                if type(config["generate_filenames_kwargs"][_key]) == str:
                    config["generate_filenames_kwargs"][_key] = config["generate_filenames_kwargs"][_key].replace(
                        namespace.replace[0], namespace.replace[1])
                else:
                    config["generate_filenames_kwargs"][_key] = [template.replace(namespace.replace[0],
                                                                                  namespace.replace[1]) for template in
                                                                 config["generate_filenames_kwargs"][_key]]

    if namespace.directory_template is not None:
        config["generate_filenames_kwargs"]["directory"] = namespace.directory_template

    if key not in config:
        if namespace.subjects_config_filename:
            config[namespace.group] = load_json(namespace.subjects_config_filename)[namespace.group]
        filenames = generate_filenames(config, namespace.group, namespace.machine_config_filename,
                                       skip_targets=(not namespace.eval))
        if config["sequence"] == "multisource_templates":
            if "inputs_per_epoch" not in config:
                config["sequence_kwargs"]["inputs_per_epoch"] = dict()
            for dataset in filenames:
                config["sequence_kwargs"]["inputs_per_epoch"][dataset] = "all"

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

    labels = config["sequence_kwargs"]["labels"] if namespace.segmentation else None
    if in_config("add_contours", config["sequence_kwargs"], False):
        config["n_outputs"] = config["n_outputs"] * 2
        if namespace.use_contours:
            # this sets the labels for the contours
            labels = list(labels) + list(labels)

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
                                  metric_names=in_config("metric_names", config, None),
                                  evaluate_predictions=namespace.eval,
                                  resample_predictions=(not namespace.no_resample),
                                  interpolation=namespace.interpolation,
                                  output_template=namespace.output_template,
                                  segmentation=namespace.segmentation,
                                  segmentation_labels=labels,
                                  threshold=namespace.threshold,
                                  sum_then_threshold=(namespace.no_sum is False))


if __name__ == '__main__':
    main()
