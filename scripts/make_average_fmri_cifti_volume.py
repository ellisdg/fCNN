import subprocess
import json
import sys
import os


def load_json(filename):
    with open(filename) as opened_file:
        return json.load(opened_file)


def main(args):
    config = load_json(args[1])
    directory = os.path.abspath(args[2])
    output_directory = os.path.abspath(args[3])
    subset = str(args[4])
    try:
        output_basename = str(args[5])
    except IndexError:
        output_basename = ""
    make_average_cifti_volume(config=config, directory=directory, subset=subset, output_directory=output_directory,
                              output_basename=output_basename)


def run_command(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)


def make_average_cifti_volume(config, directory, output_directory, subset, output_basename, reference_volume=None):
    if reference_volume is None:
        reference_volume = "/work/aizenberg/dgellis/tools/HCPpipelines/global/templates/MNI152_T1_2mm.nii.gz"
    if subset not in config and "subjects_filename" in config:
        if os.path.exists(config["subjects_filename"]):
            subjects_config_filename = config["subjects_filename"]
        else:
            subjects_config_filename = os.path.join(os.path.dirname(__file__), "..", config["subjects_filename"])
        subjects_config = load_json(subjects_config_filename)
        subject_ids = subjects_config[subset]
    else:
        subject_ids = config[subset]

    target_basenames = config["target_basenames"]
    if type(target_basenames) == str:
        target_basenames = [target_basenames]
    for target_basename in target_basenames:
        make_average_cifti_volume_for_target(target_basename, output_directory, output_basename, subset, subject_ids,
                                             directory, reference_volume)


def make_average_cifti_volume_for_target(target_basename, output_directory, output_basename, subset, subject_ids,
                                         directory, reference_volume):
    output_filename = os.path.join(output_directory,
                                   output_basename + os.path.basename(target_basename.format(subset)))
    cifti_avg_cmd = ["wb_command", "-cifti-average", output_filename]
    for subject_id in subject_ids:
        subject_id = str(subject_id)

        cifti_resample_cmd = ["wb_command", "-volume-warpfield-resample"]
        moving_volume = os.path.join(directory, subject_id, target_basename.format(subject_id))
        warpfield = os.path.join(directory, subject_id, "MNINonLinear", "xfms", "acpc_dc2standard.nii.gz")
        output_volume = moving_volume.replace("T1w", "MNINonLinear").replace(".volume", "_resampled.volume")
        cifti_resample_cmd.extend([moving_volume, warpfield, reference_volume, "TRILINEAR", output_volume,
                                   "-fnirt", reference_volume])
        run_command(cifti_resample_cmd)
        cifti_avg_cmd.append("-cifti")
        cifti_avg_cmd.append(output_volume)
    run_command(cifti_avg_cmd)


if __name__ == "__main__":
    main(sys.argv)
