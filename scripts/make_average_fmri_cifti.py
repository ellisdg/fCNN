import subprocess
import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_filename",
                        default="/home/aizenberg/dgellis/fCNN/v4_struct14_unet_ALL-TAVOR_2mm_v2_pt_config")
    parser.add_argument("--directory", default="/work/aizenberg/dgellis/HCP/HCP_1200")
    parser.add_argument("--output_directory", default="/work/aizenberg/dgellis/fCNN")
    parser.add_argument("--subset", default="test")
    parser.add_argument("--output_basename", default="")
    parser.add_argument("--target_basenames", nargs="+")
    parser.add_argument("--replace", nargs=2, default=["", ""])
    return parser.parse_args()


def load_json(filename):
    with open(filename) as opened_file:
        return json.load(opened_file)


def main():
    namespace = parse_args()
    config = load_json(namespace.config_filename)
    if namespace.subset not in config and "subjects_filename" in config:
        subjects_config = load_json(config["subjects_filename"])
        subject_ids = subjects_config[namespace.subset]
    else:
        subject_ids = config[namespace.subset]

    if namespace.target_basenames:
        target_basenames = namespace.target_basenames
    else:
        target_basenames = config["target_basenames"]
        if type(target_basenames) == str:
            target_basenames = [target_basenames]

    make_average_cifti(target_basenames=target_basenames, subject_ids=subject_ids,
                       directory=namespace.directory, subset=namespace.subset,
                       output_directory=namespace.output_directory, output_basename=namespace.output_basename,
                       replace=namespace.replace)


def make_average_cifti(target_basenames, subject_ids, directory, output_directory, subset, output_basename,
                       replace=(".nii.gz", ".midthickness.dscalar.nii")):

    for target_basename in target_basenames:
        output_filename = os.path.join(output_directory,
                                       output_basename + os.path.basename(target_basename.format(subset)))
        if replace[0] in output_filename:
            output_filename = output_filename.replace(*replace)
        cmd = ["wb_command", "-cifti-average", output_filename]
        for subject_id in subject_ids:
            subject_id = str(subject_id)
            cifti_filename = os.path.join(directory, subject_id, target_basename.format(subject_id))
            if replace[0] in cifti_filename:
                cifti_filename = cifti_filename.replace(*replace)
            if os.path.exists(cifti_filename):
                cmd.append("-cifti")
                cmd.append(cifti_filename)
        print(" ".join(cmd))
        print("Total files: ", (len(cmd) - 3)/2)
        subprocess.call(cmd)


if __name__ == "__main__":
    main()
