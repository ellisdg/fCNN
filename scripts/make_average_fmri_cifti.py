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
    make_average_cifti(config=config, directory=directory, subset=subset, output_directory=output_directory,
                       output_basename=output_basename)


def make_average_cifti(config, directory, output_directory, subset, output_basename,
                       replace=(".nii.gz", ".midthickness.dscalar.nii")):
    if subset not in config and "subjects_filename" in config:
        subjects_config = load_json(config["subjects_filename"])
        subject_ids = subjects_config[subset]
    else:
        subject_ids = config[subset]

    target_basenames = config["target_basenames"]
    if type(target_basenames) == str:
        target_basenames = [target_basenames]
    for target_basename in target_basenames:
        output_filename = os.path.join(output_directory,
                                       output_basename + os.path.basename(target_basename.format(subset)))
        cmd = ["wb_command", "-cifti-average", output_filename]
        for subject_id in subject_ids:
            subject_id = str(subject_id)
            cmd.append("-cifti")
            cifti_filename = os.path.join(directory, subject_id, target_basename.format(subject_id))
            if replace[0] in cifti_filename:
                cifti_filename = cifti_filename.replace(*replace)
            cmd.append(cifti_filename)
        print(" ".join(cmd))
        subprocess.call(cmd)


if __name__ == "__main__":
    main(sys.argv)
