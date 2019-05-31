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
    subset = "training"
    subject_ids = config[subset]
    for target_basename in config["target_basenames"]:
        output_filename = os.path.join(output_directory,
                                       os.path.basename(target_basename.format(subset)))
        cmd = ["wb_command", "-cifti-average", output_filename]
        for subject_id in subject_ids:
            cmd.append("-cifti")
            cmd.append(os.path.join(directory, subject_id, target_basename.format(subject_id)))
        print(" ".join(cmd))
        subprocess.call(cmd)


if __name__ == "__main__":
    main(sys.argv)
