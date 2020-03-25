from fcnn.utils.utils import load_json
import sys
import os
import subprocess


def run_command(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)


def read_namefile(filename):
    with open(filename) as opened_file:
        names = list()
        for line in opened_file:
            names.append(line.strip())
    return names


def main():
    config = load_json(sys.argv[1])
    subjects = list()
    for key in config:
        subjects.extend(config[key])
    tasks = ["MOTOR", "LANGUAGE", "WM", "RELATIONAL", "EMOTION", "SOCIAL", "GAMBLING"]
    hcp_dir = "/work/aizenberg/dgellis/HCP/HCP_1200"
    name_filename_template = "/home/aizenberg/dgellis/fCNN/data/labels/{task}_name-file.txt"
    volume_template = os.path.join(hcp_dir, "{subject}", "T1w", "Results", "tfMRI_{task}",
                                   "tfMRI_{task}_hp200_s2_level2.feat",
                                   "{subject}_tfMRI_{task}_level2_zstat_hp200_s2_TAVOR.nii.gz")
    for subject in subjects:
        output_filename = volume_template.format(subject=subject, task="ALL")
        cmd = ["wb_command", "-volume-merge"]
        complete = True
        for task in tasks:
            volume_filename = volume_template.format(subject=subject, task=task)
            if not os.path.exists(volume_filename):
                complete = False
            cmd.extend(["-volume", volume_filename])
        if complete and not os.path.exists(output_filename):
            if not os.path.exists(os.path.dirname(output_filename)):
                os.makedirs(os.path.dirname(output_filename))
            run_command(cmd)

    name_filename = "/work/aizenberg/dgellis/TAVOR_name-file.txt"
    names = list()
    for task in tasks:
        names.extend(read_namefile(name_filename_template.format(task=task)))
    with open(name_filename, "w") as opened_file:
        opened_file.writelines(names)


if __name__ == "__main__":
    main()
