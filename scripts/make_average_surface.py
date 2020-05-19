import subprocess
import json
import sys
import os


def load_json(filename):
    with open(filename) as opened_file:
        return json.load(opened_file)


def main(args):
    subjects_config = load_json(args[1])
    directory = os.path.abspath(args[2])
    output_directory = os.path.abspath(args[3])
    subset = str(args[4])
    try:
        output_basename = str(args[5])
    except IndexError:
        output_basename = ""

    hemispheres = ["L", "R"]
    surface_basename_template = "{subject}.{hemi}.{surface}.32k_fs_LR.surf.gii"
    surface_template = os.path.join(directory, "{subject}", "MNINonLinear", "fsaverage_LR32k",
                                    surface_basename_template)
    surface_name = "inflated"

    if subset == "all":
        subject_ids = []
        for subset_subject_ids in subjects_config.values():
            subject_ids.extend(subset_subject_ids)
    else:
        subject_ids = subjects_config[subset]

    for surface_hemisphere in hemispheres:
        surface_basename = surface_basename_template.format(hemi=surface_hemisphere,
                                                            subject="{subject}",
                                                            surface=surface_name)

        output_filename = os.path.join(output_directory,
                                       output_basename + os.path.basename(surface_basename.format(subject=subset)))
        cmd = ["wb_command", "-surface-average", output_filename]
        for subject_id in subject_ids:
            subject_id = str(subject_id)
            surface_filename = surface_template.format(subject=subject_id, hemi=surface_hemisphere,
                                                       surface=surface_name)
            if os.path.exists(surface_filename):
                cmd.append("-surf")
                cmd.append(surface_filename)
        print(" ".join(cmd))
        subprocess.call(cmd)


if __name__ == "__main__":
    main(sys.argv)
