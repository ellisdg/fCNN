import subprocess
import os
import json
import sys


def update_progress(progress, bar_length=30, message=""):
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(bar_length * progress))
    text = "\r{0}[{1}] {2:.2f}% {3}".format(message, "#" * block + "-" * (bar_length - block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()


def run_command(cmd, verbose=False):
    if verbose:
        print(" ".join(cmd))
    subprocess.call(cmd)


def volume_to_surface(volume, left_surface, right_surface, output_filename, surface_name, method="enclosing",
                      name_file=None, verbose=False):
    tmp_gifti_left = volume.split(".")[0] + "L.{}.func.gii".format(surface_name)
    cmd = ["wb_command", "-volume-to-surface-mapping", volume, left_surface, tmp_gifti_left, "-" + method]
    run_command(cmd, verbose=verbose)
    tmp_gifti_right = volume.split(".")[0] + "R.{}.func.gii".format(surface_name)
    cmd = ["wb_command", "-volume-to-surface-mapping", volume, right_surface, tmp_gifti_right, "-" + method]
    run_command(cmd, verbose=verbose)
    cmd = ["wb_command", "-cifti-create-dense-scalar", output_filename,
           "-left-metric", tmp_gifti_left,
           "-right-metric", tmp_gifti_right]
    if name_file is not None:
        cmd.extend(["-name-file", name_file])
    run_command(cmd, verbose=verbose)
    os.remove(tmp_gifti_left)
    os.remove(tmp_gifti_right)


def download(f1, f2, include=None, exclude=None, verbose=False):
    cmd = ["aws", "s3", "sync", f1, f2]
    if include is not None:
        for inclusion in include:
            cmd.extend(["--include", inclusion])
    run_command(cmd, verbose=verbose)


def main():
    name_file = "/home/neuro-user/PycharmProjects/fCNN/scripts/DTI_names.txt"
    method = "enclosing"
    directory = "/media/crane/HCP/HCP_1200"
    surface_template = "T1w/fsaverage_LR32k/{}.{}.{}.32k_fs_LR.surf.gii"
    volume_template = "T1w/Diffusion/dti_12.nii.gz"
    config_filename = "/home/neuro-user/PycharmProjects/fCNN/data/subjects_v4.json"
    subject_ids = list()
    verbose = False
    overwrite = False
    with open(config_filename, "r") as op:
        data = json.load(op)
        for key in data:
            subject_ids.extend(data[key])
    for i, subject in enumerate(sorted(subject_ids)):
        for surface_name in ["white", "midthickness"]:
            left_surf = os.path.join(directory, subject, surface_template.format(subject, "L", surface_name))
            right_surf = os.path.join(directory, subject, surface_template.format(subject, "R", surface_name))
            for surf, hemi in ((left_surf, "L"), (right_surf, "R")):
                if not os.path.exists(surf):
                    download(os.path.join("s3://hcp-openaccess/HCP_1200/", subject,
                                          os.path.dirname(surface_template)),
                             os.path.join(directory, subject, os.path.dirname(surface_template)),
                             verbose=verbose,
                             include=[os.path.basename(surf)])
            vol = os.path.join(directory, subject, volume_template)
            filename = vol.replace(".nii.gz", ".{}.dscalar.nii".format(surface_name))
            if os.path.exists(vol) and (overwrite or not os.path.exists(filename)):
                volume_to_surface(vol, left_surf, right_surf, filename, surface_name, method, name_file, verbose=verbose)
        update_progress((i+1)/len(subject_ids))


if __name__ == "__main__":
    main()
