import subprocess
import os
import json
import argparse
from fcnn.utils.utils import update_progress


def run_command(cmd, verbose=False):
    if verbose:
        print(" ".join(cmd))
    subprocess.call(cmd)


def volume_to_surface(volume, left_surface, right_surface, output_filename, surface_name, left_atlas_roi,
                      right_atlas_roi, method="enclosing", name_file=None, verbose=False):
    tmp_gifti_left = volume.split(".")[0] + "L.{}.func.gii".format(surface_name)
    left_cmd = ["wb_command", "-volume-to-surface-mapping", volume, left_surface, tmp_gifti_left, "-" + method]
    mask_left_cmd = ["wb_command", "-metric-mask", tmp_gifti_left, left_atlas_roi, tmp_gifti_left]
    tmp_gifti_right = volume.split(".")[0] + "R.{}.func.gii".format(surface_name)
    right_cmd = ["wb_command", "-volume-to-surface-mapping", volume, right_surface, tmp_gifti_right, "-" + method]
    mask_right_cmd = ["wb_command", "-metric-mask", tmp_gifti_right, right_atlas_roi, tmp_gifti_right]
    cmd = ["wb_command", "-cifti-create-dense-scalar", output_filename,
           "-left-metric", tmp_gifti_left,
           "-right-metric", tmp_gifti_right]
    if name_file is not None:
        cmd.extend(["-name-file", name_file])
    if method == "ribbon-constrained":
        surface_name = surface_name.split("_")[0]  # in case midthickness_MSMAll is the surface name
        right_cmd.append(right_surface.replace(surface_name, "white"))
        right_cmd.append(right_surface.replace(surface_name, "pial"))
        left_cmd.append(left_surface.replace(surface_name, "white"))
        left_cmd.append(left_surface.replace(surface_name, "pial"))
    run_command(right_cmd, verbose=verbose)
    run_command(mask_right_cmd, verbose=verbose)
    run_command(left_cmd, verbose=verbose)
    run_command(mask_left_cmd, verbose=verbose)
    run_command(cmd, verbose=verbose)
    os.remove(tmp_gifti_left)
    os.remove(tmp_gifti_right)


def download(f1, f2, include=None, exclude=None, verbose=False):
    cmd = ["aws", "s3", "sync", f1, f2]
    if include is not None:
        for inclusion in include:
            cmd.extend(["--include", inclusion])
    run_command(cmd, verbose=verbose)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--volume_template', required=True,
                        help="Filename with '{subject}' in place of the subject id.")
    parser.add_argument('--config', default="/home/aizenberg/dgellis/fCNN/data/subjects_v4-retest.json")
    parser.add_argument('--hcp_dir', default="/work/aizenberg/dgellis/HCP/HCP_1200")
    parser.add_argument('--surface_name', default="midthickness")
    parser.add_argument('--method', default="ribbon-constrained")
    parser.add_argument('--surface_template',
                        default="T1w/fsaverage_LR32k/{subject}.{hemi}.{surface_name}.32k_fs_LR.surf.gii")
    parser.add_argument('--atlas_roi_template',
                        default="MNINonLinear/fsaverage_LR32k/{subject}.{hemi}.atlasroi.32k_fs_LR.shape.gii")
    parser.add_argument('--task_names', default="/home/aizenberg/dgellis/fCNN/data/labels/ALL-TAVOR_name-file.txt")
    parser.add_argument('--overwrite', action="store_true", default=False)
    parser.add_argument('--verbose', action="store_true", default=False)
    return vars(parser.parse_args())


def main():
    args = parse_args()
    atlas_roi_template = os.path.join(args["hcp_dir"], "{subject}", args["atlas_roi_template"])
    surface_template = os.path.join(args["hcp_dir"], "{subject}", args["surface_template"])
    volume_template = args["volume_template"]
    if not os.path.isabs(volume_template):
        volume_template = os.path.join(args["hcp_dir"], "{subject}", volume_template)
    subject_ids = list()
    verbose = args["verbose"]
    overwrite = args["overwrite"]

    with open(args["config"], "r") as op:
        data = json.load(op)
        for key in data:
            subject_ids.extend(data[key])
    for i, subject in enumerate(sorted(subject_ids)):
        update_progress(i/len(subject_ids), message=str(subject))
        left_surf = surface_template.format(subject=subject, hemi="L", surface_name=args["surface_name"])
        right_surf = surface_template.format(subject=subject, hemi="R", surface_name=args["surface_name"])
        left_atlas_roi = atlas_roi_template.format(subject=subject, hemi="L")
        right_atlas_roi = atlas_roi_template.format(subject=subject, hemi="R")
        vol = volume_template.format(subject=subject)
        filename = vol.replace(".nii.gz", ".{}.dscalar.nii".format(args["surface_name"]))
        if os.path.exists(vol) and (overwrite or not os.path.exists(filename)):
            for surf, hemi in ((left_surf, "L"), (right_surf, "R")):
                if not os.path.exists(surf):
                    download(os.path.join("s3://hcp-openaccess/{}/".format(os.path.basename(args["hcp_dir"])),
                                          subject,
                                          os.path.dirname(surface_template)),
                             os.path.join(args["hcp_dir"], subject, os.path.dirname(surface_template)),
                             verbose=verbose,
                             include=[os.path.basename(surf)])
            volume_to_surface(vol, left_surf, right_surf, filename, args["surface_name"], method=args["method"],
                              name_file=args["task_names"], verbose=verbose,
                              right_atlas_roi=right_atlas_roi, left_atlas_roi=left_atlas_roi)
        elif not os.path.exists(vol):
            if verbose:
                print("File does not exist:", vol)
    update_progress(1)


if __name__ == "__main__":
    main()
