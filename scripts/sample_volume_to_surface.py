import sys
import subprocess
import os
import json
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


def main():
    directory = sys.argv[1]
    config_filename = sys.argv[2]
    surface_template = sys.argv[3]
    method = sys.argv[4]
    volume_template = sys.argv[5]
    name_file = sys.argv[6]
    atlas_roi_template = sys.argv[7]
    surface_name = sys.argv[8]

    atlas_roi_template = os.path.join(directory, "{subject}", atlas_roi_template)
    if not os.path.isabs(volume_template):
        volume_template = os.path.join(directory, "{subject}", volume_template)
    surface_template = os.path.join(directory, "{subject}", surface_template)
    subject_ids = list()
    verbose = False
    overwrite = False

    with open(config_filename, "r") as op:
        data = json.load(op)
        for key in data:
            subject_ids.extend(data[key])
    for i, subject in enumerate(sorted(subject_ids)):
        update_progress(i/len(subject_ids), message=str(subject))
        left_surf = surface_template.format(subject=subject, hemi="L")
        right_surf = surface_template.format(subject=subject, hemi="R")
        left_atlas_roi = atlas_roi_template.format(subject=subject, hemi="L")
        right_atlas_roi = atlas_roi_template.format(subject=subject, hemi="R")
        vol = volume_template.format(subject=subject)
        filename = vol.replace(".nii.gz", ".{}.dscalar.nii".format(surface_name))
        if os.path.exists(vol) and (overwrite or not os.path.exists(filename)):
            for surf, hemi in ((left_surf, "L"), (right_surf, "R")):
                if not os.path.exists(surf):
                    download(os.path.join("s3://hcp-openaccess/{}/".format(os.path.basename(directory)),
                                          subject,
                                          os.path.dirname(surface_template)),
                             os.path.join(directory, subject, os.path.dirname(surface_template)),
                             verbose=verbose,
                             include=[os.path.basename(surf)])
            volume_to_surface(vol, left_surf, right_surf, filename, surface_name, method=method,
                              name_file=name_file, verbose=verbose,
                              right_atlas_roi=right_atlas_roi, left_atlas_roi=left_atlas_roi)
        elif not os.path.exists(vol):
            if verbose:
                print("File does not exist:", vol)
    update_progress(1)


if __name__ == "__main__":
    main()
