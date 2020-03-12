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
    task = sys.argv[1]
    # name_file = "/home/neuro-user/PycharmProjects/fCNN/data/labels/DTI_name-file.txt"
    fcnn_dir = "/home/neuro-user/PycharmProjects/fCNN"
    # name_file = fcnn_dir + "/data/labels/MOTOR-TAVOR_name-file.txt"
    # name_file = fcnn_dir + "/data/labels/LANGUAGE-TAVOR_name-file.txt"
    name_file = fcnn_dir + "/data/labels/{task}-TAVOR_name-file.txt".format(task=task)
    # method = "enclosing"
    method = "ribbon-constrained"
    directory = "/media/crane/HCP/HCP_1200"
    surface_template = "T1w/fsaverage_LR32k/{}.{}.{}.32k_fs_LR.surf.gii"
    # volume_template = "T1w/Diffusion/dti_12.nii.gz"
    # volume_template = os.path.join(directory, "{subject}", "T1w/Results/tfMRI_MOTOR/tfMRI_MOTOR_hp200_s2_level2.feat",
    #                                "{subject}_tfMRI_MOTOR_level2_zstat_hp200_s2_TAVOR.nii.gz")
    # volume_template = "/home/neuro-user/PycharmProjects/fCNN/trials/predictions/v4_struct6_unet_MOTOR-TAVOR_2mm_v1_pt/{subject}_model_v4_struct6_unet_MOTOR-TAVOR_2mm_v1_pt_struct6_normalized.nii.gz"
    # volume_template = "/home/neuro-user/PycharmProjects/fCNN/trials/predictions/v4_struct6_unet_LANGUAGE_2mm_v1_pt/{subject}_model_v4_struct6_unet_LANGUAGE_2mm_v1_pt_struct6_normalized.nii.gz"
    # volume_template = "/home/neuro-user/PycharmProjects/fCNN/trials/predictions/v4_struct6_unet_WM_2mm_v1_pt/{subject}_model_v4_struct6_unet_WM_2mm_v1_pt_struct6_normalized.nii.gz"
    volume_template = os.path.join(directory, "{subject}", "T1w/Results/tfMRI_{task}/tfMRI_{task}_hp200_s2_level2.feat",
                                   "{subject}_tfMRI_{task}_level2_zstat_hp200_s2_TAVOR.nii.gz")
    config_filename = fcnn_dir + "/data/subjects_v4.json"
    subject_ids = list()
    verbose = False
    overwrite = False
    surface_names = ["midthickness"]
    atlas_roi_template = os.path.join(directory, "{subject}", "MNINonLinear", "fsaverage_LR32k",
                                      "{subject}.{hemi}.atlasroi.32k_fs_LR.shape.gii")
    with open(config_filename, "r") as op:
        data = json.load(op)
        for key in data:
            subject_ids.extend(data[key])
    for i, subject in enumerate(sorted(subject_ids)):
        update_progress(i/len(subject_ids), message=str(subject))
        for surface_name in surface_names:
            left_surf = os.path.join(directory, subject, surface_template.format(subject, "L", surface_name))
            right_surf = os.path.join(directory, subject, surface_template.format(subject, "R", surface_name))
            left_atlas_roi = atlas_roi_template.format(subject=subject, hemi="L")
            right_atlas_roi = atlas_roi_template.format(subject=subject, hemi="R")
            for surf, hemi in ((left_surf, "L"), (right_surf, "R")):
                if not os.path.exists(surf):
                    download(os.path.join("s3://hcp-openaccess/HCP_1200/", subject,
                                          os.path.dirname(surface_template)),
                             os.path.join(directory, subject, os.path.dirname(surface_template)),
                             verbose=verbose,
                             include=[os.path.basename(surf)])
            # vol = os.path.join(directory, subject, volume_template)
            # if "{subject}" in vol:
            #    vol = vol.format(subject=subject)
            vol = volume_template.format(subject=subject, task=task)
            filename = vol.replace(".nii.gz", ".{}.dscalar.nii".format(surface_name))
            if os.path.exists(vol) and (overwrite or not os.path.exists(filename)):
                volume_to_surface(vol, left_surf, right_surf, filename, surface_name, method=method,
                                  name_file=name_file, verbose=verbose,
                                  right_atlas_roi=right_atlas_roi, left_atlas_roi=left_atlas_roi)
            elif not os.path.exists(vol):
                if verbose:
                    print("File does not exist:", vol)
    update_progress(1)


if __name__ == "__main__":
    main()