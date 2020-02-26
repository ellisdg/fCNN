import subprocess
import json
import sys
import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd


def load_json(filename):
    with open(filename) as opened_file:
        return json.load(opened_file)


def main(args):
    reference_volume = "/media/crane/tools/HCPpipelines/global/templates/MNI152_T1_2mm.nii.gz"
    prediction_directory = os.path.abspath(args[1])
    directory = os.path.abspath(args[2])
    config = load_json(args[3])
    dir_name = "MNINonLinear"
    overwrite = False
    transformed = transform_prediction_dir(prediction_dir=prediction_directory, directory=directory,
                                           reference_volume=reference_volume, dir_name=dir_name, overwrite=overwrite,
                                           target_basename=config["target_basenames"])
    output_filename = os.path.join(prediction_directory, dir_name, dir_name + "errors.csv")
    compute_errors(transformed, output_filename=output_filename, columns=config["metric_names"])


def compute_errors(transformed, output_filename, columns):
    errors = list()
    index = list()
    for subject_id, filename, target_filename in transformed:
        mae = compute_mean_absolute_error(filename, target_filename)
        errors.append(mae)
        index.append(subject_id)
    df = pd.DataFrame(errors, index=index, columns=columns)
    df.to_csv(output_filename)


def compute_mean_absolute_error(filename1, filename2, axis=(0, 1, 2)):
    image1 = nib.load(filename1)
    image2 = nib.load(filename2)
    np.testing.assert_equal(image1.affine, image2.affine)
    np.testing.assert_equal(image1.shape, image2.shape)
    error = image1.get_fdata() - image2.get_fdata()
    return np.mean(np.abs(error), axis=axis)


def run_command(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)


def transform_prediction_dir(prediction_dir, directory, reference_volume, target_basename, dir_name="MNINonLinear",
                             overwrite=False):
    output_dir = os.path.join(prediction_dir, dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    outputs = list()
    for prediction_volume in glob.glob(os.path.join(prediction_dir, "*.nii.gz")):
        basename = os.path.basename(prediction_volume)
        if "target" not in basename:
            subject_id = basename.split("_")[0]
            print(subject_id)
            warpfield = os.path.join(directory, subject_id, "MNINonLinear", "xfms", "acpc_dc2standard.nii.gz")
            output_filename = os.path.join(output_dir, basename)
            transformed_volume = transform_volume(prediction_volume, warpfield, reference_volume,
                                                  output_volume=output_filename, overwrite=overwrite)
            target_filename = os.path.join(directory, subject_id,
                                           target_basename.format(subject_id))
            transformed_target_filename = target_filename.replace("T1w", dir_name).replace(".volume.dscalar",
                                                                                           "_resampled")
            transform_volume(target_filename, warpfield, reference_volume, transformed_target_filename,
                             overwrite=overwrite)
            outputs.append((subject_id, transformed_volume, transformed_target_filename))
    return outputs


def transform_volume(nifti_volume, warpfield, reference_volume, output_volume, overwrite=False):
    resample_cmd = ["wb_command", "-volume-warpfield-resample", nifti_volume, warpfield, reference_volume,
                    "TRILINEAR", output_volume, "-fnirt", reference_volume]
    if overwrite or not os.path.exists(output_volume):
        run_command(resample_cmd)
    return output_volume


if __name__ == "__main__":
    main(sys.argv)
