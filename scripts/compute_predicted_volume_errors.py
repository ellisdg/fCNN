import subprocess
import json
import sys
import os
import glob
import numpy as np
import nibabel as nib
import pandas as pd
from multiprocessing import Pool
from functools import partial
from scipy.stats import pearsonr


def load_json(filename):
    with open(filename) as opened_file:
        return json.load(opened_file)


def main(args):
    reference_volume = "/work/aizenberg/tools/HCPpipelines/global/templates/MNI152_T1_2mm.nii.gz"
    prediction_directory = os.path.abspath(args[1])
    directory = os.path.abspath(args[2])
    config = load_json(args[3])
    n_threads = int(args[4])
    dir_name = "MNINonLinear"
    overwrite = True
    transformed = transform_prediction_dir(prediction_dir=prediction_directory, directory=directory,
                                           reference_volume=reference_volume, dir_name=dir_name, overwrite=overwrite,
                                           target_basename=config["target_basenames"])
    output_filename = os.path.join(prediction_directory, dir_name, dir_name + "errors.csv")
    compute_errors(transformed, output_filename=output_filename, columns=config["metric_names"], pool_size=n_threads)
    compute_correlations(transformed, output_file=os.path.join(prediction_directory, dir_name, dir_name +
                                                               "correlations.npy"),
                         metric_names=config["metric_names"], pool_size=n_threads, verbose=True)


def compute_errors(transformed, output_filename, columns, pool_size=1, axis=(0, 1, 2)):
    if pool_size > 1:
        index = list()
        filenames = list()
        for subject_id, filename, target_filename in transformed:
            filenames.append((filename, target_filename))
            index.append(subject_id)
        pool = Pool(pool_size)
        func = partial(mp_compute_meant_absolute_error, axis=axis)
        errors = pool.map(func, filenames)
    else:
        errors = list()
        index = list()
        for subject_id, filename, target_filename in transformed:
            mae = compute_mean_absolute_error(filename, target_filename, axis=axis)
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


def mp_compute_meant_absolute_error(filenames, axis=(0, 1, 2)):
    return compute_mean_absolute_error(filenames[0], filenames[1], axis=axis)


def run_command(cmd):
    print(" ".join(cmd))
    subprocess.call(cmd)


def transform_prediction_dir(prediction_dir, directory, reference_volume, target_basename, dir_name="MNINonLinear",
                             overwrite=False, pool_size=1):
    output_dir = os.path.join(prediction_dir, dir_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prediction_volumes = glob.glob(os.path.join(prediction_dir, "*.nii.gz"))
    if pool_size > 1:
        pool = Pool(pool_size)
        func = partial(transform_prediction_target_volumes, hcp_directory=directory, output_dir=output_dir,
                       reference_volume_filename=reference_volume, target_basename_template=target_basename,
                       overwrite=overwrite)
        outputs = pool.map(func, prediction_volumes)
    else:
        outputs = list()
        for prediction_volume in prediction_volumes:
            outputs.append(transform_prediction_target_volumes(prediction_volume=prediction_volume,
                                                               hcp_directory=directory,
                                                               output_dir=output_dir,
                                                               reference_volume_filename=reference_volume,
                                                               target_basename_template=target_basename,
                                                               overwrite=overwrite))
    while None in outputs:
        outputs.remove(None)
    return outputs


def transform_prediction_target_volumes(prediction_volume, hcp_directory, output_dir, reference_volume_filename,
                                        target_basename_template, overwrite=False):
    basename = os.path.basename(prediction_volume)
    if "target" not in basename and "dscalar" not in basename:
        subject_id = basename.split("_")[0]
        print(subject_id)
        warpfield = os.path.join(hcp_directory, subject_id, "MNINonLinear", "xfms", "acpc_dc2standard.nii.gz")
        output_filename = os.path.join(output_dir, basename)
        transformed_volume_filename = transform_volume(prediction_volume, warpfield, reference_volume_filename,
                                                       output_volume=output_filename, overwrite=overwrite)
        target_filename = os.path.join(hcp_directory, subject_id,
                                       target_basename_template.format(subject_id))
        transformed_target_filename = target_filename.replace("T1w/", "MNINonLinear/").replace(".nii",
                                                                                               "_resampled.nii")
        target_dirname = os.path.dirname(transformed_target_filename)
        if not os.path.exists(target_dirname):
            os.makedirs(target_dirname)
        transform_volume(target_filename, warpfield, reference_volume_filename, transformed_target_filename,
                         overwrite=overwrite)
        return subject_id, transformed_volume_filename, transformed_target_filename


def transform_volume(nifti_volume, warpfield, reference_volume, output_volume, overwrite=False):
    resample_cmd = ["wb_command", "-volume-warpfield-resample", nifti_volume, warpfield, reference_volume,
                    "TRILINEAR", output_volume, "-fnirt", reference_volume]
    if overwrite or not os.path.exists(output_volume):
        run_command(resample_cmd)
    return output_volume


def compute_correlations(inputs, output_file, metric_names, pool_size=1, verbose=True):
    subjects = list()
    prediction_images = list()
    target_images = list()
    for subject_id, filename, target_filename in inputs:
        prediction_images.append(filename)
        target_images.append(target_filename)
        subjects.append(subject_id)
    correlations = list()
    if pool_size is not None:
        func = partial(compute_correlation_row, target_fns=target_images, metric_names=metric_names, verbose=verbose)
        pool = Pool(pool_size)
        correlations = pool.map(func, prediction_images)
    else:
        for i, p_image_fn in enumerate(prediction_images):
            correlations.append(compute_correlation_row(p_image_fn, target_images, metric_names, verbose=verbose))
    np.save(output_file, correlations)
    np.save(output_file.replace(".npy", "_subjects.npy"), subjects)


def compute_correlation_row(predicted_fn, target_fns, metric_names, verbose=False):
    if verbose:
        print(predicted_fn)
    predicted_image = nib.load(predicted_fn)
    predicted_data = predicted_image.get_fdata()
    row = list()
    for fn in target_fns:
        row.append(compute_correlation(target_fn=fn, predicted_data=predicted_data, metric_names=metric_names))
    return row


def compute_correlation(target_fn, predicted_data, metric_names):
    target_image = nib.load(target_fn)
    target_data = target_image.get_fdata()
    task_row = list()
    for i, task_name in enumerate(metric_names):
        task_row.append(pearsonr(predicted_data[..., i].flatten(), target_data[..., i].flatten()))
    return task_row


if __name__ == "__main__":
    main(sys.argv)
